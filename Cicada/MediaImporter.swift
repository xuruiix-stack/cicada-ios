import Foundation
import AVFoundation
import AudioToolbox
import CoreMedia
import Darwin

struct MediaImportError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

final class MediaImportCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    func cancel() { lock.lock(); value = true; lock.unlock() }
    var isCancelled: Bool {
        lock.lock(); defer { lock.unlock() }
        return value
    }
}

/// Converts the first audio track of an audio/video asset into a sandboxed
/// 16 kHz mono PCM WAV. The output is intentionally a normal WAV so the
/// existing ASR and AVAudioPlayer paths can consume imports unchanged.
enum MediaImporter {
    private static let sampleRate = 16_000
    private static let framesPerChunk = 4_096
    // RIFF's size includes the 36 bytes preceding the data payload, not just PCM.
    private static let maximumDataBytes = UInt64(UInt32.max) - 36

    static func copySource(from source: URL, to destination: URL, sourceAlreadyScoped: Bool = false,
                           isCancelled: @escaping () -> Bool = { false },
                           progress: @escaping (Double) -> Void = { _ in }) throws {
        let scoped = sourceAlreadyScoped ? false : source.startAccessingSecurityScopedResource()
        defer { if scoped { source.stopAccessingSecurityScopedResource() } }
        try FileManager.default.createDirectory(at: destination.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        let temp = temporaryURL(for: destination)
        defer { try? FileManager.default.removeItem(at: temp) }
        guard FileManager.default.createFile(atPath: temp.path, contents: nil) else {
            throw MediaImportError(message: "Could not create the temporary source file.")
        }
        let input = try FileHandle(forReadingFrom: source)
        let output = try FileHandle(forWritingTo: temp)
        defer { try? input.close(); try? output.close() }
        let fileSize = (try? source.resourceValues(forKeys: [.fileSizeKey]).fileSize).map(Int64.init) ?? 0
        var copied: Int64 = 0
        while true {
            if isCancelled() { throw CancellationError() }
            let chunk = try input.read(upToCount: 1_048_576) ?? Data()
            if chunk.isEmpty { break }
            try output.write(contentsOf: chunk)
            copied += Int64(chunk.count)
            if fileSize > 0 { progress(min(1, Double(copied) / Double(fileSize))) }
        }
        if isCancelled() { throw CancellationError() }
        try output.close()
        try input.close()
        progress(1)
        // Stage beside the destination so a failed copy never destroys an old import.
        try installTemporaryFile(temp, at: destination)
    }

    static func decodeAudio(from source: URL, to destination: URL,
                            isCancelled: @escaping () -> Bool,
                            progress: @escaping (Double) -> Void) throws -> Double {
        func checkCancellation() throws {
            if isCancelled() { throw CancellationError() }
        }
        try checkCancellation()
        let scoped = source.startAccessingSecurityScopedResource()
        defer { if scoped { source.stopAccessingSecurityScopedResource() } }

        let asset = AVAsset(url: source)
        guard let track = asset.tracks(withMediaType: .audio).first else {
            try checkCancellation()
            throw MediaImportError(message: "This file has no audio track.")
        }
        // Container duration is only a progress estimate, never the returned WAV duration.
        let assetDuration = asset.duration
        let seconds = assetDuration.isNumeric ? assetDuration.seconds : 0
        let estimatedDuration = seconds.isFinite && seconds > 0 ? seconds : 0
        try checkCancellation()
        let outputSettings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 32,
            AVLinearPCMIsFloatKey: true,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false
        ]
        let reader = try AVAssetReader(asset: asset)
        defer {
            if reader.status == .reading || reader.status == .unknown {
                reader.cancelReading()
            }
        }
        // Unlike a track output, the mix output supports sample-rate conversion
        // and channel downmixing, even when only one source track is selected.
        let output = AVAssetReaderAudioMixOutput(audioTracks: [track], audioSettings: outputSettings)
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else {
            throw MediaImportError(message: "The audio track could not be decoded.")
        }
        reader.add(output)

        let temp = temporaryURL(for: destination)
        try FileManager.default.createDirectory(at: temp.deletingLastPathComponent(),
                                                 withIntermediateDirectories: true)
        // Install cleanup before either creation or opening can fail. The handle
        // cleanup below runs first, so we never unlink a still-open writer here.
        defer { try? FileManager.default.removeItem(at: temp) }
        guard FileManager.default.createFile(atPath: temp.path, contents: nil) else {
            throw MediaImportError(message: "Could not create the temporary audio file.")
        }
        let handle = try FileHandle(forWritingTo: temp)
        var closed = false
        defer { if !closed { try? handle.close() } }
        try handle.write(contentsOf: wavHeader(dataBytes: 0))
        try checkCancellation()
        guard reader.startReading() else {
            try checkCancellation()
            if reader.status == .cancelled { throw CancellationError() }
            throw reader.error ?? MediaImportError(message: "The media decoder could not start.")
        }

        var dataBytes: UInt64 = 0
        // Only fixed-size scratch storage is allocated, regardless of asset length
        // or the size/contiguity of a decoded sample buffer.
        var samples = [Float](repeating: 0, count: framesPerChunk)
        var pcm = Data(count: framesPerChunk * MemoryLayout<Int16>.size)
        while true {
            // Release each AVFoundation buffer and bridged Foundation temporaries
            // before requesting the next buffer, including on throws/cancellation.
            let hasSample = try autoreleasepool { () throws -> Bool in
                try checkCancellation()
                guard let sample = output.copyNextSampleBuffer() else {
                    try checkCancellation()
                    return false
                }
                try checkCancellation()
                let (block, frameCount) = try validatedAudioData(in: sample)
                let nextBytes = UInt64(frameCount) * 2
                guard nextBytes <= maximumDataBytes - dataBytes else {
                    throw MediaImportError(message: "The audio is too large for a standard WAV file.")
                }

                var offset = 0
                while offset < frameCount {
                    try checkCancellation()
                    let count = min(framesPerChunk, frameCount - offset)
                    // Copy into aligned Float storage; a CMBlockBuffer need not
                    // be contiguous or have suitably aligned backing memory.
                    let status = samples.withUnsafeMutableBufferPointer { buffer in
                        CMBlockBufferCopyDataBytes(block, atOffset: offset * MemoryLayout<Float>.size,
                                                   dataLength: count * MemoryLayout<Float>.size,
                                                   destination: UnsafeMutableRawPointer(buffer.baseAddress!))
                    }
                    guard status == kCMBlockBufferNoErr else {
                        throw MediaImportError(message: "The audio samples could not be read.")
                    }
                    pcm.withUnsafeMutableBytes { (bytes: UnsafeMutableRawBufferPointer) in
                        for index in 0..<count {
                            // Malformed/non-finite Float samples become silence;
                            // clipping before integer conversion also prevents traps.
                            let value = samples[index].isFinite ? samples[index] : 0
                            let clipped = max(Float(-1), min(Float(1), value))
                            let scaled = (clipped * 32_768).rounded()
                            let sample16 = Int16(max(Float(-32_768), min(Float(32_767), scaled)))
                            let bits = UInt16(bitPattern: sample16)
                            bytes[index * 2] = UInt8(truncatingIfNeeded: bits)
                            bytes[index * 2 + 1] = UInt8(truncatingIfNeeded: bits >> 8)
                        }
                    }
                    try handle.write(contentsOf: pcm.prefix(count * 2))
                    dataBytes += UInt64(count) * 2
                    offset += count
                }
                if estimatedDuration > 0 {
                    let writtenSeconds = Double(dataBytes / 2) / Double(sampleRate)
                    progress(min(0.95, writtenSeconds / estimatedDuration))
                }
                return true
            }
            if !hasSample { break }
        }
        try checkCancellation()
        guard reader.status == .completed else {
            if reader.status == .cancelled { throw CancellationError() }
            throw reader.error ?? MediaImportError(message: "The file could not be decoded into audio.")
        }
        guard dataBytes > 0 else {
            throw MediaImportError(message: "The file contains no decodable audio samples.")
        }
        // The checked limit above makes both UInt32 conversion and RIFF addition safe.
        try handle.seek(toOffset: 0)
        try handle.write(contentsOf: wavHeader(dataBytes: UInt32(dataBytes)))
        try handle.close()
        closed = true
        try checkCancellation()
        // This rename is the commit point; earlier failures preserve any old file.
        try installTemporaryFile(temp, at: destination)
        progress(1)
        return Double(dataBytes / 2) / Double(sampleRate)
    }

    private static func validatedAudioData(in sample: CMSampleBuffer) throws -> (CMBlockBuffer, Int) {
        guard CMSampleBufferIsValid(sample), CMSampleBufferDataIsReady(sample),
              let description = CMSampleBufferGetFormatDescription(sample),
              CMFormatDescriptionGetMediaType(description) == kCMMediaType_Audio,
              let format = CMAudioFormatDescriptionGetStreamBasicDescription(description) else {
            throw MediaImportError(message: "The decoded audio has no valid format description.")
        }
        let asbd = format.pointee
        // Validate what the decoder actually delivered, not just the requested settings.
        // The conversion loop interprets little-endian packed Float32 mono PCM.
        guard asbd.mFormatID == kAudioFormatLinearPCM,
              asbd.mSampleRate.isFinite, asbd.mSampleRate == Double(sampleRate),
              asbd.mChannelsPerFrame == 1, asbd.mBitsPerChannel == 32,
              asbd.mBytesPerFrame == 4, asbd.mBytesPerPacket == 4,
              asbd.mFramesPerPacket == 1,
              asbd.mFormatFlags == (kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked) else {
            throw MediaImportError(message: "The decoded audio is not 16 kHz mono Float32 PCM.")
        }
        let frames = CMSampleBufferGetNumSamples(sample)
        let (expectedBytes, overflow) = frames.multipliedReportingOverflow(by: MemoryLayout<Float>.size)
        guard frames > 0, !overflow,
              let block = CMSampleBufferGetDataBuffer(sample),
              CMSampleBufferGetTotalSampleSize(sample) == expectedBytes,
              CMBlockBufferGetDataLength(block) == expectedBytes else {
            throw MediaImportError(message: "The decoded audio sample buffer is invalid or incomplete.")
        }
        return (block, frames)
    }

    private static func temporaryURL(for destination: URL) -> URL {
        destination.deletingLastPathComponent()
            .appendingPathComponent(".\(destination.lastPathComponent).\(UUID().uuidString).tmp")
    }

    private static func installTemporaryFile(_ temp: URL, at destination: URL) throws {
        // Both paths are siblings on the same filesystem. POSIX rename atomically
        // installs a new file or replaces an existing one without a delete/move gap.
        try temp.withUnsafeFileSystemRepresentation { sourcePath in
            try destination.withUnsafeFileSystemRepresentation { destinationPath in
                guard let sourcePath = sourcePath, let destinationPath = destinationPath else {
                    throw MediaImportError(message: "The import destination is not a valid file path.")
                }
                guard Darwin.rename(sourcePath, destinationPath) == 0 else {
                    let code = errno
                    throw NSError(domain: NSPOSIXErrorDomain, code: Int(code),
                                  userInfo: [NSFilePathErrorKey: destination.path])
                }
            }
        }
    }

    private static func wavHeader(dataBytes: UInt32) -> Data {
        // Only called with a payload already checked against maximumDataBytes.
        var data = Data()
        data.append(contentsOf: Array("RIFF".utf8))
        appendLE(UInt32(36) + dataBytes, to: &data)
        data.append(contentsOf: Array("WAVEfmt ".utf8))
        appendLE(UInt32(16), to: &data)
        appendLE(UInt16(1), to: &data)
        appendLE(UInt16(1), to: &data)
        appendLE(UInt32(sampleRate), to: &data)
        appendLE(UInt32(sampleRate * 2), to: &data)
        appendLE(UInt16(2), to: &data)
        appendLE(UInt16(16), to: &data)
        data.append(contentsOf: Array("data".utf8))
        appendLE(dataBytes, to: &data)
        return data
    }

    private static func appendLE<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        var little = value.littleEndian
        withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
    }
}
