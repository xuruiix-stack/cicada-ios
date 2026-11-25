import Accelerate
import AVFoundation
import CoreML
import Foundation
import os

// 说话人分离 (speaker diarization): each heard segment's audio becomes a voiceprint
// (WeSpeaker ResNet34-LM, VoxCeleb, CC BY 4.0, bundled as Core ML), voiceprints are grouped
// as they arrive, and a voice gets a number once it has enough substantive lines. Ported
// from the upstream Mac app's Sources/Cicada/SpeakerID.swift; thresholds were tuned on AMI
// meetings, the model came from upstream's scripts/convert-voiceprint.py.

/// Kaldi log-mel filterbank, as `torchaudio.compliance.kaldi.fbank` computes it with the defaults the
/// model was trained on: 16 kHz, 25 ms Povey windows every 10 ms, DC removed, pre-emphasis 0.97,
/// 512-point power spectrum, 80 mel bins from 20 Hz to 8 kHz, natural log.
enum Fbank {
    static let bins = CicadaTuning.fbankBins
    static let frameLength = CicadaTuning.fbankWindow
    static let shift = CicadaTuning.fbankShift
    private static let fftSize = CicadaTuning.fbankFFT

    private static let window: [Float] = (0..<frameLength).map {
        pow(0.5 - 0.5 * cos(2 * Float.pi * Float($0) / Float(frameLength - 1)), 0.85)
    }

    /// `bins` rows of `fftSize / 2 + 1` weights (the Nyquist column is always zero, as in Kaldi).
    private static let melBanks: [Float] = {
        func mel(_ f: Double) -> Double { 1127 * log(1 + f / 700) }
        let cols = fftSize / 2 + 1
        let lo = mel(20), hi = mel(8000), delta = (hi - lo) / Double(bins + 1)
        var w = [Float](repeating: 0, count: bins * cols)
        for m in 0..<bins {
            let left = lo + Double(m) * delta, center = left + delta, right = center + delta
            for k in 0..<(fftSize / 2) {
                let x = mel(Double(k) * 16000 / Double(fftSize))
                let v = min((x - left) / (center - left), (right - x) / (right - center))
                if v > 0 { w[m * cols + k] = Float(v) }
            }
        }
        return w
    }()

    private static let setup = vDSP_create_fftsetup(9, FFTRadix(kFFTRadix2))!

    /// Frames × `bins`, row-major. `samples` are 16 kHz mono in -1...1.
    static func compute(_ samples: [Float]) -> [Float] {
        guard samples.count >= frameLength else { return [] }
        let frames = 1 + (samples.count - frameLength) / shift
        let cols = fftSize / 2 + 1
        var out = [Float](repeating: 0, count: frames * bins)
        var frame = [Float](repeating: 0, count: fftSize)
        var re = [Float](repeating: 0, count: fftSize / 2), im = re
        var power = [Float](repeating: 0, count: cols)
        var mels = [Float](repeating: 0, count: bins)
        for f in 0..<frames {
            let start = f * shift
            for i in 0..<frameLength { frame[i] = samples[start + i] * 32768 }
            for i in frameLength..<fftSize { frame[i] = 0 }
            var mean: Float = 0
            vDSP_meanv(frame, 1, &mean, vDSP_Length(frameLength))
            for i in 0..<frameLength { frame[i] -= mean }
            for i in stride(from: frameLength - 1, to: 0, by: -1) { frame[i] -= CicadaTuning.preEmphasis * frame[i - 1] }
            frame[0] -= CicadaTuning.preEmphasis * frame[0]
            vDSP_vmul(frame, 1, window, 1, &frame, 1, vDSP_Length(frameLength))
            re.withUnsafeMutableBufferPointer { rp in
                im.withUnsafeMutableBufferPointer { ip in
                    var split = DSPSplitComplex(realp: rp.baseAddress!, imagp: ip.baseAddress!)
                    frame.withUnsafeBytes { raw in
                        vDSP_ctoz(raw.bindMemory(to: DSPComplex.self).baseAddress!, 2, &split, 1, vDSP_Length(fftSize / 2))
                    }
                    vDSP_fft_zrip(setup, &split, 1, 9, FFTDirection(FFT_FORWARD))
                }
            }
            // vDSP's real FFT returns twice the true values, Nyquist packed into im[0].
            power[0] = re[0] * re[0] / 4
            power[cols - 1] = im[0] * im[0] / 4
            for k in 1..<(fftSize / 2) { power[k] = (re[k] * re[k] + im[k] * im[k]) / 4 }
            vDSP_mmul(melBanks, 1, power, 1, &mels, 1, vDSP_Length(bins), 1, vDSP_Length(cols))
            for b in 0..<bins { out[f * bins + b] = log(max(mels[b], .ulpOfOne)) }
        }
        return out
    }
}

/// The bundled voiceprint model. Xcode compiles Core ML models at build time, but the
/// `.mlpackage` is compiled on first use all the same when loaded from the app bundle the
/// way the upstream Mac app does it (Command Line Tools can't compile Core ML models);
/// the result is kept in Application Support so the work happens once.
final class Voiceprints: @unchecked Sendable {
    static let minSeconds = CicadaTuning.minEmbedSeconds
    /// Longer segments are cut: 20 s is plenty to know a voice and bounds the work.
    static let maxSeconds = CicadaTuning.maxEmbedSeconds

    private let model: MLModel

    private init(model: MLModel) { self.model = model }

    static func load() throws -> Voiceprints {
        guard let pkg = Bundle.main.url(forResource: "SpeakerEmbedding", withExtension: "mlpackage") else {
            throw CocoaError(.fileNoSuchFile)
        }
        let size = (try? FileManager.default.attributesOfItem(atPath: pkg.appendingPathComponent("Data/com.apple.CoreML/weights/weight.bin").path)[.size] as? Int) ?? 0
        // Cache under a name keyed on the weight size: a model update in a new app version
        // must not reuse the previous version's compiled copy.
        let support = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Cicada", isDirectory: true)
        try? FileManager.default.createDirectory(at: support, withIntermediateDirectories: true)
        let cache = support.appendingPathComponent("SpeakerEmbedding-\(size).mlmodelc")
        if !FileManager.default.fileExists(atPath: cache.path) {
            let compiled = try MLModel.compileModel(at: pkg)
            try? FileManager.default.removeItem(at: cache)
            try FileManager.default.moveItem(at: compiled, to: cache)
        }
        let config = MLModelConfiguration()
        // Variable-length input doesn't compile for the Neural Engine. CPU only, not the GPU: iOS
        // kills an app that submits GPU work in the background, and a recording keeps embedding
        // voice segments with the screen locked (killed ~5 s after locking, at the first segment).
        config.computeUnits = .cpuOnly
        return Voiceprints(model: try MLModel(contentsOf: cache, configuration: config))
    }

    /// A whole audio file as 16 kHz mono floats (debug and tests).
    static func samples(of url: URL) -> [Float]? {
        guard let file = try? AVAudioFile(forReading: url),
              let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: CicadaTuning.sampleRate, channels: 1, interleaved: false),
              let input = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length)),
              (try? file.read(into: input)) != nil,
              let conv = AVAudioConverter(from: file.processingFormat, to: format) else { return nil }
        let cap = AVAudioFrameCount((Double(input.frameLength) * CicadaTuning.sampleRate / file.processingFormat.sampleRate).rounded(.up)) + 32
        guard let out = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: cap) else { return nil }
        var fed = false; var err: NSError?
        conv.convert(to: out, error: &err) { _, status in
            if fed { status.pointee = .noDataNow; return nil }
            fed = true; status.pointee = .haveData
            return input
        }
        guard fed, err == nil, let d = out.floatChannelData else { return nil }
        return Array(UnsafeBufferPointer(start: d[0], count: Int(out.frameLength)))
    }

    /// A unit-length 256-d voiceprint for 16 kHz mono samples, or nil when too short.
    func embed(_ samples: [Float]) -> [Float]? {
        guard Double(samples.count) >= Self.minSeconds * CicadaTuning.sampleRate else { return nil }
        let clip = samples.count > Int(Self.maxSeconds * CicadaTuning.sampleRate)
            ? Array(samples.prefix(Int(Self.maxSeconds * CicadaTuning.sampleRate))) : samples
        var feats = Fbank.compute(clip)
        let frames = feats.count / Fbank.bins
        guard frames >= CicadaTuning.minEmbedFrames else { return nil }
        // Mean normalisation per bin over the clip, as in training.
        for b in 0..<Fbank.bins {
            var sum: Float = 0
            for f in 0..<frames { sum += feats[f * Fbank.bins + b] }
            let mean = sum / Float(frames)
            for f in 0..<frames { feats[f * Fbank.bins + b] -= mean }
        }
        guard let input = try? MLMultiArray(shape: [1, NSNumber(value: frames), NSNumber(value: Fbank.bins)], dataType: .float32) else { return nil }
        feats.withUnsafeBufferPointer { src in
            input.withUnsafeMutableBytes { dst, _ in
                dst.copyMemory(from: UnsafeRawBufferPointer(src))
            }
        }
        guard let out = try? model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["feats": input])),
              let embs = out.featureValue(for: "embs")?.multiArrayValue else { return nil }
        var v = (0..<embs.count).map { Float(truncating: embs[$0]) }
        var norm: Float = 0
        vDSP_svesq(v, 1, &norm, vDSP_Length(v.count))
        guard norm > 0 else { return nil }
        vDSP_vsdiv(v, 1, [norm.squareRoot()], &v, 1, vDSP_Length(v.count))
        return v
    }
}

/// Groups voiceprints as lines arrive, never revising a line already labelled. A line joins the
/// closest voice when similar enough; a long line unlike every voice starts a new one; anything
/// else stays unlabelled. Tuned on AMI dev meetings.
struct VoiceClusterer {
    static let newSeconds = CicadaTuning.newVoiceMinSeconds
    static let join: Float = Float(CicadaTuning.joinThreshold)
    static let startNew: Float = Float(CicadaTuning.startNewThreshold)

    private(set) var sums: [[Float]] = []

    /// The user said two voices are one: later lines like either go to `into`.
    mutating func merge(_ from: Int, into: Int) {
        guard sums.indices.contains(from), sums.indices.contains(into), from != into else { return }
        vDSP_vadd(sums[into], 1, sums[from], 1, &sums[into], 1, vDSP_Length(sums[into].count))
        sums[from] = [Float](repeating: 0, count: sums[from].count)   // never the closest again
    }

    mutating func label(_ v: [Float], seconds: Double) -> Int? {
        var best = -1, bestSim: Float = -1
        for (i, s) in sums.enumerated() {
            var dot: Float = 0, sq: Float = 0
            vDSP_dotpr(v, 1, s, 1, &dot, vDSP_Length(v.count))
            vDSP_svesq(s, 1, &sq, vDSP_Length(s.count))
            let sim = dot / max(sq.squareRoot(), .ulpOfOne)
            if sim > bestSim { bestSim = sim; best = i }
        }
        if best >= 0, bestSim >= Self.join {
            vDSP_vadd(sums[best], 1, v, 1, &sums[best], 1, vDSP_Length(v.count))
            return best
        }
        if seconds >= Self.newSeconds, sums.isEmpty || bestSim < Self.startNew {
            sums.append(v)
            return sums.count - 1
        }
        return nil
    }
}

/// What the transcript shows per line. A voice gets a number only once it has `shownAfter`
/// substantive lines, and only when at least two voices have, so a solo recording stays "S1".
/// Numbers follow the order voices reached `shownAfter` lines, so a number never moves once shown.
enum SpeakerLabels {
    static let shownAfter = CicadaTuning.shownAfter

    /// Voice id → 1-based speaker number, for the voices on show. A line counts towards
    /// its voice only when substantive (SessionStore's rule: three words, or four CJK
    /// characters — "Yeah." doesn't say whose voice it is).
    static func voiceNumbers(_ lines: [TranscriptLine]) -> [Int: Int] {
        var count: [Int: Int] = [:], reached: [Int: Int] = [:]
        for (i, l) in lines.enumerated() {
            guard let v = l.voice, SessionStore.substantive(l.src) else { continue }
            count[v, default: 0] += 1
            if count[v] == shownAfter { reached[v] = i }
        }
        let shown = reached.keys.sorted { reached[$0]! < reached[$1]! }
        guard shown.count >= 2 else { return [:] }
        return Dictionary(uniqueKeysWithValues: shown.enumerated().map { ($1, $0 + 1) })
    }

    /// Line id → 1-based speaker number; lines without one show as the unlabelled voice.
    static func numbers(_ lines: [TranscriptLine]) -> [String: Int] {
        let number = voiceNumbers(lines)
        var out: [String: Int] = [:]
        for l in lines where l.voice != nil {
            if let v = l.voice, let n = number[v] { out[l.id] = n }
        }
        return out
    }
}

/// The file fallback uses the same voice gate as the live recorder, but its clock is the
/// decoded PCM, not wall time (interruptions need not be present in the recording).
/// Only one 100 ms chunk and one bounded voice segment are retained at a time.
private struct FileVoiceGate {
    private static let rate = CicadaTuning.sampleRate
    private static let chunkFrames = max(1, Int(rate * CicadaTuning.captureChunk))
    private var chunk: [Float] = []
    private var samples: [Float] = []
    private var processedFrames = 0
    private var silenceFrames = 0
    private var start = 0.0

    mutating func append(_ input: UnsafeBufferPointer<Float>, emit: (Double, [Float]) -> Void) {
        var offset = 0
        while offset < input.count {
            let n = min(Self.chunkFrames - chunk.count, input.count - offset)
            chunk.append(contentsOf: input[offset..<(offset + n)])
            offset += n
            if chunk.count == Self.chunkFrames { consumeChunk(emit: emit) }
        }
    }

    mutating func finish(emit: (Double, [Float]) -> Void) {
        if !chunk.isEmpty { consumeChunk(emit: emit) }
        close(emit: emit)
    }

    private mutating func consumeChunk(emit: (Double, [Float]) -> Void) {
        let loud = chunk.reduce(0) { $0 + (abs($1) > CicadaTuning.gateThreshold ? 1 : 0) }
        if loud * 2 > chunk.count {
            if samples.isEmpty {
                start = max(0, Double(processedFrames) / Self.rate - CicadaTuning.preRoll)
            }
            silenceFrames = 0
            samples.append(contentsOf: chunk)
        } else if !samples.isEmpty {
            silenceFrames += chunk.count
        }
        processedFrames += chunk.count
        chunk.removeAll(keepingCapacity: true)
        if !samples.isEmpty,
           Double(silenceFrames) / Self.rate >= CicadaTuning.pauseEndsSegment
            || Double(samples.count) / Self.rate >= CicadaTuning.maxSegmentLength {
            close(emit: emit)
        }
    }

    private mutating func close(emit: (Double, [Float]) -> Void) {
        guard !samples.isEmpty else { return }
        emit(start, samples)
        samples.removeAll(keepingCapacity: true)
        silenceFrames = 0
    }
}

private final class FileSpeakerCancellation: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false

    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
    func check() throws {
        lock.lock(); let value = cancelled; lock.unlock()
        if value { throw CancellationError() }
    }
}

private enum FileSpeakerError: LocalizedError {
    case invalidFormat, decodeFailed

    var errorDescription: String? {
        switch self {
        case .invalidFormat: return "The recording's audio format cannot be read for speaker separation."
        case .decodeFailed: return "The recording could not be fully decoded for speaker separation."
        }
    }
}

/// Mutable model state belongs to the utility queue, not SpeakerTagger's main actor.
/// The holder is only touched inside that queue's blocks.
private final class SpeakerWorkerState: @unchecked Sendable {
    var model: Voiceprints?
    var loadFailed = false
}

/// Loads the voiceprint model off the main thread and computes embeddings. The upstream Mac
/// app tags each line as it arrives (its clusterer lives here); the rebuild records a whole
/// meeting first, so the caller holds the `VoiceClusterer` and groups the embeddings in
/// segment order once the recording stops — order is the one thing online clustering can't
/// recover, and it is why the grouping is not done inside this class.
@MainActor
final class SpeakerTagger {
    private let queue = DispatchQueue(label: "com.wquguru.cicada.voiceprint", qos: .utility)
    private let worker = SpeakerWorkerState()
    /// Main-thread: whether the model is known unloadable, set from a done callback once
    /// the load has settled on the queue. The caller stops embedding when it turns false.
    private(set) var available: Bool = true

    /// A new recording: forget the model's load failure only (the clusterer is the caller's).
    func reset() { queue.async { [worker] in worker.loadFailed = false }; available = true }

    /// Load/compile the model NOW, off the recording's hot path (review-1 P1-2:
    /// the first voice segment used to trigger MLModel.compileModel mid-recording —
    /// a heavy CPU/memory spike exactly when the app may already be in the
    /// background, prime jetsam territory). Called when the user taps Record
    /// (the app is foreground and idle): the ~13 MB model compiles/loads once
    /// here, and the first segment's embed becomes a plain inference. Failure
    /// is fine — embed() retries the load and degrades gracefully as before.
    func warmUp() {
        queue.async { [worker] in
            if worker.model == nil, !worker.loadFailed {
                do { worker.model = try Voiceprints.load() }
                catch {
                    worker.loadFailed = true
                    debugLog("voiceprint model failed to load: \(error)")
                }
            }
        }
    }

    /// `done` with the unit-length voiceprint, or nil when the audio is too short or the
    /// model failed to load (the caller then leaves the line unlabelled — upstream ab787d4:
    /// no voice yet is not "like no voice", so a pending or failed voiceprint never
    /// counts as a decision).
    func embed(_ samples: [Float], done: @escaping @MainActor ([Float]?) -> Void) {
        queue.async { [self, worker] in
            if worker.model == nil, !worker.loadFailed {
                do { worker.model = try Voiceprints.load() } catch {
                    worker.loadFailed = true
                    debugLog("voiceprint model failed to load: \(error)")
                }
            }
            let emb = worker.model.flatMap { $0.embed(samples) }
            let failed = worker.loadFailed
            DispatchQueue.main.async {
                // No MainActor.assumeIsolated here: it needs iOS 17, and the deployment
                // target is 16.4. A dispatch to the main queue is the main actor.
                if failed { self.noteUnavailable() }
                done(emb)
            }
        }
    }

    private func noteUnavailable() { available = false }

    /// The same, as an await for the batch flow at stop time.
    func embed(_ samples: [Float]) async -> [Float]? {
        await withCheckedContinuation { cont in embed(samples) { cont.resume(returning: $0) } }
    }

    /// Rebuild a recording's speaker skeleton without live captions or retained raw audio.
    /// Unknown short segments stay in the result; no identified voices means the caller
    /// should keep its existing transcript. Read/decode failures never return partial lines.
    func lines(in url: URL) async throws -> [TranscriptLine] {
        let cancellation = FileSpeakerCancellation()
        return try await withTaskCancellationHandler(operation: {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<[TranscriptLine], Error>) in
                queue.async { [worker] in
                    do {
                        try cancellation.check()
                        let file = try AVAudioFile(forReading: url)
                        if worker.model == nil, !worker.loadFailed {
                            do { worker.model = try Voiceprints.load() }
                            catch {
                                worker.loadFailed = true
                                debugLog("voiceprint model failed to load: \(error)")
                            }
                        }
                        try cancellation.check()
                        guard let model = worker.model else { cont.resume(returning: []); return }
                        let lines = try Self.fileLines(file, model: model, cancellation: cancellation)
                        try cancellation.check()
                        cont.resume(returning: lines)
                    } catch { cont.resume(throwing: error) }
                }
            }
        }, onCancel: { cancellation.cancel() })
    }

    /// Queue-confined: the file, VAD and clusterer belong to this job, never the live session.
    private nonisolated static func fileLines(_ file: AVAudioFile, model: Voiceprints,
                                              cancellation: FileSpeakerCancellation) throws -> [TranscriptLine] {
        let source = file.processingFormat
        guard source.sampleRate.isFinite, source.sampleRate > 0, source.channelCount > 0,
              let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: CicadaTuning.sampleRate,
                                         channels: 1, interleaved: false) else {
            throw FileSpeakerError.invalidFormat
        }
        let frames = AVAudioFrameCount(CicadaTuning.sampleRate * CicadaTuning.captureChunk)
        guard let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames) else {
            throw FileSpeakerError.invalidFormat
        }
        var gate = FileVoiceGate()
        var clusterer = VoiceClusterer()
        var lines: [TranscriptLine] = []
        let emit = { (t: Double, samples: [Float]) in
            var line = TranscriptLine(t: t, src: "")
            line.voice = model.embed(samples).flatMap {
                clusterer.label($0, seconds: Double(samples.count) / CicadaTuning.sampleRate)
            }
            lines.append(line)
        }
        func consume() throws {
            guard output.frameLength > 0 else { return }
            guard let channel = output.floatChannelData?[0] else { throw FileSpeakerError.invalidFormat }
            gate.append(UnsafeBufferPointer(start: channel, count: Int(output.frameLength)), emit: emit)
        }

        if source == format {
            while file.framePosition < file.length {
                try cancellation.check()
                try file.read(into: output, frameCount: frames)
                guard output.frameLength > 0 else { throw FileSpeakerError.decodeFailed }
                try consume()
            }
        } else {
            guard let converter = AVAudioConverter(from: source, to: format) else {
                throw FileSpeakerError.invalidFormat
            }
            // Default conversion remaps channels; mix them so a right-only speaker survives.
            if source.channelCount > 1 { converter.downmix = true }
            // Pull small source buffers as requested by the converter, including its EOF drain.
            // A single read/convert call can stop at a codec packet boundary; it is not EOF.
            var noProgress = 0
            while true {
                try cancellation.check()
                let before = file.framePosition
                var readError: Error?
                var convertError: NSError?
                output.frameLength = 0
                let status = converter.convert(to: output, error: &convertError) { requested, inputStatus in
                    do {
                        try cancellation.check()
                        if file.framePosition >= file.length {
                            inputStatus.pointee = .endOfStream
                            return nil
                        }
                        let capacity = min(max(AVAudioFrameCount(requested), 1), 16_384)
                        guard let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: capacity) else {
                            throw FileSpeakerError.invalidFormat
                        }
                        try file.read(into: input, frameCount: capacity)
                        guard input.frameLength > 0 else { throw FileSpeakerError.decodeFailed }
                        inputStatus.pointee = .haveData
                        return input
                    } catch {
                        readError = error
                        inputStatus.pointee = .endOfStream
                        return nil
                    }
                }
                if let error = readError { throw error }
                if let error = convertError { throw error }
                if status == .error { throw FileSpeakerError.decodeFailed }
                try consume()
                if status == .endOfStream { break }
                if output.frameLength == 0, file.framePosition == before {
                    noProgress += 1
                    guard noProgress < 3 else { throw FileSpeakerError.decodeFailed }
                } else { noProgress = 0 }
            }
        }
        try cancellation.check()
        gate.finish(emit: emit)
        return lines.contains(where: { $0.voice != nil }) ? lines : []
    }
}

/// A tiny debug log: the rebuild has no Diagnostics.swift; keep the failures
/// findable in Console (sysdiagnose / `log stream --predicate 'subsystem == "com.wquguru.cicada"'`).
/// os_log, not print: print goes to stdout only — invisible in Console and in
/// device logs captured after a lock-screen kill, which is exactly where these
/// records (interruption begin/end, engine restarts, killed-session recovery)
/// need to be readable. Logger fits iOS 14+, so 16.4 is covered; the level is
/// .default so the subsystem can be raised to .info/.debug when hunting.
func debugLog(_ message: String) {
    Logger(subsystem: "com.wquguru.cicada", category: "recorder")
        .log("cicada: \(message, privacy: .public)")
}
