import AVFoundation
import Speech
import Foundation

/// Segmenter gate (segmenter.py): threshold 0.01, pause 0.6 s, max 8 s, pre-roll 0.15 s.
/// Microphone recording (AVAudioEngine → AAC .m4a) with live level, plus on-device
/// speech recognition (SFSpeechRecognizer) for live captions when no AI key is set.
final class Recorder: NSObject, ObservableObject {
    @Published var isRecording = false
    @Published var level: Float = 0
    @Published var elapsed: Double = 0
    @Published var liveText = ""
    @Published var error: String?

    private var engine: AVAudioEngine?
    private var file: AVAudioFile?
    private var timer: Timer?
    private var startedAt = Date()
    private var recognizer: SFSpeechRecognizer?
    private var request: SFSpeechAudioBufferRecognitionRequest?
    private var task: SFSpeechRecognitionTask?
    /// Mic tap (hardware rate, mono) → 16 kHz mono the file and recognizer consume.
    private var converter: AVAudioConverter?
    private let sinkFormat = AVAudioFormat(standardFormatWithSampleRate: CicadaTuning.sampleRate, channels: 1)!

    func requestPermissions() async -> Bool {
        let mic: Bool = await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                continuation.resume(returning: granted) }
        }
        let speech: SFSpeechRecognizerAuthorizationStatus = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status) }
        }
        return mic && speech == .authorized
    }

    func start(audioURL: URL, locale: String = "en-US") throws {
        stop()
        // Mic capture needs a session category that allows recording; without it the
        // engine runs but the input node delivers silence (the original app's pattern).
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
        do {
            try session.setActive(true)
            file = try AVAudioFile(forWriting: audioURL, settings: [
                AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: CicadaTuning.sampleRate, AVNumberOfChannelsKey: 1])
        }
        catch {
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            throw error
        }
        let engine = AVAudioEngine(); self.engine = engine
        let input = engine.inputNode
        let tapFmt = input.outputFormat(forBus: 0)
        // The input node runs at hardware rate (e.g. 48 kHz); the file and recognizer
        // want 16 kHz mono. Convert tap buffers instead of writing them raw:
        // AVAudioFile.write throws on a format mismatch and nothing would be recorded.
        guard let conv = AVAudioConverter(from: tapFmt, to: sinkFormat) else {
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            throw NSError(domain: AVFoundationErrorDomain, code: -10875,
                          userInfo: [NSLocalizedDescriptionKey: "Cannot convert mic input (\(tapFmt)) to recording format."])
        }
        converter = conv
        input.installTap(onBus: 0, bufferSize: 4096, format: tapFmt) { [weak self] buffer, _ in
            self?.handle(buffer)
        }
        do { try engine.start() }
        catch {
            try? session.setActive(false, options: .notifyOthersOnDeactivation)
            throw error
        }
        startedAt = Date(); isRecording = true
        // Schedule on the main run loop explicitly: start() can resume off-main after
        // the permission continuation, and a timer on a background run loop never fires.
        let t = Timer(timeInterval: 0.2, repeats: true) { _ in
            self.elapsed = Date().timeIntervalSince(self.startedAt) }
        timer = t
        RunLoop.main.add(t, forMode: .common)
        // Live on-device captions
        if let rec = SFSpeechRecognizer(locale: Locale(identifier: locale)) {
            recognizer = rec
            let req = SFSpeechAudioBufferRecognitionRequest()
            req.shouldReportPartialResults = true; req.requiresOnDeviceRecognition = true
            request = req
            task = rec.recognitionTask(with: req) { [weak self] result, _ in
                guard let r = result else { return }
                DispatchQueue.main.async { self?.liveText = r.bestTranscription.formattedString }
            }
        }
    }

    /// One tap buffer → converted 16 kHz buffers → file, level and live captions.
    private func handle(_ buffer: AVAudioPCMBuffer) {
        guard let file, let converter else { return }
        let ratio = sinkFormat.sampleRate / buffer.format.sampleRate
        let capFrames = AVAudioFrameCount((Double(buffer.frameLength) * ratio).rounded(.up)) + 32
        guard let cap = AVAudioPCMBuffer(pcmFormat: sinkFormat, frameCapacity: capFrames) else { return }
        var fed = false; var err: NSError?
        converter.convert(to: cap, error: &err) { _, inputStatus in
            if fed { inputStatus.pointee = .noDataNow; return nil }
            fed = true
            inputStatus.pointee = .haveData
            return buffer
        }
        guard fed, err == nil, cap.frameLength > 0 else { return }
        do { try file.write(from: cap) }
        catch { DispatchQueue.main.async { self.error = "Recording failed: \(error.localizedDescription)" } }
        if let ch = cap.floatChannelData?[0] {
            // Tap runs off-main; publish the level where SwiftUI reads it.
            let v = min(1, (0..<Int(cap.frameLength)).map { abs(ch[$0]) }.max() ?? 0)
            DispatchQueue.main.async { self.level = v }
        }
        request?.append(cap)
    }

    func stop() -> URL? {
        timer?.invalidate(); timer = nil
        engine?.inputNode.removeTap(onBus: 0); engine?.stop(); engine = nil; converter = nil
        request?.endAudio(); task?.cancel(); request = nil; task = nil; recognizer = nil
        isRecording = false; level = 0
        let url = file?.url
        file = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        return url
    }
}
