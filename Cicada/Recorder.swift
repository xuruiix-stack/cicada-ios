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

    func requestPermissions() async -> Bool {
        let mic: Bool = await withCheckedContinuation { continuation in
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                continuation.resume(returning: granted)
            }
        }
        let speech: SFSpeechRecognizerAuthorizationStatus = await withCheckedContinuation { continuation in
            SFSpeechRecognizer.requestAuthorization { status in
                continuation.resume(returning: status)
            }
        }
        return mic && speech == .authorized
    }

    func start(audioURL: URL, locale: String = "en-US") throws {
        stop()
        let fmt = AVAudioFormat(standardFormatWithSampleRate: CicadaTuning.sampleRate, channels: 1)!
        file = try AVAudioFile(forWriting: audioURL, settings: [
            AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: CicadaTuning.sampleRate, AVNumberOfChannelsKey: 1])
        let engine = AVAudioEngine(); self.engine = engine
        let input = engine.inputNode
        let tapFmt = input.outputFormat(forBus: 0)
        input.installTap(onBus: 0, bufferSize: 4096, format: tapFmt) { [weak self] buffer, _ in
            guard let self, let file = self.file else { return }
            try? file.write(from: buffer)
            let ch = buffer.frameLength > 0 ? buffer.floatChannelData![0] : nil
            if let ch { self.level = min(1, (0..<Int(buffer.frameLength)).map { abs(ch[$0]) }.max() ?? 0) }
            self.request?.append(buffer)
        }
        try engine.start()
        startedAt = Date(); isRecording = true
        timer = Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { _ in
            self.elapsed = Date().timeIntervalSince(self.startedAt) }
        // Live on-device captions
        if let rec = SFSpeechRecognizer(locale: Locale(identifier: locale)) {
            recognizer = rec
            let req = SFSpeechAudioBufferRecognitionRequest()
            req.shouldReportPartialResults = true; req.requiresOnDeviceRecognition = true
            request = req
            task = rec.recognitionTask(with: req) { [weak self] result, _ in
                if let r = result { self?.liveText = r.bestTranscription.formattedString }
            }
        }
    }

    func stop() -> URL? {
        engine?.inputNode.removeTap(onBus: 0); engine?.stop(); engine = nil
        request?.endAudio(); task?.cancel(); request = nil; task = nil
        timer?.invalidate(); timer = nil
        isRecording = false; level = 0
        return file?.url
    }
}
