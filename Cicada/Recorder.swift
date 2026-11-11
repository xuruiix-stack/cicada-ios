import AVFoundation
import Speech
import Foundation

/// One VAD segment (segmenter.py's gate): the audio between a voice starting and a pause
/// ending it. `samples` are the segment's 16 kHz mono audio for the voiceprint engine;
/// `t` is where the segment started, in seconds from the recording's start.
struct VoiceSegment {
    let t: Double
    let samples: [Float]
}

/// Segmenter gate (segmenter.py): threshold 0.01, pause 0.6 s, max 8 s, pre-roll 0.15 s.
/// Microphone recording (AVAudioEngine → AAC .m4a) with live level, plus on-device
/// speech recognition (SFSpeechRecognizer) for live captions when no AI key is set.
/// While recording, the same 16 kHz stream feeds a voice gate: each voiced segment's
/// audio is collected for the voiceprint engine (SpeakerID.swift).
final class Recorder: NSObject, ObservableObject {
    @Published var isRecording = false
    @Published var level: Float = 0
    @Published var elapsed: Double = 0
    @Published var liveText = ""
    @Published var error: String?

    /// Segments closed since the last read (speaker separation). Published on the main
    /// thread; the tap hands them over there. The consumer (AppModel) reads and clears.
    @Published private(set) var voiceSegments: [VoiceSegment] = []
    /// The recognizer's words with their midpoints in recording time (published on main,
    /// replaced whole on every recognition result): AppModel captions each voice segment
    /// with the words said in it, so the transcript can be laid over the segments by text.
    @Published private(set) var timedWords: [(mid: Double, text: String)] = []

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

    // Voice gate state. The tap runs on the engine's thread and stop() closes the open
    // segment on the main one, so both take the lock; per-buffer work is tiny.
    private let gateLock = NSLock()
    private var inVoice = false
    private var segStart = 0.0            // seconds from recording start
    private var segSamples: [Float] = []  // the open segment's 16 kHz audio
    private var silence = 0.0             // seconds of quiet since the last voiced chunk
    private var voicedInSeg = 0.0         // voiced seconds; a segment needs some to count

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
        gateLock.lock()
        inVoice = false; segSamples = []; silence = 0; voicedInSeg = 0
        gateLock.unlock()
        voiceSegments = []
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
                DispatchQueue.main.async {
                    self?.liveText = r.bestTranscription.formattedString
                    // Word timestamps for the segment captions (speaker separation):
                    // SFTranscription's timestamps run on the recognizer's own clock,
                    // which starts when the request was made — the same moment as
                    // `startedAt`, so they read as recording time directly.
                    self?.timedWords = r.bestTranscription.segments.map {
                        (mid: $0.timestamp + $0.duration / 2, text: String($0.substring))
                    }
                }
            }
        }
    }

    /// Segments closed so far, taken and cleared (AppModel reads on each arrive).
    func takeVoiceSegments() -> [VoiceSegment] {
        let out = voiceSegments
        voiceSegments = []
        return out
    }

    /// One tap buffer → converted 16 kHz buffers → file, level, live captions and the voice gate.
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
            gate(ch, frames: Int(cap.frameLength))
        }
        request?.append(cap)
    }

    /// The voice gate (segmenter.py): a chunk louder than the threshold opens a segment, a pause
    /// of `pauseEndsSegment` closes it, and a segment runs at most `maxSegmentLength`. Runs on the
    /// tap's thread; closed segments are handed to the main thread as `voiceSegments`.
    private func gate(_ ch: UnsafePointer<Float>, frames: Int) {
        let seconds = Double(frames) / CicadaTuning.sampleRate
        var loud = 0
        for i in 0..<frames where abs(ch[i]) > CicadaTuning.gateThreshold { loud += 1 }
        let voiced = Double(loud) / Double(frames) > 0.5
        gateLock.lock(); defer { gateLock.unlock() }
        if voiced {
            if !inVoice {
                inVoice = true
                segStart = Date().timeIntervalSince(startedAt) - CicadaTuning.preRoll
                silence = 0
                // The pre-roll audio is gone (the gate opens after it passed); the segment
                // starts empty and this chunk begins it. Voiceprints tolerate the 150 ms.
            }
            silence = 0
            voicedInSeg += seconds
            segSamples.append(contentsOf: UnsafeBufferPointer(start: ch, count: frames))
        } else if inVoice {
            silence += seconds
            // Keep the trailing silence out of the segment's samples but let the pause
            // timer run: only audio while the voice was on belongs to the voiceprint.
        } else {
            return
        }
        let segSeconds = Double(segSamples.count) / CicadaTuning.sampleRate
        if inVoice, (silence >= CicadaTuning.pauseEndsSegment && voicedInSeg > 0)
            || segSeconds >= CicadaTuning.maxSegmentLength {
            closeLockedSegment()
        }
    }

    /// Caller holds `gateLock`. Hands the closed segment to the main thread; when the
    /// caller is the main thread itself (stop), append synchronously so the flush is
    /// visible the moment stop() returns — otherwise the async hop would leave the
    /// last segment unread until the next run-loop turn, after applySpeakers drained.
    private func closeLockedSegment() {
        guard inVoice, !segSamples.isEmpty else { inVoice = false; return }
        let seg = VoiceSegment(t: max(0, segStart), samples: segSamples)
        inVoice = false; segSamples = []; silence = 0; voicedInSeg = 0
        if Thread.isMainThread {
            voiceSegments.append(seg)
        } else {
            DispatchQueue.main.async { [weak self] in
                self?.voiceSegments.append(seg)
            }
        }
    }

    func stop() -> URL? {
        timer?.invalidate(); timer = nil
        engine?.inputNode.removeTap(onBus: 0); engine?.stop(); engine = nil; converter = nil
        request?.endAudio(); task?.cancel(); request = nil; task = nil; recognizer = nil
        // A voice still open when Stop was tapped is a segment too (its pause never came).
        // The tap is gone (removeTap above), so the gate state is all ours now, but take
        // the lock anyway: a tap callback already past removeTap may still be finishing.
        gateLock.lock()
        closeLockedSegment()
        gateLock.unlock()
        isRecording = false; level = 0
        let url = file?.url
        file = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        return url
    }
}
