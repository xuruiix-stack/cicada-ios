import AVFoundation
import Speech
import Foundation
import os

/// One VAD segment (segmenter.py's gate): the audio between a voice starting and a pause
/// ending it. `samples` are the segment's 16 kHz mono audio for the voiceprint engine;
/// `t` is where the segment started, in seconds from the recording's start.
struct VoiceSegment {
    let t: Double
    let samples: [Float]
}

/// Segmenter gate (segmenter.py): threshold 0.01, pause 0.6 s, max 8 s, pre-roll 0.15 s.
/// Microphone recording (AVAudioEngine → CAF LinearPCM — crash-safe, see start())
/// with live level, plus on-device speech recognition (SFSpeechRecognizer) for
/// live captions when no AI key is set.
/// While recording, the same 16 kHz stream feeds a voice gate: each voiced segment's
/// audio is collected for the voiceprint engine (SpeakerID.swift).
final class Recorder: NSObject, ObservableObject {
    @Published var isRecording = false
    @Published var level: Float = 0
    @Published var elapsed: Double = 0
    @Published var liveText = ""
    @Published var error: String?
    /// True while a call, Siri or another app took the audio and we are waiting
    /// for it back. The recording page shows it so "Paused" is never mistaken
    /// for silence or a dead mic; elapsed keeps running (meeting time does).
    @Published private(set) var paused = false
    /// Why on-device recognition isn't producing live captions (model not
    /// downloaded, locale unsupported, recognizer failed…). The recording
    /// page shows it so a silent failure is never mistaken for silence.
    @Published var recognitionError: String?
    /// The recording could not go on (interruption ended without the system
    /// giving the audio back, or the engine could not be restarted). Set at
    /// most once per recording; AppModel finalizes the partial recording
    /// truthfully instead of leaving the UI claim "recording".
    var onRecordingLost: ((String) -> Void)?

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
    /// Interruption / route-change observers, torn down in stop().
    private var observers: [NSObjectProtocol] = []
    /// Inside an interruption: a route change then must not try to take the
    /// microphone back (upstream RoomAudio's `held` rule).
    private var held = false
    /// True when a resume must wait for the app to become active again (the
    /// interruption ended while the app was still behind a call screen).
    private var resumeWhenActive = false
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
            // Crash-safe sink (2026-11-14 field report): the recorder used to write
            // AAC .m4a, whose MP4 container is only finalized by ExtAudioFile's
            // teardown. When iOS jetsam-killed the app in the background mid-recording,
            // the file was left without its moov atom — AVAudioFile, AVAudioPlayer and
            // the transcription's converter all failed on it, so every kill was a
            // TOTAL loss (recovered session: 0 duration, unplayable, untranscribable).
            // CAF with 16-bit LinearPCM has no trailing container structures: the
            // header is written up front and each PCM packet lands sequentially, so
            // a file abandoned at any frame is readable up to the last flushed write.
            // Cost, stated honestly: 16 kHz mono 16-bit PCM ≈ 1.9 MB/min (AAC was
            // ~0.2-0.5 MB/min) — a 2-hour meeting is ~230 MB. The sink is already
            // 16 kHz mono (the voiceprint/ASR formats), so quality is identical.
            // The pipeline converts to WAV before upload (AIClient.wavCopy);
            // CAF itself never leaves the device.
            file = try AVAudioFile(forWriting: audioURL, settings: [
                AVFormatIDKey: kAudioFormatCAF, AVSampleRateKey: CicadaTuning.sampleRate,
                AVNumberOfChannelsKey: 1, AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsNonInterleaved: false])
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
        lastFootprintLog = Date()
        error = nil   // a new recording starts with a clean slate; AppModel shows the last one
        debugLog("recorder: started")
        // Schedule on the main run loop explicitly: start() can resume off-main after
        // the permission continuation, and a timer on a background run loop never fires.
        let t = Timer(timeInterval: 0.2, repeats: true) { _ in
            self.elapsed = Date().timeIntervalSince(self.startedAt) }
        timer = t
        RunLoop.main.add(t, forMode: .common)
        // Live on-device captions. requiresOnDeviceRecognition stays true —
        // offline subtitles are the point — but the model for this locale may
        // never have been downloaded, in which case the task fails outright.
        // That used to be swallowed by the result guard and looked like "no
        // captions": surface it instead (recognitionError, shown on the
        // recording page).
        recognitionError = nil
        if let rec = SFSpeechRecognizer(locale: Locale(identifier: locale)) {
            recognizer = rec
            if !rec.supportsOnDeviceRecognition {
                recognitionError = "On-device speech recognition is not available for this language on this device, so there are no offline live captions. The recording itself is unaffected; the transcript will be generated when online."
            }
            let req = SFSpeechAudioBufferRecognitionRequest()
            req.shouldReportPartialResults = true; req.requiresOnDeviceRecognition = true
            request = req
            task = rec.recognitionTask(with: req) { [weak self] result, err in
                if let r = result {
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
                // A recognition error is not the recording's end: audio keeps
                // going to the file. Say WHY the captions stopped, in plain words.
                // Cancellation is not an error here: stop() cancels the task,
                // and the cancel callback must not paint a failure note over a
                // recording whose captions worked fine (SFSpeechRecognizer
                // delivers an error on cancel; code 216 in its own domain).
                if let e = err, !Self.isCancellation(e) {
                    DispatchQueue.main.async {
                        guard let self else { return }
                        guard self.recognitionError == nil else { return }
                        // With requiresOnDeviceRecognition, an error before any
                        // result at all is, in practice, the on-device model for
                        // this locale never having been downloaded (the exact
                        // kAFAssistantErrorDomain code varies by OS). After some
                        // results, it's a mid-run failure — keep the system's
                        // wording for that one.
                        self.recognitionError = self.liveText.isEmpty
                            ? "The on-device speech model for this language is not downloaded, so there are no offline live captions. Connect to the internet once so iOS can download it — after that, live captions work offline too. The recording itself is unaffected; the transcript will be generated when online."
                            : Self.describeRecognitionError(e)
                    }
                }
            }
        } else {
            recognitionError = "No speech recognizer exists for this language, so there are no offline live captions. The recording itself is unaffected; the transcript will be generated when online."
        }
        observeAudioSession()
    }

    // MARK: - Interruption and route changes (upstream RoomListen/RoomAudio)

    /// A call, Siri or another app taking the audio used to be the end of the
    /// recording: the session went inactive, nothing fed the engine, and the
    /// UI kept claiming "recording" while the file went silent — then iOS
    /// suspended the background app for having no active audio. The pattern
    /// here is the official iOS app's (RoomListen.swift): pause on
    /// interruption began, take the microphone back when it ends
    /// (shouldResume), and re-tap the engine in whatever format the new route
    /// has. If the audio never comes back, say so and finalize truthfully.
    private func observeAudioSession() {
        let session = AVAudioSession.sharedInstance()
        let center = NotificationCenter.default
        observers = [
            center.addObserver(forName: AVAudioSession.interruptionNotification, object: session, queue: .main) {
                [weak self] n in self?.interrupted(n)
            },
            // A route change (headphones out, Bluetooth off) stops the engine
            // with a configuration change: on again in the new input's format,
            // unless an interruption is holding the audio (upstream's rule).
            center.addObserver(forName: .AVAudioEngineConfigurationChange, object: engine, queue: .main) {
                [weak self] _ in self?.routeChanged()
            },
            // The whole media server died (user rebooted the audio daemon, or a
            // system audio crash). Round 1 covered interruptions and route
            // changes; this one invalidates everything — the engine is dead, the
            // session is reset, and without handling it the app sits with
            // isRecording true while nothing feeds the file, which is exactly
            // the stall-then-suspend chain that gets the app killed.
            center.addObserver(forName: AVAudioSession.mediaServicesWereResetNotification, object: session, queue: .main) {
                [weak self] _ in self?.mediaServicesReset()
            },
        ]
    }

    private func interrupted(_ n: Notification) {
        guard isRecording, let raw = n.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
              let type = AVAudioSession.InterruptionType(rawValue: raw) else { return }
        if type == .began {
            guard !paused else { return }
            held = true
            paused = true
            level = 0
            engine?.pause()
            debugLog("recorder: interruption began")
            return
        }
        // Interruption ended: only resume when the system says it's ours to take.
        let optionsRaw = n.userInfo?[AVAudioSessionInterruptionOptionKey] as? UInt ?? 0
        let shouldResume = AVAudioSession.InterruptionOptions(rawValue: optionsRaw).contains(.shouldResume)
        debugLog("recorder: interruption ended shouldResume=\(shouldResume)")
        guard paused else { return }
        if shouldResume {
            do { try resume() }
            catch {
                // The system won't hand the microphone back yet (the app can
                // still be behind a call screen): try again on activation.
                debugLog("recorder: resume failed: \(error.localizedDescription)")
                resumeWhenActive = true
            }
        } else {
            // Not ours to take back (a call still going in the background):
            // wait for the app to become active, and give up honestly if
            // even then the audio does not return.
            resumeWhenActive = true
        }
    }

    /// A route change with no interruption in flight: the engine stopped on a
    /// configuration change (e.g. headphones unplugged), so start it again in
    /// the new input format. The old device being gone is handled by the
    /// engine itself (it re-reads the route); our job is to notice and re-tap.
    private func routeChanged() {
        guard isRecording, !held, !(engine?.isRunning ?? false) else { return }
        debugLog("recorder: engine configuration changed, restarting")
        do { try resume() }
        catch {
            debugLog("recorder: restart after route change failed: \(error.localizedDescription)")
            recordingLost("The audio route changed and the microphone could not be recovered, so the recording was stopped.")
        }
    }

    /// The media server was reset while recording: the engine object is now
    /// invalid (its underlying units point at a dead server) and the session
    /// lost its category/activation. Rebuild both from scratch and re-tap; the
    /// CAF file keeps its format, so the recording continues into the same
    /// file as if nothing had happened. If the rebuild fails, finalize honestly
    /// through the existing onRecordingLost path instead of stalling.
    private func mediaServicesReset() {
        guard isRecording else { return }
        debugLog("recorder: media services were reset, rebuilding engine")
        engine?.inputNode.removeTap(onBus: 0)
        engine?.stop()
        engine = nil
        converter = nil
        held = false
        paused = true
        do {
            let newEngine = AVAudioEngine()
            try AVAudioSession.sharedInstance().setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
            try AVAudioSession.sharedInstance().setActive(true)
            engine = newEngine
            // Re-attach the configuration-change observer to the NEW engine (the
            // old one's observer watched a dead object) and track it in `observers`
            // so stop() removes it with the rest.
            let center = NotificationCenter.default
            let token = center.addObserver(forName: .AVAudioEngineConfigurationChange, object: newEngine, queue: .main) {
                [weak self] _ in self?.routeChanged()
            }
            observers.append(token)
            try tapAndStart(engine: newEngine)
            paused = false
        } catch {
            debugLog("recorder: rebuild after media reset failed: \(error.localizedDescription)")
            recordingLost("The device's audio services restarted and the microphone could not be recovered, so the recording was stopped. The audio up to that point is saved.")
        }
    }

    /// The microphone again after an interruption or route change, in whatever
    /// format the route now has (upstream RoomAudio.resume): re-activate the
    /// session, and re-tap the input when the engine is down. The file keeps
    /// its 16 kHz format; only the tap's input format can change.
    private func resume() throws {
        held = false
        resumeWhenActive = false
        try AVAudioSession.sharedInstance().setActive(true)
        if let engine, !engine.isRunning { try tapAndStart(engine: engine) }
        paused = false
        debugLog("recorder: resumed")
    }

    /// Re-tap the input at the route's current format and start the engine.
    /// The converter is rebuilt for the new format; the file and the
    /// recognizer keep receiving 16 kHz mono as before.
    private func tapAndStart(engine: AVAudioEngine) throws {
        guard let file else { return }
        let input = engine.inputNode
        input.removeTap(onBus: 0)
        let tapFmt = input.outputFormat(forBus: 0)
        guard let conv = AVAudioConverter(from: tapFmt, to: sinkFormat) else {
            throw NSError(domain: AVFoundationErrorDomain, code: -10875,
                          userInfo: [NSLocalizedDescriptionKey: "Cannot convert mic input (\(tapFmt)) to recording format."])
        }
        converter = conv
        input.installTap(onBus: 0, bufferSize: 4096, format: tapFmt) { [weak self] buffer, _ in
            self?.handle(buffer)
        }
        engine.prepare()
        try engine.start()
    }

    /// The app became active again (scenePhase != background): a pause whose
    /// interruption ended too early to take the microphone back gets another
    /// try now (upstream's appBecameActive). Called by AppModel.
    func appBecameActive() {
        guard resumeWhenActive, isRecording, paused else { return }
        do { try resume() }
        catch {
            debugLog("recorder: resume on activation failed: \(error.localizedDescription)")
            // The audio is not coming back: keep the promise honest — the
            // partial recording is finalized instead of "recording" forever.
            recordingLost("The microphone could not be recovered after the interruption, so the recording was stopped. The audio up to that point is saved.")
        }
    }

    /// Sets the failure note and hands the loss to AppModel once: the UI must
    /// stop claiming "recording" the moment recording cannot continue.
    private func recordingLost(_ why: String) {
        guard isRecording else { return }
        debugLog("recorder: recording lost: \(why)")
        error = why
        onRecordingLost?(why)
    }

    /// Turn a mid-run SFSpeechRecognizer error into something a person can
    /// act on. The never-started case (model missing) is handled at the call
    /// site; this covers captions that worked and then stopped.
    static func describeRecognitionError(_ e: Error) -> String {
        let ns = e as NSError
        return "Live captions stopped: \(ns.localizedDescription) The recording itself is unaffected; the transcript will be generated when online."
    }

    /// Recognition-task cancellation (stop()): delivered to the callback as an
    /// error, but not a failure. Identified by the cancel code in the
    /// recognizer's own domain, or by the Cocoa cancellation code / Swift's
    /// CancellationError as fallbacks.
    static func isCancellation(_ e: Error) -> Bool {
        let ns = e as NSError
        if ns.domain == "SFSpeechRecognizerErrorDomain", ns.code == 216 { return true }
        if ns.domain == NSCocoaErrorDomain, ns.code == NSUserCancelledError { return true }
        return (e as? CancellationError) != nil
    }

    /// Segments closed so far, taken and cleared (AppModel reads on each arrive).
    func takeVoiceSegments() -> [VoiceSegment] {
        let out = voiceSegments
        voiceSegments = []
        return out
    }

    /// One tap buffer → converted 16 kHz buffers → file, level, live captions and the voice gate.
    /// Per-tap buffer allocation is bounded churn, not growth (each is freed when
    /// the callback returns); reusing one buffer across taps was considered and
    /// rejected: SFSpeechAudioBufferRecognitionRequest.append's copy/retain
    /// semantics are not documented for iOS 16, and a retained buffer rewritten
    /// by the next tap would corrupt the live captions. Recording integrity
    /// first — the malloc is cheap, the risk was not.
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
        // Flush discipline, stated honestly: AVAudioFile offers no explicit
        // flush API, and none is needed for this sink — each write(from:) goes
        // through ExtAudioFile's convert-then-write, and for LinearPCM CAF the
        // packets reach the kernel's page cache on every call (no trailing
        // container structures are held back for finalize, unlike MP4/AAC).
        // A jetsam kill (SIGKILL) cannot lose page-cache data, so the file is
        // readable up to the last tap buffer that completed — a sub-second
        // loss window at most. Only device power loss could take more, which
        // no app-level flushing can prevent.
        // Footprint trail (review-1 P1-2): if the process is jetsam-killed in the
        // background, the last log lines must show whether memory was climbing.
        // One line per ~30 s: elapsed and the remaining memory headroom via
        // os_proc_available_memory (what jetsam compares against its limit).
        // No transcript content.
        if Date().timeIntervalSince(lastFootprintLog) >= 30 {
            lastFootprintLog = Date()
            let elapsed = Date().timeIntervalSince(startedAt)
            let headroom = os_proc_available_memory()
            debugLog(String(format: "recorder: %ld s elapsed, %ld KB memory headroom", Int(elapsed), Int(headroom / 1024)))
        }
    }
    private var lastFootprintLog = Date()

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
        observers.forEach(NotificationCenter.default.removeObserver); observers = []
        engine?.inputNode.removeTap(onBus: 0); engine?.stop(); engine = nil; converter = nil
        held = false; resumeWhenActive = false
        request?.endAudio(); task?.cancel(); request = nil; task = nil; recognizer = nil
        // recognitionError stays readable until stopRecording() (AppModel)
        // moves it onto the session as liveCaptionNote; start() clears it.
        // A voice still open when Stop was tapped is a segment too (its pause never came).
        // The tap is gone (removeTap above), so the gate state is all ours now, but take
        // the lock anyway: a tap callback already past removeTap may still be finishing.
        gateLock.lock()
        closeLockedSegment()
        gateLock.unlock()
        isRecording = false; level = 0; paused = false
        debugLog("recorder: stopped after \(Int(Date().timeIntervalSince(startedAt))) s")
        let url = file?.url
        file = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
        return url
    }

    /// The length of a recording's audio file, in seconds. Killed-recording
    /// recovery (AppModel at launch): the real duration of a session whose
    /// Stop never ran, derived from the file the recorder kept writing to
    /// (upstream SessionStore.load's partial-recording rule). 0 when the file
    /// is missing or unreadable.
    static func audioSeconds(at url: URL) -> Double {
        guard let f = try? AVAudioFile(forReading: url) else { return 0 }
        return Double(f.length) / f.processingFormat.sampleRate
    }
}
