import AVFoundation
import SwiftUI
import Combine
import Foundation

/// Deterministic, extractable checks only; these do not establish semantic validity.
enum FaceToFacePolicy {
    static let maximumTurns = 50
    static let maximumTextCharacters = 4_000
    static let minimumRMS = 0.003 // Roughly -50 dBFS; not a speech detector.

    struct CaptureMetrics {
        let seconds: Double
        let rms: Double
        let signalSeconds: Double
    }

    static func acceptsAudio(_ metrics: CaptureMetrics) -> Bool {
        metrics.seconds.isFinite && metrics.rms.isFinite && metrics.signalSeconds.isFinite
            && metrics.seconds >= 0.5 && metrics.rms >= minimumRMS
            && metrics.signalSeconds >= 0.12
    }

    static func allowsDirection(source: String, target: String) -> Bool {
        let supported = ["en", "zh", "ja", "ko"]
        return supported.contains(target) && (source == "auto" || supported.contains(source))
            && source != target
    }

    private static func normalized(_ value: String) -> String {
        String(value.lowercased().unicodeScalars.filter {
            CharacterSet.letters.contains($0) || CharacterSet.decimalDigits.contains($0)
        })
    }

    static func containsSpeech(_ value: String) -> Bool {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.count <= maximumTextCharacters else { return false }
        let key = normalized(trimmed)
        guard !key.isEmpty else { return false }
        let sentinels: Set<String> = [
            "blankaudio", "noaudio", "noaudiodetected", "nospeech", "nospeechdetected",
            "nospeechrecognized", "nospeechwasrecognized", "silence", "silent",
            "inaudible", "unintelligible", "music", "error", "unknownerror",
            "translationfailed", "transcriptionfailed", "translationerror", "transcriptionerror",
            "unabletotranslate", "unabletotranscribe", "notranslation", "notranscription",
            "notranslationavailable", "notranscriptionavailable", "notranslationreturned",
            "未检测到语音", "没有检测到语音", "未识别到语音", "无语音", "静音",
            "音声なし", "音声が検出されませんでした", "無音",
            "음성없음", "음성이감지되지않았습니다"
        ]
        guard !sentinels.contains(key) else { return false }
        let lower = trimmed.lowercased()
        return !["error:", "[error]", "translation error:", "transcription error:",
                 "translation failed:", "transcription failed:", "{\"error\""].contains {
            lower.hasPrefix($0)
        }
    }

    static func usableTranslation(_ value: String, source: String) -> Bool {
        containsSpeech(source) && containsSpeech(value) && normalized(value) != normalized(source)
    }
}

struct FaceTurn: Identifiable {
    let id: UUID
    let source: String
    var translation: String?
    var translationFailed = false
    let sourceLanguage: String
    let targetLanguage: String
}

private struct FaceLanguage: Identifiable {
    let code: String
    let title: String
    var id: String { code }
}

/// Writes one held utterance to a temporary CAF. The audio engine's tap is the
/// sole writer; the lock only serializes the release path with its last callback.
private final class FaceCaptureSink: @unchecked Sendable {
    private let lock = NSLock()
    private var file: AVAudioFile?
    private let sampleRate: Double
    private var frames: AVAudioFramePosition = 0
    private var writeError: Error?
    private var squaredSamples = 0.0
    private var sampleCount = 0
    private var signalSeconds = 0.0
    private var invalidSamples = false

    init(url: URL, format: AVAudioFormat) throws {
        sampleRate = format.sampleRate
        file = try AVAudioFile(forWriting: url, settings: format.settings,
                               commonFormat: format.commonFormat, interleaved: format.isInterleaved)
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        lock.lock(); defer { lock.unlock() }
        guard let file, writeError == nil else { return }
        // At most 1,024 float samples per callback, from the first input channel.
        // Respect interleaved buffer stride; missing/invalid float data fails closed.
        let count = Int(buffer.frameLength)
        if count > 0, let channel = buffer.floatChannelData?[0], buffer.stride > 0 {
            let step = max(1, (count + 1_023) / 1_024)
            var energy = 0.0
            var sampled = 0
            for frame in stride(from: 0, to: count, by: step) {
                let value = Double(channel[frame * buffer.stride])
                guard value.isFinite else { invalidSamples = true; continue }
                let bounded = min(1, max(-1, value))
                energy += bounded * bounded
                sampled += 1
            }
            squaredSamples += energy
            sampleCount += sampled
            if sampled > 0, sampleRate > 0,
               sqrt(energy / Double(sampled)) >= FaceToFacePolicy.minimumRMS {
                signalSeconds += Double(count) / sampleRate
            }
        } else if count > 0 {
            invalidSamples = true
        }
        do {
            try file.write(from: buffer)
            frames += AVAudioFramePosition(buffer.frameLength)
        } catch { writeError = error }
    }

    func finish() throws -> FaceToFacePolicy.CaptureMetrics {
        lock.lock(); defer { lock.unlock() }
        file = nil
        if let writeError { throw writeError }
        return FaceToFacePolicy.CaptureMetrics(
            seconds: sampleRate > 0 ? Double(frames) / sampleRate : 0,
            rms: !invalidSamples && sampleCount > 0 ? sqrt(squaredSamples / Double(sampleCount)) : 0,
            signalSeconds: signalSeconds)
    }
}

@MainActor
final class FaceToFaceCoordinator: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    @Published private(set) var turns: [FaceTurn] = []
    @Published private(set) var capturing = false
    @Published private(set) var processing = false
    @Published private(set) var speaking = false
    @Published private(set) var message: String?
    @Published var sourceLanguage = "auto"
    @Published var targetLanguage = "en"

    enum HoldOrigin { case gesture, accessibility }

    @Published private var hold: HoldOrigin?
    @Published private(set) var coolingDown = false
    private var synthesizer = AVSpeechSynthesizer()
    private var engine: AVAudioEngine?
    private var sink: FaceCaptureSink?
    private var activeUtterance: AVSpeechUtterance?
    private var captureURL: URL?
    private var ownsAudioSession = false
    private var generation = UUID()
    private var permissionTask: Task<Void, Never>?
    private var maxHoldTask: Task<Void, Never>?
    private var processingTask: Task<Void, Never>?
    private var cooldownTask: Task<Void, Never>?
    private var speechWatchdogTask: Task<Void, Never>?

    override init() {
        super.init()
        synthesizer.delegate = self
        let savedTarget = AISettings.targetLang
        if ["en", "zh", "ja", "ko"].contains(savedTarget) { targetLanguage = savedTarget }
    }

    var engaged: Bool { hold != nil || capturing || processing || speaking || coolingDown }
    var languageControlsDisabled: Bool { engaged }

    func beginHold(_ origin: HoldOrigin, ai: AIClient, allowed: Bool) {
        guard hold == nil else { return }
        guard allowed else {
            message = "Finish the current recording or import before using face-to-face mode."
            return
        }
        guard !speaking, !coolingDown else {
            message = "Wait for playback to finish before speaking again."
            return
        }
        guard !capturing, !processing else { return }
        guard AISettings.isConfigured else {
            message = "Configure the AI service before speaking."
            return
        }
        guard FaceToFacePolicy.allowsDirection(source: sourceLanguage, target: targetLanguage) else {
            message = "Choose different source and target languages before speaking."
            return
        }
        hold = origin
        message = nil
        let token = UUID()
        generation = token
        permissionTask = Task { [weak self] in
            guard !Task.isCancelled else { return }
            let granted: Bool = await withCheckedContinuation { continuation in
                AVAudioSession.sharedInstance().requestRecordPermission { continuation.resume(returning: $0) }
            }
            guard !Task.isCancelled, let self, self.generation == token, self.hold == origin else { return }
            self.permissionTask = nil
            guard granted else {
                self.hold = nil
                self.message = "Microphone access is required for face-to-face mode. Enable it in Settings."
                return
            }
            self.startCapture(ai: ai, token: token)
        }
    }

    func endHold(_ origin: HoldOrigin, ai: AIClient) {
        guard hold == origin else { return }
        hold = nil
        permissionTask?.cancel(); permissionTask = nil
        if capturing { finishCapture(ai: ai, token: generation) }
    }

    func cancelGesture() {
        if hold == .gesture { stop() }
    }

    private func startCapture(ai: AIClient, token: UUID) {
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker])
            try session.setActive(true)
            ownsAudioSession = true
            let audioEngine = AVAudioEngine()
            let input = audioEngine.inputNode
            let format = input.outputFormat(forBus: 0)
            guard format.channelCount > 0, format.sampleRate.isFinite, format.sampleRate > 0,
                  format.commonFormat == .pcmFormatFloat32 else {
                message = "The microphone has no usable input format."
                stop()
                return
            }
            let url = FileManager.default.temporaryDirectory
                .appendingPathComponent("cicada-face-\(UUID().uuidString).caf")
            captureURL = url
            let newSink = try FaceCaptureSink(url: url, format: format)
            input.installTap(onBus: 0, bufferSize: 2048, format: format) { [weak newSink] buffer, _ in
                newSink?.append(buffer)
            }
            engine = audioEngine
            sink = newSink
            try audioEngine.start()
            capturing = true
            maxHoldTask = Task { [weak self] in
                try? await Task.sleep(nanoseconds: 60_000_000_000)
                guard !Task.isCancelled, let self, self.generation == token, self.capturing else { return }
                self.hold = nil
                self.message = "Utterance limit reached; processing the captured audio."
                self.finishCapture(ai: ai, token: token)
            }
        } catch {
            stop(reason: "Could not start microphone capture: \(error.localizedDescription)")
        }
    }

    private func stopCaptureEngine() {
        if let engine {
            engine.inputNode.removeTap(onBus: 0)
            engine.stop()
        }
        engine = nil
        capturing = false
    }

    private func releaseAudioSession() {
        guard ownsAudioSession else { return }
        ownsAudioSession = false
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }

    private func finishCapture(ai: AIClient, token: UUID) {
        maxHoldTask?.cancel(); maxHoldTask = nil
        guard let fileURL = captureURL, let captureSink = sink else { stop(); return }
        stopCaptureEngine()
        sink = nil
        captureURL = nil
        releaseAudioSession()

        var handedOff = false
        defer { if !handedOff { try? FileManager.default.removeItem(at: fileURL) } }
        do {
            let metrics = try captureSink.finish()
            guard FaceToFacePolicy.acceptsAudio(metrics) else {
                message = metrics.seconds < 0.5 ? "Hold for at least half a second to capture speech."
                    : "No usable speech signal was captured. Hold to try again."
                return
            }
        } catch {
            message = "The captured audio could not be saved: \(error.localizedDescription)"
            return
        }

        processing = true
        handedOff = true
        let sourceLanguage = self.sourceLanguage
        let targetLanguage = self.targetLanguage
        processingTask = Task { [weak self] in
            defer {
                try? FileManager.default.removeItem(at: fileURL)
                if let self, self.generation == token {
                    self.processing = false
                    self.processingTask = nil
                }
            }
            guard !Task.isCancelled else { return }
            do {
                let source = try await ai.transcribe(audioURL: fileURL,
                    language: sourceLanguage == "auto" ? nil : sourceLanguage)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                guard let self, self.generation == token, !Task.isCancelled else { return }
                guard FaceToFacePolicy.containsSpeech(source) else {
                    self.message = "No speech was recognized. Hold to try again."
                    return
                }
                let turn = FaceTurn(id: UUID(), source: source, translation: nil,
                                    sourceLanguage: sourceLanguage, targetLanguage: targetLanguage)
                self.turns.append(turn)
                if self.turns.count > FaceToFacePolicy.maximumTurns { self.turns.removeFirst() }
                let index = self.turns.count - 1 // No other turn can start while processing.
                do {
                    let translated = try await ai.translate(source, to: targetLanguage)
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    guard self.generation == token, !Task.isCancelled else { return }
                    guard FaceToFacePolicy.usableTranslation(translated, source: source) else {
                        self.turns[index].translationFailed = true
                        self.message = "No distinct translation was returned; nothing was spoken."
                        return
                    }
                    self.turns[index].translation = translated
                    self.speak(translated, language: targetLanguage)
                } catch {
                    guard self.generation == token, !Task.isCancelled else { return }
                    self.turns[index].translationFailed = true
                    self.message = "Translation failed; the original text is shown and nothing was spoken."
                }
            } catch {
                guard let self, self.generation == token, !Task.isCancelled else { return }
                self.message = "Speech recognition failed: \(error.localizedDescription)"
            }
        }
    }

    private func speak(_ text: String, language: String) {
        guard let locale = Self.speechLocale(language),
              let voice = AVSpeechSynthesisVoice(language: locale) else {
            message = "No matching system voice is installed; the translation is shown but not spoken."
            return
        }
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .spokenAudio)
            try session.setActive(true)
            ownsAudioSession = true
        } catch {
            message = "System speech playback could not start; the translation is shown but not spoken."
            return
        }
        let utterance = AVSpeechUtterance(string: text)
        utterance.voice = voice
        let id = ObjectIdentifier(utterance)
        activeUtterance = utterance
        speaking = true
        synthesizer.speak(utterance)
        speechWatchdogTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 60_000_000_000)
            guard !Task.isCancelled else { return }
            self?.finishPlayback(id, stopSynthesizer: true, reason: "Speech playback timed out.")
        }
    }

    func stopPlayback() {
        guard let activeUtterance else { return }
        finishPlayback(ObjectIdentifier(activeUtterance), stopSynthesizer: true, reason: "Speech playback stopped.")
    }

    private func finishPlayback(_ id: ObjectIdentifier, stopSynthesizer: Bool = false, reason: String? = nil) {
        guard let activeUtterance, ObjectIdentifier(activeUtterance) == id else { return }
        self.activeUtterance = nil
        if stopSynthesizer { synthesizer.stopSpeaking(at: .immediate) }
        speechWatchdogTask?.cancel(); speechWatchdogTask = nil
        if let reason { message = reason }
        coolingDown = true
        speaking = false
        releaseAudioSession()
        let token = generation
        cooldownTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 400_000_000)
            guard !Task.isCancelled, let self, self.generation == token else { return }
            self.coolingDown = false
            self.cooldownTask = nil
        }
    }

    func handleInterruption(_ notification: Notification) {
        guard let raw = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt,
              AVAudioSession.InterruptionType(rawValue: raw) == .began, engaged else { return }
        stop(reason: "Audio was interrupted. Hold again when the interruption is over.")
    }

    func handleRouteChange(_ notification: Notification) {
        guard let raw = notification.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt,
              AVAudioSession.RouteChangeReason(rawValue: raw) == .oldDeviceUnavailable,
              capturing || speaking else { return }
        stop(reason: "An audio device disconnected. Hold again to use the current route.")
    }

    func handleEngineChange(_ notification: Notification) {
        guard let engine, let changed = notification.object as? AVAudioEngine, changed === engine else { return }
        stop(reason: "Microphone configuration changed. Hold again to start a new turn.")
    }

    func handleMediaReset() {
        stop(reason: engaged ? "Audio services restarted. Hold again to start a new turn." : nil)
        synthesizer.delegate = nil
        synthesizer = AVSpeechSynthesizer()
        synthesizer.delegate = self
    }

    func stop(reason: String? = nil) {
        generation = UUID()
        hold = nil
        permissionTask?.cancel(); permissionTask = nil
        maxHoldTask?.cancel(); maxHoldTask = nil
        cooldownTask?.cancel(); cooldownTask = nil
        speechWatchdogTask?.cancel(); speechWatchdogTask = nil
        processingTask?.cancel(); processingTask = nil
        stopCaptureEngine()
        _ = try? sink?.finish()
        sink = nil
        if let captureURL { try? FileManager.default.removeItem(at: captureURL) }
        captureURL = nil
        if processing, let index = turns.indices.last, turns[index].translation == nil {
            turns[index].translationFailed = true
        }
        processing = false
        if activeUtterance != nil {
            activeUtterance = nil
            synthesizer.stopSpeaking(at: .immediate)
        }
        speaking = false
        coolingDown = false
        releaseAudioSession()
        if let reason { message = reason }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor [weak self] in self?.finishPlayback(ObjectIdentifier(utterance)) }
    }

    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor [weak self] in
            self?.finishPlayback(ObjectIdentifier(utterance), reason: "Speech playback was cancelled.")
        }
    }

    private static func speechLocale(_ language: String) -> String? {
        switch language {
        case "en": return "en-US"
        case "zh": return "zh-CN"
        case "ja": return "ja-JP"
        case "ko": return "ko-KR"
        default: return nil
        }
    }
}

struct FaceToFaceView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.scenePhase) private var scenePhase
    @StateObject private var coordinator = FaceToFaceCoordinator()
    @GestureState private var touching = false
    @State private var gestureID: UUID?
    @State private var showingCancelDialog = false

    private var audioAvailable: Bool {
        !model.busy && !model.recorder.isRecording && model.importingSessionID == nil
    }

    private let languages: [FaceLanguage] = [
        FaceLanguage(code: "auto", title: "Auto detect"),
        FaceLanguage(code: "en", title: "English"),
        FaceLanguage(code: "zh", title: "Chinese"),
        FaceLanguage(code: "ja", title: "Japanese"),
        FaceLanguage(code: "ko", title: "Korean")
    ]

    var body: some View {
        NavigationStack {
            VStack(spacing: 14) {
                HStack(spacing: 10) {
                    languagePicker("I speak", selection: $coordinator.sourceLanguage,
                                   choices: languages)
                    Button {
                        let oldSource = coordinator.sourceLanguage
                        let oldTarget = coordinator.targetLanguage
                        coordinator.sourceLanguage = oldTarget
                        coordinator.targetLanguage = oldSource == "auto"
                            ? (languages.first(where: { $0.code != "auto" && $0.code != oldTarget })?.code ?? "zh")
                            : oldSource
                    } label: { Image(systemName: "arrow.left.arrow.right") }
                        .accessibilityLabel("Swap languages")
                    languagePicker("Translate to", selection: $coordinator.targetLanguage,
                                   choices: Array(languages.dropFirst()))
                }
                .disabled(coordinator.languageControlsDisabled)
                .padding(.horizontal)

                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(coordinator.turns) { turn in
                            HStack(alignment: .top, spacing: 12) {
                                VStack(alignment: .leading, spacing: 5) {
                                    Text(Self.languageName(turn.sourceLanguage)).font(.caption.bold())
                                        .foregroundStyle(Ink.secondary)
                                    Text(turn.source).font(.body)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                VStack(alignment: .trailing, spacing: 5) {
                                    Text(Self.languageName(turn.targetLanguage)).font(.caption.bold())
                                        .foregroundStyle(Ink.secondary)
                                    if let translation = turn.translation {
                                        Text(translation).font(.body.weight(.medium)).multilineTextAlignment(.trailing)
                                    } else {
                                        Text(turn.translationFailed ? "Translation unavailable" : "Translating…")
                                            .font(.footnote).foregroundStyle(Ink.secondary)
                                    }
                                }
                                .frame(maxWidth: .infinity, alignment: .trailing)
                            }
                            .padding(12)
                            .background(Ink.accentWash.opacity(0.45), in: RoundedRectangle(cornerRadius: 12))
                        }
                        if coordinator.turns.isEmpty {
                            Text("Choose the speaking direction, then hold the button and release when finished.")
                                .foregroundStyle(Ink.secondary).padding(.top, 40)
                        }
                    }
                    .padding(.horizontal, 16).padding(.vertical, 8)
                }

                if let message = coordinator.message {
                    Text(message).font(.footnote).foregroundStyle(Ink.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading).padding(.horizontal)
                }
                if coordinator.speaking {
                    Button("Stop playback") { coordinator.stopPlayback() }
                }
                if coordinator.engaged {
                    Button(role: .destructive) {
                        showingCancelDialog = true
                    } label: {
                        Label("Cancel this turn", systemImage: "xmark.circle")
                            .font(.subheadline.weight(.medium))
                    }
                    .accessibilityHint("Stops capture or playback and discards any unfinished translation")
                }
                Group {
                    VStack(spacing: 6) {
                        Image(systemName: coordinator.capturing ? "waveform" : "mic.fill")
                            .font(.system(size: 28, weight: .semibold))
                        Text(coordinator.capturing ? "Release to translate" : coordinator.speaking ? "Speaking…"
                             : coordinator.processing ? "Translating…" : coordinator.coolingDown ? "Wait a moment…" : "Hold to speak")
                            .font(.headline)
                    }
                    .foregroundStyle(.white)
                    .frame(maxWidth: .infinity, minHeight: 76)
                    .background(coordinator.capturing ? Ink.accent : Ink.ink, in: RoundedRectangle(cornerRadius: 18))
                }
                .contentShape(Rectangle())
                .gesture(DragGesture(minimumDistance: 0)
                    .updating($touching) { _, state, _ in state = true }
                    .onChanged { _ in
                        guard gestureID == nil, scenePhase == .active else { return }
                        gestureID = UUID()
                        coordinator.beginHold(.gesture, ai: model.ai, allowed: audioAvailable)
                    }
                    .onEnded { _ in
                        gestureID = nil
                        coordinator.endHold(.gesture, ai: model.ai)
                    })
                .onChange(of: touching) { active in
                    guard !active, let id = gestureID else { return }
                    // A system-cancelled drag resets GestureState without calling onEnded.
                    DispatchQueue.main.async {
                        guard gestureID == id else { return }
                        gestureID = nil
                        coordinator.cancelGesture()
                    }
                }
                .accessibilityElement(children: .ignore)
                .accessibilityAddTraits(.isButton)
                .accessibilityLabel("Hold to speak; release to translate and play")
                .accessibilityAction(named: Text("Start speaking")) {
                    guard scenePhase == .active else { return }
                    coordinator.beginHold(.accessibility, ai: model.ai, allowed: audioAvailable)
                }
                .accessibilityAction(named: Text("Stop speaking")) {
                    coordinator.endHold(.accessibility, ai: model.ai)
                }
                Text("This is push-to-talk, not hands-free two-way audio. Microphone capture stops before speech playback.")
                    .font(.caption).foregroundStyle(Ink.secondary).multilineTextAlignment(.center)
                    .padding(.horizontal, 20)
            }
            .padding(.top, 12).padding(.bottom, 8)
            .navigationTitle("Face to face")
            .navigationBarTitleDisplayMode(.inline)
            .confirmationDialog("Cancel this turn?",
                                isPresented: $showingCancelDialog,
                                titleVisibility: .visible) {
                Button("Cancel turn", role: .destructive) {
                    coordinator.stop(reason: "Turn cancelled. Hold to speak again.")
                }
                Button("Keep going", role: .cancel) {}
            } message: {
                Text("Stops microphone capture or speech playback and discards any unfinished translation. Previous turns are kept.")
            }
            .onChange(of: coordinator.engaged) { active in
                if !active { showingCancelDialog = false }
            }
            .toolbar {
                ToolbarItem(placement: .navigationBarLeading) {
                    Button("Done") {
                        coordinator.stop()
                        model.showingFaceToFace = false
                    }
                }
            }
            .onChange(of: scenePhase) { phase in
                if phase != .active { coordinator.stop() }
            }
            .onChange(of: audioAvailable) { available in
                if !available { coordinator.stop() }
            }
            .onReceive(NotificationCenter.default.publisher(for: AVAudioSession.interruptionNotification)
                .receive(on: DispatchQueue.main)) { coordinator.handleInterruption($0) }
            .onReceive(NotificationCenter.default.publisher(for: AVAudioSession.routeChangeNotification)
                .receive(on: DispatchQueue.main)) { coordinator.handleRouteChange($0) }
            .onReceive(NotificationCenter.default.publisher(for: .AVAudioEngineConfigurationChange)
                .receive(on: DispatchQueue.main)) { coordinator.handleEngineChange($0) }
            .onReceive(NotificationCenter.default.publisher(for: AVAudioSession.mediaServicesWereResetNotification)
                .receive(on: DispatchQueue.main)) { _ in coordinator.handleMediaReset() }
            .onDisappear { coordinator.stop() }
        }
    }

    private func languagePicker(_ title: String, selection: Binding<String>,
                                choices: [FaceLanguage]) -> some View {
        Menu {
            ForEach(choices) { language in
                Button(language.title) { selection.wrappedValue = language.code }
            }
        } label: {
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.caption).foregroundStyle(Ink.secondary)
                Text(languages.first(where: { $0.code == selection.wrappedValue })?.title ?? "English")
                    .font(.subheadline.weight(.medium)).foregroundStyle(Ink.ink).lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(10).background(Ink.line.opacity(0.35), in: RoundedRectangle(cornerRadius: 10))
        }
    }

    private static func languageName(_ code: String) -> String {
        switch code {
        case "auto": return "Detected speech"
        case "en": return "English"
        case "zh": return "Chinese"
        case "ja": return "Japanese"
        case "ko": return "Korean"
        default: return code
        }
    }
}
