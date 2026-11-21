import Foundation
import Combine

/// Ordered, bounded-in-memory live processing for an active microphone meeting.
/// The Recorder remains the owner of the audio tap; this object only receives
/// already-closed VAD segments and persists the resulting text.
@MainActor
final class LiveMeetingCoordinator: ObservableObject {
    @Published private(set) var lines: [TranscriptLine] = []
    @Published private(set) var notes: MeetingNotes?
    @Published private(set) var problem: String?
    @Published private(set) var processing = false
    @Published private(set) var notesUpdating = false
    @Published private(set) var accepting = false

    private var sessionID: String?
    var activeSessionID: String? { sessionID }
    private var queuedSegments = 0
    private var startedAt = Date()
    private var target = "en"
    private var interpretation = true
    private var generation = UUID()
    private var tail: Task<Void, Never>?
    private var notesTask: Task<Void, Never>?
    private var notesRequestTask: Task<Void, Never>?
    private var activeTasks: [UUID: Task<Void, Never>] = [:]
    private var lastNotesLineCount = 0
    private var notesAttempted = false
    private var lastNotesAttempt = Date.distantPast
    private var onUpdate: (([TranscriptLine], MeetingNotes?) -> Void)?

    var hasText: Bool { lines.contains { !$0.src.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty } }

    func begin(sessionID: String, startedAt: Date, ai: AIClient,
               onUpdate: @escaping ([TranscriptLine], MeetingNotes?) -> Void) {
        stop()
        self.sessionID = sessionID
        self.startedAt = startedAt
        target = AISettings.targetLang
        interpretation = AISettings.simultaneousInterpretation
        generation = UUID()
        queuedSegments = 0
        lastNotesLineCount = 0
        notesAttempted = false
        lastNotesAttempt = Date.distantPast
        problem = nil
        lines = []
        notes = nil
        accepting = true
        self.onUpdate = onUpdate
        notesTask = Task { [weak self, weak ai] in
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 5_000_000_000)
                guard !Task.isCancelled, let self, let ai else { return }
                self.requestNotesIfDue(ai: ai, force: false, token: self.generation)
            }
        }
    }

    /// Queue one segment behind the previous one. Requests are deliberately
    /// serialized so a slow reply cannot reorder subtitles in the meeting page.
    func append(_ segment: VoiceSegment, fallback: String, ai: AIClient) {
        guard accepting, !segment.samples.isEmpty else { return }
        let lineID = UUID().uuidString
        let caption = fallback.trimmingCharacters(in: .whitespacesAndNewlines)
        if !caption.isEmpty {
            var line = TranscriptLine(id: lineID, t: segment.t, speaker: 1, src: caption)
            line.srcLang = "auto"
            line.dstLang = target
            lines.append(line)
            lines.sort { $0.t < $1.t }
            publish()
        }
        guard AISettings.isConfigured else {
            if caption.isEmpty { problem = "Live subtitles need the AI service configured in Settings." }
            return
        }
        guard queuedSegments < 4 else {
            problem = "Live processing is behind; the recording is safe and final transcription will catch up after Stop."
            return
        }
        let token = generation
        queuedSegments += 1
        let previous = tail
        let taskID = UUID()
        processing = true
        let task = Task { [weak self, weak ai] in
            defer {
                if let self, self.generation == token {
                    self.activeTasks.removeValue(forKey: taskID)
                    self.queuedSegments = max(0, self.queuedSegments - 1)
                    self.processing = self.queuedSegments > 0
                }
            }
            await previous?.value
            guard !Task.isCancelled, let self, let ai,
                  self.generation == token else { return }
            await self.process(segment, lineID: lineID, ai: ai, token: token)
        }
        activeTasks[taskID] = task
        tail = task
    }

    /// Keep the saved draft, but cancel live work before the formal file pass.
    /// Stop must not wait through several queued network timeouts.
    func finishAccepting() {
        accepting = false
        generation = UUID()
        notesTask?.cancel()
        notesTask = nil
        notesRequestTask?.cancel()
        notesRequestTask = nil
        activeTasks.values.forEach { $0.cancel() }
        activeTasks.removeAll()
        tail = nil
        queuedSegments = 0
        processing = false
        notesUpdating = false
    }

    func waitForPending() async {
        await tail?.value
        processing = false
    }

    func stop() {
        generation = UUID()
        accepting = false
        notesTask?.cancel()
        notesTask = nil
        notesRequestTask?.cancel()
        notesRequestTask = nil
        activeTasks.values.forEach { $0.cancel() }
        activeTasks.removeAll()
        tail = nil
        processing = false
        sessionID = nil
        onUpdate = nil
        lines = []
        notes = nil
        problem = nil
        notesUpdating = false
    }

    private func process(_ segment: VoiceSegment, lineID: String, ai: AIClient, token: UUID) async {
        do {
            let source = try await ai.transcribeLive(samples: segment.samples)
                .trimmingCharacters(in: .whitespacesAndNewlines)
            guard generation == token, !Task.isCancelled, !source.isEmpty, containsSpeech(source) else { return }
            var line = lines.first(where: { $0.id == lineID })
                ?? TranscriptLine(id: lineID, t: segment.t, speaker: 1, src: source)
            line.src = source
            line.srcLang = "auto"
            line.dstLang = target
            if let i = lines.firstIndex(where: { $0.id == lineID }) {
                lines[i] = line
            } else {
                lines.append(line)
                lines.sort { $0.t < $1.t }
            }
            publish()
            if interpretation, target != "auto" {
                do {
                    let translated = try await ai.translate(source, to: target)
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                    guard generation == token, !Task.isCancelled else { return }
                    if isUsableTranslation(translated, source: source) {
                        line.dst = translated
                        line.dstLang = target
                        if let i = lines.firstIndex(where: { $0.id == line.id }) {
                            lines[i] = line
                            publish()
                        }
                    }
                } catch {
                    guard generation == token, !Task.isCancelled else { return }
                    problem = "Live translation is unavailable; the original text is still shown."
                }
            }
            requestNotesIfDue(ai: ai, force: false, token: token)
        } catch is CancellationError {
            return
        } catch {
            guard generation == token else { return }
            problem = "Live subtitles are temporarily unavailable; recording continues."
        }
    }

    private func requestNotesIfDue(ai: AIClient?, force: Bool, token: UUID) {
        guard accepting, hasText, generation == token else { return }
        let now = Date()
        let elapsed = now.timeIntervalSince(startedAt)
        let enoughForFirst = lines.count >= CicadaTuning.firstNotesLines || elapsed >= CicadaTuning.firstNotesWait
        let due = !notesAttempted ? enoughForFirst
            : (lines.count > lastNotesLineCount && now.timeIntervalSince(lastNotesAttempt) >= CicadaTuning.notesInterval)
        guard force || due, notesRequestTask == nil, let ai,
              AIClient.Side.notes.isConfigured else { return }
        notesUpdating = true
        notesAttempted = true
        lastNotesAttempt = now
        let requestedLineCount = lines.count
        lastNotesLineCount = requestedLineCount
        let transcript = lines.map { line in
            let original = line.src.trimmingCharacters(in: .whitespacesAndNewlines)
            let translation = line.dst.map { " / \($0)" } ?? ""
            return "\(TranscriptExport.clock(line.t)): \(original)\(translation)"
        }.joined(separator: "\n")
        notesRequestTask = Task { [weak self, weak ai] in
            guard let self, let ai else { return }
            do {
                let result = try await ai.meetingNotes(transcript: transcript)
                if self.generation == token, !result.isEmpty, !Task.isCancelled, self.accepting {
                    self.notes = result
                    self.lastNotesLineCount = requestedLineCount
                    self.publish()
                }
            } catch {
                // Keep the last good draft; the next notes interval can retry.
            }
            if self.generation == token {
                self.notesUpdating = false
                self.notesRequestTask = nil
            }
        }
    }

    private func publish() {
        onUpdate?(lines, notes)
        objectWillChange.send()
    }

    private func containsSpeech(_ text: String) -> Bool {
        text.unicodeScalars.contains { CharacterSet.letters.contains($0) || CharacterSet.decimalDigits.contains($0) }
    }

    private func isUsableTranslation(_ text: String, source: String) -> Bool {
        guard !text.isEmpty, text.caseInsensitiveCompare(source) != .orderedSame else { return false }
        return containsSpeech(text)
    }
}
