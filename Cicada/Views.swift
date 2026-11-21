import SwiftUI
import AVFoundation
import Combine
import Network
import UIKit
import UniformTypeIdentifiers

// MARK: - App model wiring record / transcribe / translate / notes

@MainActor
final class AppModel: ObservableObject {
    @Published var store = SessionStore()
    @Published var ai = AIClient()
    @Published var recorder = Recorder()
    /// Auto-save Markdown to the chosen folder (item 3): owns the bookmark's
    /// lifecycle; its failures surface through `message`, never through the
    /// pipeline.
    @Published var exporter = FolderExporter()
    @Published var tab: Tab = .records
    @Published var selection: String?
    @Published var busy = false
    @Published var message: String?
    @Published var liveMeeting = LiveMeetingCoordinator()
    @Published var showingLiveMeeting = false
    @Published var showingFaceToFace = false
    @Published private(set) var importingSessionID: String?
    private var importCancellation: MediaImportCancellation?
    private var importJobTask: Task<Void, Never>?
    enum Tab { case records, settings }

    var selected: Session? { store.sessions.first { $0.id == selection } }
    /// The session currently being recorded (kept out of `selection` so the user
    /// stays where the Stop control is; `selection` is set only when it finishes).
    @Published var recordingId: String?

    /// Set by the background-task expiration handler: runPipeline checks it at
    /// each step boundary and saves the partial progress reached so far.
    private var pipelineStopRequested = false
    private var pipelineCancelled: Bool {
        get { pipelineStopRequested || Task.isCancelled || importCancellation?.isCancelled == true }
        set { pipelineStopRequested = newValue }
    }
    /// Active UIApplication background task for the pipeline, ended when it
    /// finishes (or left to the expiration handler if the app is torn down).
    private var backgroundTaskId: UIBackgroundTaskIdentifier = .invalid
    /// True while a pipeline is in flight (set synchronously here, unlike
    /// `busy`, which runPipeline sets after its first await — too late to close
    /// the overlap race between two process() calls).
    private var pipelineActive = false
    private var startingRecording = false

    private var subs = Set<AnyCancellable>()
    /// Speaker separation: the voiceprint engine (SpeakerID.swift). Segments' audio is
    /// embedded as they close during the recording; the grouping happens at stop, in
    /// segment order.
    private let speakers = SpeakerTagger()
    /// Voice segments closed so far in the current recording: id, start time, the audio's
    /// length in frames (for the clusterer's seconds), the voiceprint (nil until it comes
    /// back — `unsettled` still true means it is on the work queue), and the caption words
    /// said in it.
    private struct PendingSeg {
        let id: String
        let t: Double
        let samplesFrames: Int
        var emb: [Float]?
        var unsettled = true
        var caption = ""
    }
    private var pendingSegs: [PendingSeg] = []
    private var cumulativeFallbackCaptions: [String: String] = [:]
    /// `true` while the model is known missing/unloadable: skip the per-segment embedding
    /// work from then on (graceful degradation — every line keeps speaker 1).
    private var voiceprintsUnavailable = false

    // Offline usability (2026-11-11 user report: recorded with no signal, got
    // nothing). Cloud transcription needs the network; when it fails, the
    // session sits with transcribed == false until someone taps ↻. Watch the
    // network and, when it comes back, re-run the pipeline on the sessions
    // the recording of which predates this connectivity event.
    private var pathMonitor: NWPathMonitor?
    private var monitorQueue = DispatchQueue(label: "cicada.netmonitor")
    /// Current connectivity, from the NWPathMonitor. The detail page uses it to
    /// say WHY a stopped session has no transcript (offline) and that the
    /// transcript will be filled in automatically on reconnect.
    @Published private(set) var networkAvailable = true
    /// Set once the monitor delivers its first verdict, so the initial
    /// callback (always sent at startup, whatever the state) is not mistaken
    /// for a recovery and triggers a pointless retry pass.
    private var sawFirstPath = false
    /// Sessions already retried for the current connectivity window. Cleared
    /// when the network drops again, so the NEXT recovery retries them once
    /// more — but within one window each session gets exactly one shot.
    private var autoRetriedIds: Set<String> = []

    init() {
        // store and recorder are nested ObservableObjects; their changes don't fire
        // AppModel.objectWillChange on their own, so forward them (elapsed, level,
        // liveText and isRecording would otherwise never re-render). ai too: the
        // Settings check status (AIClient.status) would never re-render without it.
        recorder.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &subs)
        store.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &subs)
        ai.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &subs)
        // Folder export (item 3): the Settings rows read FolderSettings
        // through the exporter; its change (folder chosen / forgotten) must
        // re-render them, which needs this forward like the others above.
        exporter.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &subs)
        liveMeeting.objectWillChange
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &subs)
        // Recording errors (disk full, write failure) are set on Recorder.error
        // from tap/interruption contexts that have no UI of their own; surface
        // them as the app-level alert. objectWillChange fires before the change
        // lands, so read the value on the next run-loop turn.
        recorder.objectWillChange
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in self?.surfaceRecorderError() }
            .store(in: &subs)
        // Speaker separation: the recorder's voice gate closes segments while the
        // recording runs; embed each one's audio as it arrives (voiceSegmentArrived),
        // so stop-time work is only the grouping. objectWillChange fires before the
        // change lands, so read the segments on the next run-loop turn.
        recorder.objectWillChange
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in self?.drainVoiceSegments() }
            .store(in: &subs)
        // A session persisted as .running means the app was killed mid-pipeline
        // (background time ran out or iOS terminated us): no Task is working on
        // it anymore. Mark it failed so the Notes tab shows the reason instead
        // of "Generating…" forever.
        for i in store.sessions.indices where store.sessions[i].processingState == .running {
            store.sessions[i].processingState = .failed
            store.sessions[i].lastError = "Processing was interrupted before it finished."
            if store.sessions[i].isImported && store.sessions[i].importState == .processing {
                store.sessions[i].importState = .paused
            }
            store.save(store.sessions[i])
        }
        for i in store.sessions.indices where store.sessions[i].isImported &&
            (store.sessions[i].importState == .extracting || store.sessions[i].importState == .queued) {
            store.sessions[i].importState = .paused
            store.sessions[i].lastError = "Import was interrupted. Resume to continue from the saved source file."
            store.save(store.sessions[i])
        }
        // Killed mid-recording (upstream SessionStore.load's rule): a session
        // with audio but no endedAt was recording when the app was terminated
        // (iOS reclaimed the background app, or a crash) — Stop never ran, so
        // its JSON never got an end. Without this pass it would show as
        // "0 min · 0 lines" and look lost, while its audio is safe in
        // Recordings. Finalize it with the real duration from the audio file.
        // The recorder now writes crash-safe CAF (Recorder.start), so the file
        // a kill leaves behind IS readable up to the last flushed write; the
        // old .m4a sink often was not (unfinalized MP4 container), which is
        // why recovered sessions used to show 0 and refuse to play.
        for i in store.sessions.indices where store.sessions[i].endedAt == nil {
            guard let url = store.audioURL(for: store.sessions[i]),
                  FileManager.default.fileExists(atPath: url.path) else { continue }
            let seconds = Recorder.audioSeconds(at: url)
            guard seconds > 0 else { continue }
            store.sessions[i].endedAt = store.sessions[i].startedAt.addingTimeInterval(seconds)
            store.save(store.sessions[i])
            debugLog("recovered session \(store.sessions[i].id) killed mid-recording: \(Int(seconds)) s of audio")
        }
        // A recovered session whose audio is unreadable (a pre-CAF .m4a the kill
        // left unfinalized — the 2026-11-14 field report's exact case — or a lost
        // file) must say so honestly instead of showing as an empty record the
        // user cannot tell from silence: the file exists but its duration reads
        // 0, so nothing can be played or transcribed from it. processingState
        // .failed is what makes the Notes tab render lastError at all.
        for i in store.sessions.indices where store.sessions[i].endedAt == nil {
            guard store.sessions[i].audioFile != nil else { continue }
            guard let url = store.audioURL(for: store.sessions[i]),
                  FileManager.default.fileExists(atPath: url.path),
                  Recorder.audioSeconds(at: url) <= 0 else { continue }
            store.sessions[i].endedAt = store.sessions[i].startedAt
            store.sessions[i].processingState = .failed
            store.sessions[i].lastError = "This recording's audio file is damaged (the app was terminated while it was being written, before the crash-safe format). It cannot be played or transcribed."
            store.save(store.sessions[i])
            debugLog("recovered session \(store.sessions[i].id) has unreadable audio; marked honestly")
        }
        // Interruption handling (upstream RoomListen): when the audio cannot be
        // taken back after a call/Siri/route change, the recorder tells us here
        // and the session is finalized truthfully — the UI stops claiming
        // "recording" and the partial audio is kept, not lost behind a stuck timer.
        recorder.onRecordingLost = { [weak self] _ in
            self?.stopRecording()
        }
        // Recovered sessions (the killed-mid-recording pass above) must be
        // transcribable without a delete/re-record: the auto-retry on reconnect
        // keys on `endedAt != nil && audioFile != nil && !transcribed`, both of
        // which the pass now sets — but its `guard recordingId == nil` and the
        // `AISettings.isConfigured` check run at init time, when the network
        // monitor has not settled yet. Nothing blocks the manual ↻ in the detail
        // screen (it calls process() directly), and the network monitor's first
        // offline→online transition retries these as soon as the service is set
        // up — no change needed there, this note is the audit trail.
        //
        // Folder export problems (stale bookmark, write error) arrive after the
        // pipeline is done; show them as the app alert, once per failure text.
        exporter.onSurface = { [weak self] text in
            self?.message = text
        }
        // A pause whose interruption ended while the app was still behind a
        // call screen could not take the microphone back yet (upstream's
        // resumeWhenActive): try again the moment we are the front app.
        NotificationCenter.default.addObserver(forName: UIApplication.didBecomeActiveNotification,
                                               object: nil, queue: .main) { [weak self] _ in
            self?.recorder.appBecameActive()
            if let self { self.exporter.retryPending(sessions: self.store.sessions) }
        }
        exporter.retryPending(sessions: store.sessions)
        startNetworkMonitor()
    }

    /// The recording error surfaced by the last pass (surfaceRecorderError), so
    /// a value still on Recorder.error is not re-alerted on every change.
    private var surfacedRecorderError: String?

    /// Show Recorder.error as the app-level message, once per value: the tap
    /// write-failure path (disk full) and recordingLost set it from contexts
    /// that cannot present anything themselves. onRecordingLost already calls
    /// stopRecording() for the unrecoverable cases; this makes the reason
    /// visible either way.
    private func surfaceRecorderError() {
        // The error cleared (a new recording started): arm the dedup again so
        // the same failure text in a later recording still alerts.
        guard let e = recorder.error else { surfacedRecorderError = nil; return }
        guard e != surfacedRecorderError else { return }
        surfacedRecorderError = e
        message = e
    }

    // MARK: - Network recovery: auto-retry cloud transcription when signal returns

    /// Foreground-only NWPathMonitor (no background audio / voip entitlements
    /// touched: the app is not allowed to run for network changes in the
    /// background, and we don't need it to — the retry can wait until the user
    /// opens the app, which is when this monitor is live again). The first
    /// callback is the current state, not a transition: it only records the
    /// baseline. Only an unsatisfied → satisfied transition triggers retries.
    private func startNetworkMonitor() {
        let m = NWPathMonitor()
        m.pathUpdateHandler = { [weak self] path in
            DispatchQueue.main.async { self?.pathChanged(path) }
        }
        m.start(queue: monitorQueue)
        pathMonitor = m
    }

    private func pathChanged(_ path: NWPath) {
        let now = path.status == .satisfied
        let first = !sawFirstPath
        let wasAvailable = networkAvailable
        sawFirstPath = true
        networkAvailable = now
        if first || now == wasAvailable { return }   // baseline or no transition
        if !now {
            autoRetriedIds = []   // a new offline window opens a new retry budget
            return
        }
        retryUntranscribedAfterReconnect()
    }

    /// Network just came back: re-run process() on every stopped session that
    /// has audio but no server transcription. runPipeline's `if !s.transcribed`
    /// skip keeps already-succeeded steps from re-running, so this only costs
    /// the transcription that never happened. One retry per session per
    /// connectivity window (autoRetriedIds, cleared on the next drop): a
    /// server that is still unreachable fails into lastError as before, and
    /// the next offline→online cycle takes another single shot at it.
    private func retryUntranscribedAfterReconnect() {
        // Never while a recording is in flight: the pipeline would race the
        // recorder and the about-to-start processing of the live session.
        guard recordingId == nil, !recorder.isRecording else { return }
        // If a pipeline is already running, the user beat us to ↻ — don't
        // queue a second one behind it (process() would drop it anyway).
        guard !pipelineActive, !busy, !startingRecording else { return }
        guard AISettings.isConfigured else { return }
        // Mark them all up front: a later re-entrant call (the monitor can
        // fire more than once per window) must not double-book a session.
        let todo = store.sessions.filter {
            !$0.isImported && $0.endedAt != nil && $0.audioFile != nil && !$0.transcribed && !autoRetriedIds.contains($0.id)
        }
        guard !todo.isEmpty else { return }
        autoRetriedIds.formUnion(todo.map(\.id))
        // Sequentially: process() serializes via pipelineActive and drops
        // concurrent runs, so one Task per session would drop all but the
        // first. One Task awaiting them in order runs each to completion.
        Task { [weak self] in
            for s in todo {
                await self?.process(s)
            }
        }
    }

    /// Segments the gate closed since the last drain: embed each one's audio off the
    /// main thread right away — a long meeting would otherwise hold every segment's
    /// raw audio in memory until stop (20 s of 16 kHz mono is 1.2 MB; an embedding
    /// is 1 KB). Runs during the recording via the recorder's change stream, and
    /// once more from applySpeakers after stop (the guard lets that last drain in).
    private func drainVoiceSegments() {
        guard recordingId != nil || !recorder.voiceSegments.isEmpty else { return }
        for seg in recorder.takeVoiceSegments() { voiceSegmentArrived(seg) }
    }

    private func voiceSegmentArrived(_ seg: VoiceSegment) {
        // The live coordinator receives the same closed segment as the voiceprint
        // path. It never owns the tap and has a small bounded queue, so recording
        // remains safe if the service is slow or unavailable.
        let start = max(0, seg.t - CicadaTuning.preRoll)
        let end = seg.t + Double(seg.samples.count) / CicadaTuning.sampleRate + 0.4
        let fallback = recorder.timedWords.filter { $0.mid >= start && $0.mid <= end }
            .map(\.text).joined(separator: " ")
        liveMeeting.append(seg, fallback: fallback, ai: ai)
        let id = UUID().uuidString
        var p = PendingSeg(id: id, t: seg.t, samplesFrames: seg.samples.count)
        guard !voiceprintsUnavailable else {
            p.unsettled = false   // no model: nothing is coming, don't wait for it
            pendingSegs.append(p)
            return
        }
        pendingSegs.append(p)
        speakers.embed(seg.samples) { [weak self] emb in
            guard let self else { return }
            if !self.speakers.available { self.voiceprintsUnavailable = true }
            if let i = self.pendingSegs.firstIndex(where: { $0.id == id }) {
                self.pendingSegs[i].emb = emb
                self.pendingSegs[i].unsettled = false
            }
        }
    }

    func startRecording() async {
        guard importingSessionID == nil, !busy, !pipelineActive, !startingRecording, recordingId == nil, !recorder.isRecording else { return }
        startingRecording = true
        defer { startingRecording = false }
        message = nil
        guard await recorder.requestPermissions() else {
            message = "Microphone / speech permission denied. Enable the microphone in Settings › Privacy › Microphone."
            return
        }
        // Speaker separation: compile/load the voiceprint model before the engine
        // starts, not on the first voice segment mid-recording (review-1 P1-2: the
        // first-segment compile was a background memory spike feeding jetsam).
        speakers.warmUp()
        var s = Session(); s.startedAt = Date()
        // .caf, not .m4a: the recorder's crash-safe sink (see Recorder.start).
        // The extension is what recovery (audioURL), playback (AVAudioPlayer) and
        // the upload mime all key off, so it must match the format actually written.
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Cicada/Recordings/\(s.id).caf")
        s.audioFile = url.lastPathComponent
        do {
            try recorder.start(audioURL: url)
            store.save(s)
            recordingId = s.id
            // Speaker separation: a fresh recording starts with fresh voices.
            speakers.reset()
            pendingSegs = []
            voiceprintsUnavailable = false
            let liveID = s.id
            liveMeeting.begin(sessionID: liveID, startedAt: s.startedAt, ai: ai) { [weak self] lines, notes in
                self?.persistLive(sessionID: liveID, lines: lines, notes: notes)
            }
            showingLiveMeeting = true
        }
        catch { message = "Recording could not start: \(error.localizedDescription)" }
    }
    private func persistLive(sessionID: String, lines: [TranscriptLine], notes: MeetingNotes?) {
        guard liveMeeting.activeSessionID == sessionID,
              var current = store.sessions.first(where: { $0.id == sessionID }) else { return }
        current.lines = lines
        if let notes { current.notes = notes }
        store.save(current)
    }

    func stopRecording() {
        _ = recorder.stop()
        // stop() flushes its last VAD segment; drain synchronously before closing
        // the live queue so the final utterance is not lost to the next run loop.
        drainVoiceSegments()
        liveMeeting.finishAccepting()
        showingLiveMeeting = false
        // Look up by recordingId, not `selected`: stopping happens on the list,
        // where selection is nil and the old lookup silently dropped endedAt.
        guard let id = recordingId, var s = store.sessions.first(where: { $0.id == id }) else {
            recordingId = nil; liveMeeting.stop(); return
        }
        recordingId = nil
        s.endedAt = Date()
        // Keep live bilingual lines and their draft notes. The on-device caption
        // remains a fallback only when no live/cloud line was saved.
        let live = recorder.liveText; recorder.liveText = ""
        if !live.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            cumulativeFallbackCaptions[id] = live
            if s.lines.isEmpty {
                s.lines = [TranscriptLine(t: 0, speaker: 1, src: live)]
            }
        }
        // Same for the caption failure reason: with it on the session, the
        // transcript page can say why there are no lines at all (offline
        // model not downloaded) instead of showing bare "No transcript yet."
        s.liveCaptionNote = recorder.recognitionError
        recorder.recognitionError = nil
        store.save(s)
        exporter.markPending(s)
        selection = s.id
        // Drain live segment requests before the whole-file pipeline so the two
        // passes cannot reorder or overwrite one another's intermediate saves.
        busy = true
        Task {
            await self.liveMeeting.waitForPending()
            let latest = self.store.sessions.first(where: { $0.id == id }) ?? s
            await self.process(latest, useRecordingSegments: true)
            self.liveMeeting.stop()
        }
    }

    /// Speaker separation at stop: cluster the segments' voiceprints in recording order
    /// (online clustering — a line already labelled is never revisited), then rebuild the
    /// session's lines so each carries its voice's speaker number and the transcript text
    /// said in its segment (RecordRefiner's alignment). With no voices on show (a solo
    /// recording, or the model unavailable) the fallback line below stays as it was.
    ///
    /// The transcript at this point is the live captions fallback (one line); the server
    /// transcript comes later in process(). The alignment therefore runs here against the
    /// captions, and again in the pipeline once the server text arrives (refineTranscript).
    private func applySpeakers(_ s: Session) async {
        let hasCumulativeFallback = cumulativeFallbackSessions.remove(s.id) != nil
        // Anything the gate closed between the last drain and the flush at stop.
        drainVoiceSegments()
        guard !pendingSegs.isEmpty else { return }
        // Wait for the embeddings still in flight. `settled` marks a segment whose embed
        // came back (nil = too short, a breath or a click, not a voiceprint — it stays
        // unlabelled, as upstream); an unsettled one is still on the work queue.
        while pendingSegs.contains(where: { $0.emb == nil && $0.unsettled }) {
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        // Caption each segment with the on-device words said in it.
        let timed = recorder.timedWords
        var captioned = pendingSegs
        for i in captioned.indices {
            let t0 = captioned[i].t
            let t1 = i + 1 < captioned.count ? captioned[i + 1].t : .infinity
            captioned[i].caption = timed.filter { $0.mid >= t0 && $0.mid < t1 }
                .map(\.text).joined(separator: " ")
        }
        let segs = captioned
        pendingSegs = []
        guard var session = store.sessions.first(where: { $0.id == s.id }) else { return }
        // Group the voices in order (online clustering, upstream's rule). `label`'s
        // seconds are the segment's speaking time, not the embedding's length.
        var clusterer = VoiceClusterer()
        var voices: [Int?] = []
        for seg in segs {
            let seconds = Double(seg.samplesFrames) / CicadaTuning.sampleRate
            voices.append(seg.emb.flatMap { clusterer.label($0, seconds: seconds) })
        }
        let hasCumulativeFallback = cumulativeFallbackSessions.remove(s.id) != nil
        if !session.lines.isEmpty && !hasCumulativeFallback {
            // Keep live text intact and attach embeddings only when live lines
            // correspond closely to individual segments. A cumulative on-device
            // fallback is instead split using the segment captions below.
            if segs.count == 1, session.lines.count == 1 {
                session.lines[0].voice = voices.first ?? nil
            } else if session.lines.count > 1 {
                for lineIndex in session.lines.indices {
                    guard let match = segs.indices.min(by: {
                        abs(segs[$0].t - session.lines[lineIndex].t) < abs(segs[$1].t - session.lines[lineIndex].t)
                    }), abs(segs[match].t - session.lines[lineIndex].t) <= 1 else { continue }
                    session.lines[lineIndex].voice = voices[match]
                }
            } else {
                return
            }
            let numbers = SpeakerLabels.voiceNumbers(session.lines)
            for i in session.lines.indices {
                if let voice = session.lines[i].voice, let number = numbers[voice] {
                    session.lines[i].speaker = number
                }
            }
            store.save(session)
            return
        }
        // Lines: one per segment. A segment with no voice of its own (too short to
        // embed, or unlike every known voice) keeps voice nil — upstream ab787d4's
        // rule: a short reply in a voice not yet known does not join the answer
        // before it, so it never takes its neighbour's voice either.
        var lines: [TranscriptLine] = []
        for (i, seg) in segs.enumerated() {
            var l = TranscriptLine(t: seg.t, speaker: 1, src: seg.caption)
            l.voice = voices[i]
            lines.append(l)
        }
        if hasCumulativeFallback, let fallback = session.lines.first?.src,
           !fallback.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            lines = RecordRefiner.refine(fallback, onto: lines, duration: session.duration)
        }
        // Speaker numbers: only shown voices get one; the rest keep 1 (S1).
        let numbers = SpeakerLabels.voiceNumbers(lines)
        for i in lines.indices {
            if let v = lines[i].voice, let n = numbers[v] { lines[i].speaker = n }
        }
        // No voice was identified anywhere (the model never loaded, or a recording
        // of silence): leave the liveText fallback line as the transcript instead
        // of replacing it with caption fragments no voice was told apart in.
        guard lines.contains(where: { $0.voice != nil }) else { return }
        session.lines = lines
        store.save(session)
    }

    private func updateImportProgress(_ session: inout Session, to value: Double) {
        guard session.isImported, session.importState == .processing else { return }
        session.importProgress = min(0.99, max(session.importProgress, value))
        store.save(session)
    }

    /// AI pipeline for a finished recording: transcribe → translate → notes.
    /// Wraps runPipeline(_:) in a UIApplication background task so the pipeline
    /// can finish after the user locks the screen or switches away; every caller
    /// (Stop and the manual ↻ button) gets the background protection.
    func process(_ s: Session, retranscribe: Bool = false, useRecordingSegments: Bool = false) async {
        guard !pipelineActive, !startingRecording, !recorder.isRecording else {
            message = "Still recording or processing — try again in a moment."
            return
        }
        pipelineActive = true
        busy = true
        var taskID: UIBackgroundTaskIdentifier = .invalid
        defer {
            pipelineActive = false
            busy = importingSessionID != nil
            // Include local-caption-only, failed and interrupted runs. Export failure
            // queues a retry in FolderExporter; it never changes the pipeline result.
            if let latest = store.sessions.first(where: { $0.id == s.id }),
               latest.lines.contains(where: { !$0.src.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) {
                exporter.write(latest)
            }
            // Keep the background grant until AFTER the final write/persistent queue update.
            if backgroundTaskId == taskID {
                backgroundTaskId = .invalid
                if taskID != .invalid { UIApplication.shared.endBackgroundTask(taskID) }
            }
        }
        // Session is a value type. The caller may hold a snapshot from before the
        // speaker pass (or a previous refresh); never save that snapshot over the store.
        guard var current = store.sessions.first(where: { $0.id == s.id }) else { return }
        exporter.markPending(current)
        if retranscribe {
            guard AISettings.isConfigured else {
                message = "Set the transcription service URL and key before re-transcribing. Nothing was replaced."
                return
            }
            guard let url = store.audioURL(for: current), Recorder.audioSeconds(at: url) > 0 else {
                message = "The saved recording is missing or unreadable. The existing transcript and notes were kept."
                return
            }
            do { try store.backupBeforeRetranscription(current) }
            catch {
                message = "Could not back up this meeting. Nothing was replaced: \(error.localizedDescription)"
                return
            }
        }
        let id = UIApplication.shared.beginBackgroundTask(withName: "cicada-pipeline") { [weak self] in
            // Expiration: iOS grants no more background time and runs this on a
            // background thread. Hop to the main actor and set the cancel flag;
            // runPipeline checks it at its step boundaries, saves the partial
            // progress reached so far and returns. process() below then ends
            // the background task; if the pipeline hangs in an await, iOS
            // suspends the app as usual (state was already persisted).
            Task { @MainActor in self?.pipelineCancelled = true }
        }
        backgroundTaskId = id
        taskID = id
        pipelineCancelled = false
        if useRecordingSegments {
            await applySpeakers(current)
            guard let latest = store.sessions.first(where: { $0.id == s.id }) else { return }
            current = latest
        }
        if pipelineCancelled {
            current.processingState = .failed
            current.lastError = "Processing was interrupted."
            store.save(current)
            return
        }
        await runPipeline(current, retranscribe: retranscribe)
    }

    /// The pipeline, restructured after the "real transcript but no notes" bug:
    /// each step collects its own error and the next step still runs — a
    /// transcribe or translate failure must never skip notes. Errors persist on
    /// the session (Notes tab shows them; the alert is a bonus, not the record).
    ///
    /// Background survival: process() wraps this in a UIApplication background
    /// task, so it keeps running briefly after the user leaves the app. Its
    /// expiration handler sets pipelineCancelled; this function checks that flag
    /// at each step boundary and saves the partial progress reached so far.
    private func runPipeline(_ s: Session, retranscribe: Bool) async {
        let hasText = s.lines.contains { !$0.src.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
        guard AISettings.isConfigured || (!retranscribe && hasText && AIClient.Side.notes.isConfigured) else {
            let reason = "Set the API base URL and key in Settings › AI Service."
            message = reason
            if var current = store.sessions.first(where: { $0.id == s.id }), current.isImported {
                current.processingState = .failed
                current.lastError = reason
                store.save(current)
            }
            return
        }
        var s = s
        var errors: [String] = []
        s.processingState = .running
        s.lastError = nil
        store.save(s)
        updateImportProgress(&s, to: 0.4)

        // Step 1 — transcribe. Skipped when a server transcription has already
        // succeeded once: a refresh after a notes-only failure must not re-upload
        // every audio chunk. The liveText fallback line counts as transcript text
        // for the notes step, but does not set `transcribed` — a server
        // transcription can still improve on it.
        var replacedTranscript = false
        if AISettings.isConfigured, !s.transcribed || retranscribe, let url = store.audioURL(for: s) {
            do {
                let text: String
                var failedChunks = 0
                if ai.baseURLHost.contains("ant-ling.com") {
                    let r = try await ai.transcribeChunked(audioURL: url)
                    text = r.text; failedChunks = r.failedChunks
                } else {
                    text = try await ai.transcribe(audioURL: url)
                }
                if pipelineCancelled { throw CancellationError() }
                if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    // A partial replacement must not destroy a complete saved transcript.
                    if retranscribe && failedChunks > 0 {
                        throw NSError(domain: "Cicada", code: 1, userInfo: [NSLocalizedDescriptionKey:
                            "\(failedChunks) audio chunk(s) failed. The existing transcript and notes were kept."])
                    }
                    var segments = retranscribe ? [] : s.lines
                    // Recover voices from the FILE after a relaunch/offline failure. In-memory
                    // gate/caption state is not available then. A fresh re-transcription also
                    // gets a fresh voice namespace; old names must not label new clusters.
                    if retranscribe || !segments.contains(where: { $0.voice != nil }) {
                        do {
                            let recovered = try await speakers.lines(in: url)
                            if !recovered.isEmpty { segments = recovered }
                        } catch {
                            if retranscribe { throw error }
                            errors.append("Speaker separation failed: \(error.localizedDescription)")
                        }
                    }
                    if pipelineCancelled { throw CancellationError() }
                    // Empty text / a nil voice is a valid short unknown reply, not a reason
                    // to collapse every other segment into one line (upstream ab787d4).
                    let segmented = segments.count > 1 || segments.contains(where: { $0.voice != nil })
                    let seconds = Recorder.audioSeconds(at: url)
                    var replacement = segmented
                        ? RecordRefiner.refine(text, onto: segments, duration: seconds)
                        : [TranscriptLine(t: 0, speaker: 1, src: text)]
                    let numbers = SpeakerLabels.voiceNumbers(replacement)
                    for i in replacement.indices {
                        if let voice = replacement[i].voice, let number = numbers[voice] {
                            replacement[i].speaker = number
                        }
                    }
                    s.lines = replacement
                    // Keep a live notes draft until this formal notes pass returns;
                    // a transcript refresh alone is not a successful notes update.
                    s.transcribed = true
                    s.failedChunkCount = failedChunks
                    replacedTranscript = true
                    // Persist the transcript immediately: if translate or notes
                    // fails below, the transcript is still on the session.
                    store.save(s)
                } else {
                    // No server text: the liveText fallback line (if any) is
                    // what notes will run on. Record how many chunks failed so
                    // the user can tell a fallback transcript from a real one.
                    s.failedChunkCount = failedChunks
                    if failedChunks > 0 {
                        // Name the host, like the thrown errors do (item 2): the
                        // user may have more than one service configured over time.
                        errors.append("Transcription on \(ai.baseURLHost) failed for all \(failedChunks) audio chunk(s).")
                    }
                    store.save(s)
                }
            } catch is CancellationError {
                // The interruption state is persisted below without treating cancellation as an ASR failure.
            } catch {
                if !pipelineCancelled { errors.append("Transcription failed: \(error.localizedDescription)") }
            }
        }
        if pipelineCancelled {
            s.processingState = .failed
            s.lastError = errors.isEmpty ? "Processing was interrupted." : errors.joined(separator: " · ")
            store.save(s)
            return
        }
        updateImportProgress(&s, to: 0.65)
        if retranscribe && !replacedTranscript {
            s.processingState = .failed
            s.lastError = errors.isEmpty ? "No new transcript was returned. The existing transcript and notes were kept." : errors.joined(separator: " · ")
            store.save(s)
            message = s.lastError
            return
        }
        if pipelineCancelled {
            s.processingState = .failed
            s.lastError = errors.isEmpty ? "Processing was interrupted." : errors.joined(separator: " · ")
            store.save(s); return
        }

        // Step 2 — translate. When simultaneous interpretation is off, skip
        // translation entirely: no dst values are set (TranscriptRow hides dst
        // when nil) and notes are generated from the original-language lines.
        // A translate failure must not stop notes: the line keeps its src text.
        if AISettings.isConfigured && AISettings.simultaneousInterpretation && !s.lines.isEmpty && AISettings.targetLang != "auto" {
            for i in s.lines.indices where !s.lines[i].src.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                if pipelineCancelled { break }
                do {
                    let translated = try await ai.translate(s.lines[i].src, to: AISettings.targetLang)
                    if pipelineCancelled { break }
                    s.lines[i].dst = translated
                } catch is CancellationError {
                    break
                } catch {
                    if !pipelineCancelled { errors.append("Translation failed: \(error.localizedDescription)") }
                    break
                }
            }
            if !pipelineCancelled { store.save(s) }
        }
        if pipelineCancelled {
            s.processingState = .failed
            s.lastError = errors.isEmpty ? "Processing was interrupted." : errors.joined(separator: " · ")
            store.save(s); return
        }
        updateImportProgress(&s, to: 0.82)

        // Step 3 — notes. Runs whenever ANY transcript text exists (server
        // transcript or the liveText fallback line); a step-1/2 error above
        // does not skip it. Speaker labels match the transcript rows: an
        // unidentified voice is "Unknown", not a false "Speaker 1".
        let numbers = SpeakerLabels.voiceNumbers(s.lines)
        let transcript = s.lines.filter { !$0.src.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }.map { l -> String in
            let who: String
            if let name = l.speakerName, !name.isEmpty { who = name }
            else if let v = l.voice, let n = numbers[v] { who = "Speaker \(n)" }
            else if numbers.isEmpty { who = "Speaker \(l.speaker)" }
            else { who = "Unknown" }
            return "\(who): \(l.src)"
        }.joined(separator: "\n")
        if !transcript.isEmpty {
            do {
                let updatedNotes = try await ai.meetingNotes(transcript: transcript)
                if !pipelineCancelled, !updatedNotes.isEmpty { s.notes = updatedNotes }
            } catch is CancellationError {
                // Cancellation is persisted below as an interrupted partial run.
            } catch {
                if !pipelineCancelled { errors.append("Notes failed: \(error.localizedDescription)") }
            }
        } else if s.audioFile?.isEmpty == false {
            errors.append("No transcript text to write notes from.")
        }
        if pipelineCancelled {
            s.processingState = .failed
            s.lastError = errors.isEmpty ? "Processing was interrupted." : errors.joined(separator: " · ")
            store.save(s)
            return
        }

        s.lastError = errors.isEmpty ? nil : errors.joined(separator: " · ")
        s.processingState = (s.lastError == nil) ? .done : .failed
        store.save(s)
        if let e = s.lastError { message = e }
        // process() exports the latest saved state in defer, including early exits.
    }

    func nameSpeaker(in session: Session, voice: Int, name: String) {
        guard !busy, !recorder.isRecording,
              var current = store.sessions.first(where: { $0.id == session.id }),
              current.canNameVoice(voice) else { return }
        current.nameVoice(voice, name: name)
        store.save(current)
        exporter.write(current)
    }

    func pickMedia() {
        guard !busy, !startingRecording, !recorder.isRecording, importingSessionID == nil else { return }
        MediaPickerPresenter.shared.present { [weak self] url in self?.importMedia(url) }
    }

    private func importMedia(_ url: URL) {
        guard !busy, !startingRecording, !recorder.isRecording, importingSessionID == nil else { return }
        var session = Session()
        session.sourceFile = url.lastPathComponent
        session.importState = .queued
        let destination = store.importedSourceURL(for: session.id, filename: url.lastPathComponent)
        session.importSourceFile = destination.lastPathComponent
        store.save(session)
        selection = session.id
        resumeImport(session, copyFrom: url)
    }

    func resumeImport(_ snapshot: Session, copyFrom: URL? = nil) {
        guard !busy, !pipelineActive, !startingRecording, !recorder.isRecording,
              importingSessionID == nil,
              var session = store.sessions.first(where: { $0.id == snapshot.id }), session.isImported else { return }
        let cancellation = MediaImportCancellation()
        importCancellation = cancellation
        importingSessionID = session.id
        busy = true
        session.lastError = nil
        store.save(session)
        let copyScopeStarted = copyFrom?.startAccessingSecurityScopedResource() ?? false
        importJobTask = Task { [weak self] in
            defer {
                if copyScopeStarted, let copyFrom { copyFrom.stopAccessingSecurityScopedResource() }
            }
            guard let self else { return }
            let id = session.id
            let grant = UIApplication.shared.beginBackgroundTask(withName: "cicada-import") { [weak self] in
                cancellation.cancel()
                Task { @MainActor in self?.importJobTask?.cancel() }
            }
            defer {
                if grant != .invalid { UIApplication.shared.endBackgroundTask(grant) }
                self.importingSessionID = nil
                self.importCancellation = nil
                self.importJobTask = nil
                self.busy = false
            }
            do {
                var current = self.store.sessions.first(where: { $0.id == id }) ?? session
                if let copyFrom {
                    guard let sourceDestination = self.store.importURL(for: current) else {
                        throw MediaImportError(message: "Could not create a local source-file path.")
                    }
                    try await Task.detached(priority: .userInitiated) {
                        try MediaImporter.copySource(from: copyFrom, to: sourceDestination,
                            sourceAlreadyScoped: copyScopeStarted,
                            isCancelled: { cancellation.isCancelled },
                            progress: { fraction in
                                Task { @MainActor [weak self] in
                                    guard let self, !cancellation.isCancelled,
                                          var saved = self.store.sessions.first(where: { $0.id == id }),
                                          saved.importState == .queued else { return }
                                    saved.importProgress = fraction * 0.05
                                    self.store.save(saved)
                                }
                            })
                    }.value
                    if cancellation.isCancelled || Task.isCancelled { throw CancellationError() }
                    current = self.store.sessions.first(where: { $0.id == id }) ?? current
                    current.importProgress = 0.05
                    self.store.save(current)
                    if cancellation.isCancelled || Task.isCancelled { throw CancellationError() }
                }
                let existingAudio = self.store.audioURL(for: current)
                let hasExistingAudio = existingAudio.map { Recorder.audioSeconds(at: $0) > 0 } ?? false
                if !hasExistingAudio {
                    guard let source = self.store.importURL(for: current),
                          FileManager.default.fileExists(atPath: source.path) else {
                        throw MediaImportError(message: "The saved source file is missing. Choose the file again.")
                    }
                    current.importState = .extracting
                    current.importProgress = 0
                    self.store.save(current)
                    let destination = self.store.recordingURL(for: id).deletingPathExtension().appendingPathExtension("wav")
                    let seconds = try await Task.detached(priority: .userInitiated) {
                        try MediaImporter.decodeAudio(from: source, to: destination,
                            isCancelled: { cancellation.isCancelled }, progress: { fraction in
                                Task { @MainActor [weak self] in
                                    guard let self, !cancellation.isCancelled,
                                          var saved = self.store.sessions.first(where: { $0.id == id }),
                                          saved.importState == .extracting else { return }
                                    // Progress is stage based, not an invented upload percentage.
                                    let next = fraction * 0.35
                                    if next - saved.importProgress >= 0.02 {
                                        saved.importProgress = next
                                        self.store.save(saved)
                                    }
                                }
                            })
                    }.value
                    if cancellation.isCancelled { throw CancellationError() }
                    current = self.store.sessions.first(where: { $0.id == id }) ?? current
                    current.audioFile = destination.lastPathComponent
                    current.endedAt = current.startedAt.addingTimeInterval(seconds)
                    current.importState = .ready
                    current.importProgress = 0.35
                    self.store.save(current)
                }
                if cancellation.isCancelled { throw CancellationError() }
                current.importState = .processing
                self.store.save(current)
                await self.process(current)
                guard var result = self.store.sessions.first(where: { $0.id == id }) else { return }
                if cancellation.isCancelled || Task.isCancelled { throw CancellationError() }
                result.importState = result.processingState == .done ? .done : .failed
                if result.importState == .done { result.importProgress = 1 }
                self.store.save(result)
            } catch {
                guard var result = self.store.sessions.first(where: { $0.id == id }) else { return }
                let paused = cancellation.isCancelled || Task.isCancelled || error is CancellationError
                result.importState = paused ? .paused : .failed
                result.processingState = .failed
                result.lastError = paused ? "Import paused. Resume to restart the unfinished stage; the source and completed text are saved." : error.localizedDescription
                self.store.save(result)
                if !paused { self.message = result.lastError }
            }
        }
    }

    func pauseImport() {
        guard importingSessionID != nil else { return }
        importCancellation?.cancel()
        importJobTask?.cancel()
        pipelineCancelled = true
    }

    func editLine(sessionID: String, lineID: String, source: String, translation: String) {
        guard !busy, !recorder.isRecording,
              var current = store.sessions.first(where: { $0.id == sessionID }),
              let index = current.lines.firstIndex(where: { $0.id == lineID }) else { return }
        current.lines[index].src = source.trimmingCharacters(in: .whitespacesAndNewlines)
        let dst = translation.trimmingCharacters(in: .whitespacesAndNewlines)
        current.lines[index].dst = dst.isEmpty ? nil : dst
        store.save(current)
        exporter.write(current)
    }

    func exportMarkdown(_ s: Session) -> String { TranscriptExport.markdown(s) }
}

// MARK: - Views

/// Foreground meeting page. Dismissing this view does not stop the Recorder;
/// the records list keeps a permanent route back while the session is active.
struct LiveMeetingView: View {
    @EnvironmentObject var model: AppModel
    @State private var page = 1

    private var session: Session? {
        guard let id = model.recordingId else { return nil }
        return model.store.sessions.first { $0.id == id }
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                HStack(spacing: 8) {
                    Circle().fill(model.recorder.paused ? Ink.secondary : Ink.accent)
                        .frame(width: 9, height: 9)
                    Text(model.recorder.paused ? "Paused" : "Listening")
                        .font(.subheadline.weight(.semibold))
                    Text(TranscriptExport.clock(model.recorder.elapsed))
                        .font(.subheadline.monospacedDigit()).foregroundStyle(Ink.secondary)
                    Spacer()
                    Text("Auto → \(AISettings.targetLang.uppercased())")
                        .font(.caption).foregroundStyle(Ink.secondary)
                }
                .padding(.horizontal, 20).padding(.vertical, 12)
                Picker("Live meeting", selection: $page) {
                    Text("Live notes").tag(0)
                    Text("Subtitles").tag(1)
                }
                .pickerStyle(.segmented)
                .padding(.horizontal, 20).padding(.bottom, 8)

                if model.recorder.paused {
                    Text("The microphone is temporarily unavailable. Recording will resume when it can.")
                        .font(.footnote).foregroundStyle(Ink.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 20).padding(.bottom, 8)
                }
                if let problem = model.liveMeeting.problem {
                    Text(problem).font(.footnote).foregroundStyle(Ink.accentText)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 20).padding(.bottom, 8)
                }

                if page == 0 {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 12) {
                            if let notes = model.liveMeeting.notes, !notes.isEmpty {
                                NotesView(notes: notes, state: .done)
                            } else if model.liveMeeting.notesUpdating {
                                HStack(spacing: 8) {
                                    ProgressView()
                                    Text("Preparing the first notes draft…").foregroundStyle(Ink.secondary)
                                }.padding(.top, 24)
                            } else {
                                Text("Live notes appear after a few spoken lines.")
                                    .foregroundStyle(Ink.secondary).padding(.top, 24)
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 20).padding(.top, 8)
                    }
                } else {
                    ScrollViewReader { proxy in
                        ScrollView {
                            LazyVStack(alignment: .leading, spacing: 12) {
                                ForEach(Array(model.liveMeeting.lines.suffix(12))) { line in
                                    VStack(alignment: .leading, spacing: 5) {
                                        HStack {
                                            Text("Speaker \(line.speaker)").font(.caption.weight(.semibold))
                                            Spacer()
                                            Text(TranscriptExport.clock(line.t)).font(.caption.monospacedDigit())
                                                .foregroundStyle(Ink.tertiary)
                                        }
                                        if let dst = line.dst, !dst.isEmpty {
                                            Text(dst).font(.body.weight(.medium))
                                            Text(line.src).font(.footnote).foregroundStyle(Ink.secondary)
                                        } else {
                                            Text(line.src).font(.body)
                                        }
                                    }
                                    .padding(12)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(Ink.accentWash.opacity(0.45), in: RoundedRectangle(cornerRadius: 12))
                                    .id(line.id)
                                }
                                if model.liveMeeting.lines.isEmpty {
                                    if !model.recorder.liveText.isEmpty {
                                        VStack(alignment: .leading, spacing: 5) {
                                            Text("On-device caption").font(.caption.weight(.semibold))
                                                .foregroundStyle(Ink.secondary)
                                            Text(model.recorder.liveText).font(.body)
                                        }
                                        .padding(12).frame(maxWidth: .infinity, alignment: .leading)
                                        .background(Ink.line.opacity(0.35), in: RoundedRectangle(cornerRadius: 12))
                                    } else {
                                        Text("Waiting for speech…")
                                            .foregroundStyle(Ink.secondary).padding(.top, 32)
                                            .frame(maxWidth: .infinity)
                                    }
                                }
                            }
                            .padding(.horizontal, 16).padding(.top, 8).padding(.bottom, 16)
                        }
                        .onChange(of: model.liveMeeting.lines.last?.id) { id in
                            if let id { withAnimation(.easeOut(duration: 0.2)) { proxy.scrollTo(id, anchor: .bottom) } }
                        }
                    }
                }
                Spacer(minLength: 0)
                Button("Stop meeting", role: .destructive) { model.stopRecording() }
                    .buttonStyle(.borderedProminent).tint(Ink.accent)
                    .padding(.top, 8).padding(.bottom, 16)
            }
            .navigationTitle("In meeting")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Done") { model.showingLiveMeeting = false }
                }
            }
            .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
            .onDisappear { UIApplication.shared.isIdleTimerDisabled = false }
        }
    }
}

struct RootView: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        ZStack {
            if model.tab == .records { RecordsView() } else { SettingsView() }
        }
        .overlay(alignment: .bottom) {
            if model.selection == nil {
                HStack(spacing: 4) {
                    TabBtn(title: "Records", icon: "list.bullet", on: model.tab == .records) { model.tab = .records }
                    TabBtn(title: "Settings", icon: "gearshape", on: model.tab == .settings) { model.tab = .settings }
                }
                .padding(5).background(.thinMaterial, in: Capsule())
                .padding(.bottom, 8)
            }
        }
        .tint(Ink.accentText).preferredColorScheme(.light)
        .alert("Cicada", isPresented: Binding(get: { model.message != nil }, set: { if !$0 { model.message = nil } })) {
            Button("OK") { model.message = nil }
        } message: { Text(model.message ?? "") }
        .fullScreenCover(isPresented: $model.showingLiveMeeting) {
            LiveMeetingView()
                .environmentObject(model)
        }
        .fullScreenCover(isPresented: $model.showingFaceToFace) {
            FaceToFaceView().environmentObject(model)
        }
    }
}
struct TabBtn: View {
    let title: String; let icon: String; let on: Bool; let tap: () -> Void
    var body: some View {
        Button(action: tap) {
            VStack(spacing: 2) { Image(systemName: icon).font(.system(size: 19, weight: .medium)); Text(title).font(.system(size: 11, weight: on ? .semibold : .regular)) }
            .foregroundStyle(on ? Ink.accentText : Ink.secondary)
            .frame(width: 84, height: 52)
            .background(on ? Ink.accentWash : .clear, in: Capsule())
        }.buttonStyle(.plain)
    }
}

struct RecordsView: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        NavigationStack {
            if let s = model.selected { RecordDetail(session: s) } else { RecordsList() }
        }
    }
}

struct RecordsList: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        List {
            Section {
                HStack {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Listen to the room").font(.headline)
                        if model.recorder.isRecording {
                            HStack(spacing: 6) {
                                Circle().fill(Ink.accent).frame(width: 8, height: 8)
                                    .opacity(0.4 + 0.6 * Double(model.recorder.level))
                                Text(TranscriptExport.clock(model.recorder.elapsed)).font(.footnote.monospacedDigit())
                            }
                            if model.recorder.paused {
                                Text("Paused — the audio was taken by a call or another app; recording resumes when it ends.")
                                    .font(.footnote).foregroundStyle(Ink.secondary)
                            }
                            if !model.recorder.liveText.isEmpty {
                                Text(model.recorder.liveText).font(.footnote)
                                    .foregroundStyle(Ink.secondary).lineLimit(2)
                            }
                            if let re = model.recorder.recognitionError {
                                Text(re).font(.footnote)
                                    .foregroundStyle(Ink.secondary).lineLimit(4)
                            }
                        } else {
                            Text("Record, transcribe, translate and write notes")
                                .font(.footnote).foregroundStyle(Ink.secondary)
                        }
                    }
                    Spacer()
                    if model.recorder.isRecording {
                        VStack(alignment: .trailing, spacing: 6) {
                            Button("Open meeting") { model.showingLiveMeeting = true }
                                .buttonStyle(.bordered)
                            Button("Stop", role: .destructive) { model.stopRecording() }
                        }
                    } else {
                        Button { Task { await model.startRecording() } } label: {
                            Image(systemName: "mic.fill").font(.title2)
                                .foregroundStyle(.white).frame(width: 52, height: 52)
                                .background(Ink.accent, in: Circle())
                        }.disabled(model.busy)
                    }
                }
                .padding(.vertical, 4)
            }
            Section {
                Button { model.showingFaceToFace = true } label: {
                    Label("Face-to-face · hold to speak", systemImage: "person.2.wave.2")
                }
                .disabled(model.busy || model.recorder.isRecording || model.importingSessionID != nil)
            }
            Section("Import audio or video") {
                if let id = model.importingSessionID,
                   let active = model.store.sessions.first(where: { $0.id == id }) {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(active.sourceFile ?? "Media file").font(.subheadline.weight(.medium))
                        ProgressView(value: active.importProgress)
                        Text(active.importState == .extracting ? "Extracting audio…" : "Transcribing and preparing notes…")
                            .font(.footnote).foregroundStyle(Ink.secondary)
                        Button("Pause import", role: .destructive) { model.pauseImport() }
                    }
                } else {
                    Button { model.pickMedia() } label: {
                        Label("Choose audio or video…", systemImage: "square.and.arrow.down")
                    }
                    .disabled(model.busy || model.recorder.isRecording)
                    Text("Choose a file from Files. Keep Cicada open for reliable import; iOS may suspend background work.")
                        .font(.footnote).foregroundStyle(Ink.secondary)
                }
            }
            Section("Meetings") {
                ForEach(model.store.sessions) { s in
                    VStack(alignment: .leading, spacing: 6) {
                        Button { model.selection = s.id } label: {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(TranscriptExport.displayTitle(s)).font(.body.weight(.medium)).foregroundStyle(Ink.ink)
                                Text("\(Int(s.duration/60)) min · \(s.lines.count) lines\(s.isImported ? " · Imported" : "")")
                                    .font(.footnote).foregroundStyle(Ink.secondary)
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        if s.isImported && (s.importState == .paused || s.importState == .failed || s.importState == .ready) {
                            Button(s.importState == .failed ? "Retry import" : "Resume import") {
                                model.resumeImport(s)
                            }
                            .font(.footnote)
                            .disabled(model.busy || model.recorder.isRecording)
                        } else if s.isImported && s.importState == .processing {
                            Text("Import processing…").font(.footnote).foregroundStyle(Ink.secondary)
                        }
                    }
                }
                .onDelete {
                    guard !model.busy, !model.recorder.isRecording else { return }
                    if let i = $0.first, i < model.store.sessions.count {
                        model.store.delete(model.store.sessions[i]); model.selection = nil
                    }
                }
            }
        }
        .listStyle(.insetGrouped)
        .navigationTitle("Records")
    }
}

struct RecordDetail: View {
    @EnvironmentObject var model: AppModel
    let session: Session
    @State private var page = 0
    @StateObject private var player = RecordPlayer()
    /// True while the list should follow the playing line (ScrollViewReader);
    /// a manual scroll stops the following until playback seeks again, so the
    /// user can read elsewhere without the list fighting them (the Mac's
    /// `following` rule).
    @State private var following = true
    @State private var namingVoice: Int?
    @State private var speakerName = ""
    @State private var showingName = false
    @State private var confirmingRetranscription = false
    @State private var editingLineID: String?
    @State private var editedSource = ""
    @State private var editedTranslation = ""

    private func beginEditing(_ line: TranscriptLine) {
        editingLineID = line.id
        editedSource = line.src
        editedTranslation = line.dst ?? ""
    }

    private func nameAction(for line: TranscriptLine) -> (() -> Void)? {
        guard !model.busy, !model.recorder.isRecording,
              let voice = line.voice, session.canNameVoice(voice) else { return nil }
        return {
            namingVoice = voice
            speakerName = line.speakerName ?? ""
            showingName = true
        }
    }

    var body: some View {
        // The line at the playback position (the official app's lineID, +0.2 s
        // tolerance), recomputed on every player tick; the rows and the
        // follow-scroll both read it.
        let playing = player.lineID(in: session)
        return ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 0) {
                    // Safety net: if this page is showing while a recording runs (e.g. the
                    // user opened an old session mid-recording), Stop is reachable here.
                    if model.recorder.isRecording {
                        RecordingBanner()
                    }
                    Text(TranscriptExport.displayTitle(session)).font(.title.bold())
                    Text("\(Int(session.duration/60)) min · \(session.lines.count) lines")
                        .font(.footnote).foregroundStyle(Ink.secondary).padding(.top, 4)
                    if session.isImported {
                        VStack(alignment: .leading, spacing: 8) {
                            Label(session.sourceFile ?? "Imported media", systemImage: "doc")
                                .font(.subheadline).lineLimit(2)
                            Text("Import: \(session.importState.rawValue)").font(.caption)
                            ProgressView(value: min(1, max(0, session.importProgress)))
                            if model.importingSessionID == session.id {
                                Button("Pause import") { model.pauseImport() }
                            } else if session.importState != .done {
                                Button("Resume / retry import") { model.resumeImport(session) }
                                    .disabled(model.busy || model.recorder.isRecording || model.importingSessionID != nil)
                            }
                            if let error = session.lastError {
                                Text(error).font(.footnote).foregroundStyle(Ink.secondary)
                            }
                            Text("Keep Cicada open. Progress marks completed stages; iOS may pause background work. Resume restarts an unfinished stage.")
                                .font(.caption).foregroundStyle(Ink.secondary)
                        }.padding(.vertical, 12)
                    }
                    Picker("Page", selection: $page) {
                        Text("Transcript").tag(0); Text("Notes").tag(1)
                    }.pickerStyle(.segmented).padding(.top, 16)
                    if page == 0 {
                        // Speaker labels need the whole session's voice numbers (a voice
                        // gets a number only once two voices are on show), so compute
                        // them here once instead of per row.
                        let numbers = SpeakerLabels.voiceNumbers(session.lines)
                        ForEach(session.lines) { l in
                            VStack(alignment: .leading, spacing: 2) {
                                TranscriptRow(line: l, numbers: numbers, playing: l.id == playing,
                                              onName: nameAction(for: l))
                                    .contentShape(Rectangle())
                                    .onTapGesture { following = true; player.seek(l.t) }
                                if session.isImported {
                                    Button("Edit line") { beginEditing(l) }
                                        .font(.caption).padding(.leading, 42)
                                }
                            }
                            .id(l.id)
                        }
                        if session.lines.isEmpty {
                            Text("No transcript yet.").foregroundStyle(Ink.secondary).padding(.top, 24)
                            // The offline case, said plainly: the recording exists and
                            // is safe, the cloud transcription needs the network, and
                            // it will be retried automatically when signal returns.
                            if !session.transcribed, session.audioFile != nil, !model.networkAvailable {
                                Text("You appear to be offline — the audio is saved, and the transcript will be generated automatically once the network is back.")
                                    .font(.footnote).foregroundStyle(Ink.secondary).padding(.top, 6)
                            }
                            // Why the live captions (and thus the fallback line)
                            // never came — e.g. the on-device model for this
                            // language was never downloaded. Persisted with the
                            // session at stop, so it survives relaunch.
                            if let note = session.liveCaptionNote {
                                Text(note).font(.footnote).foregroundStyle(Ink.secondary).padding(.top, 6)
                            }
                        }
                    } else {
                        NotesView(notes: session.notes, state: session.processingState,
                                  lastError: session.lastError, failedChunks: session.failedChunkCount)
                    }
                }
                .padding(.horizontal, 20).padding(.top, 8)
                // The player bar overlays the bottom of the scroll; extra bottom
                // padding keeps the last line clear of it.
                .padding(.bottom, player.available ? 96 : 24)
            }
            // A manual scroll stops the following (the Mac's `following` rule):
            // the list must not fight the reader who went looking elsewhere.
            // simultaneousGesture + a minimum distance keeps the ScrollView's
            // own pan in charge (iOS 16.4-safe; no iOS 17 scroll APIs).
            .simultaneousGesture(
                DragGesture(minimumDistance: 25).onChanged { _ in
                    // Only while playing: at rest there is nothing to follow and
                    // any pan (e.g. system gestures) must not touch the flag.
                    if player.playing { following = false }
                }
            )
            // The list follows the playing line while `following` holds; any
            // seek (line tap, play) re-arms it.
            .onChange(of: playing) { id in
                guard following, let id, player.playing || player.now > 0 else { return }
                proxy.scrollTo(id, anchor: .center)
            }
            .safeAreaInset(edge: .bottom) {
                if player.available {
                    PlayerBar(player: player).padding(.bottom, 8)
                }
            }
            .navigationTitle("").navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItemGroup(placement: .topBarTrailing) {
                    Button { model.selection = nil } label: {
                        Image(systemName: "chevron.left")
                    }
                    Menu {
                        Button("Retry processing / regenerate notes") {
                            Task { await model.process(session) }
                        }
                        Button("Transcribe Again…", role: .destructive) {
                            player.stop()
                            confirmingRetranscription = true
                        }.disabled(session.audioFile == nil)
                    } label: {
                        model.busy ? AnyView(ProgressView()) : AnyView(Image(systemName: "arrow.triangle.2.circlepath"))
                    }.disabled(model.busy || model.recorder.isRecording)
                    ShareLink(item: model.exportMarkdown(session), subject: Text(TranscriptExport.displayTitle(session))) {
                        Image(systemName: "square.and.arrow.up")
                    }
                }
            }
            .sheet(isPresented: Binding(
                get: { editingLineID != nil }, set: { if !$0 { editingLineID = nil } })) {
                NavigationStack {
                    Form {
                        TextField("Original text", text: $editedSource, axis: .vertical)
                            .lineLimit(3...8)
                        TextField("Translation (optional)", text: $editedTranslation, axis: .vertical)
                            .lineLimit(3...8)
                    }
                    .navigationTitle("Edit transcript")
                    .toolbar {
                        ToolbarItem(placement: .cancellationAction) {
                            Button("Cancel") { editingLineID = nil }
                        }
                        ToolbarItem(placement: .confirmationAction) {
                            Button("Save") {
                                if let id = editingLineID {
                                    model.editLine(sessionID: session.id, lineID: id,
                                                   source: editedSource, translation: editedTranslation)
                                }
                                editingLineID = nil
                            }
                        }
                    }
                }
            }
            .alert("Name this speaker", isPresented: $showingName) {
                TextField("Name (blank to clear)", text: $speakerName)
                Button("Save") {
                    if let voice = namingVoice { model.nameSpeaker(in: session, voice: voice, name: speakerName) }
                    namingVoice = nil
                }
                Button("Cancel", role: .cancel) { namingVoice = nil }
            } message: {
                Text("Applies to this voice in this meeting only. Voices are not remembered across meetings.")
            }
            .confirmationDialog("Transcribe this recording again?", isPresented: $confirmingRetranscription,
                                titleVisibility: .visible) {
                Button("Transcribe Again", role: .destructive) {
                    Task { await model.process(session, retranscribe: true) }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("A complete new transcript replaces this meeting's text, speaker names and notes. A local backup of the previous record is kept; if transcription fails, the existing content stays.")
            }
            .onAppear {
                // The player and the recorder share one AVAudioSession: the
                // player may switch it to .playback only when the recorder is
                // not mid-recording (see RecordPlayer.mayOwnSession). The
                // model lives for the app's lifetime, so the strong capture
                // is safe.
                player.mayOwnSession = { !model.recorder.isRecording }
                player.load(store: model.store, session: session)
            }
            // The detail view is reused when another record is picked
            // (selection changes, the view re-inits with the new session):
            // swap the player's audio to the new session's recording.
            .onChange(of: session.id) { _ in
                player.load(store: model.store, session: session)
            }
            .onChange(of: session.audioFile) { _ in
                player.load(store: model.store, session: session)
            }
            .onDisappear { player.stop() }   // leaving the screen stops playback
        }
    }
}

/// Live recording state: elapsed, level, on-device captions, Stop.
struct RecordingBanner: View {
    @EnvironmentObject var model: AppModel
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                HStack(spacing: 6) {
                    Circle().fill(Ink.accent).frame(width: 8, height: 8)
                        .opacity(0.4 + 0.6 * Double(model.recorder.level))
                    Text("Recording").font(.footnote.weight(.semibold)).foregroundStyle(Ink.accentText)
                    Text(TranscriptExport.clock(model.recorder.elapsed)).font(.footnote.monospacedDigit())
                }
                Spacer()
                Button("Open") { model.showingLiveMeeting = true }
                Button("Stop", role: .destructive) { model.stopRecording() }
            }
            if model.recorder.paused {
                Text("Paused — the audio was taken by a call or another app; recording resumes when it ends.")
                    .font(.footnote).foregroundStyle(Ink.secondary)
            }
            if !model.recorder.liveText.isEmpty {
                Text(model.recorder.liveText).font(.footnote)
                    .foregroundStyle(Ink.secondary).lineLimit(2)
            }
            if let re = model.recorder.recognitionError {
                Text(re).font(.footnote)
                    .foregroundStyle(Ink.secondary).lineLimit(4)
            }
        }
        .padding(12).background(Ink.accentWash, in: RoundedRectangle(cornerRadius: 12))
        .padding(.bottom, 12)
    }
}

struct TranscriptRow: View {
    let line: TranscriptLine
    /// Speaker numbers in force for this transcript (SpeakerLabels.voiceNumbers of
    /// the whole session, computed once by the caller — the row only renders).
    var numbers: [Int: Int] = [:]
    /// True when the playback position is on this line (the official app's
    /// row highlight: a light accent wash, the same 0.06 opacity as its).
    var playing: Bool = false
    var onName: (() -> Void)?

    @ViewBuilder
    private var speakerTag: some View {
        if let name = line.speakerName {
            Text(name).font(.caption.bold())
                .foregroundStyle(SpeakerInk.ink(line.speaker))
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(SpeakerInk.tint(line.speaker), in: Capsule())
        } else if let v = line.voice, let n = numbers[v] {
            Text("S\(n)").font(.caption.bold())
                .foregroundStyle(SpeakerInk.ink(n))
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(SpeakerInk.tint(n), in: Capsule())
        } else if numbers.isEmpty {
            Text("S\(line.speaker)").font(.caption.bold())
                .foregroundStyle(SpeakerInk.ink(line.speaker))
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(SpeakerInk.tint(line.speaker), in: Capsule())
        } else {
            Text("未识别").font(.caption.weight(.semibold))
                .foregroundStyle(Ink.tertiary)
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(Ink.line.opacity(0.5), in: Capsule())
        }
    }

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            // Only a voice that was actually identified claims a speaker number.
            // A line whose voice never reached the shown count (too short, unlike
            // every known voice, or below voiceNumbers' two-voice threshold) used
            // to render as S1 — falsely claiming the main speaker. It now shows
            // upstream's "未识别" label instead, and only once SOME voice is on
            // show: a genuine solo recording keeps its S1 on every line. The
            // branch order matches TranscriptExport.markdown, so the export and
            // the screen never disagree about who spoke.
            if let onName {
                Button(action: onName) { speakerTag }
                    .buttonStyle(.plain)
                    .accessibilityLabel("Name this speaker")
            } else {
                speakerTag
            }
            VStack(alignment: .leading, spacing: 2) {
                Text(line.src).font(.body)
                if let d = line.dst, d != line.src { Text(d).font(.body).foregroundStyle(Ink.secondary) }
                Text(TranscriptExport.clock(line.t)).font(.caption).foregroundStyle(Ink.tertiary)
            }
        }
        .padding(.vertical, 6)
        // The playing line's highlight, the official app's 0.06 accent wash
        // over the whole row (not just the text column).
        .background(playing ? Ink.accent.opacity(0.06) : .clear, in: Rectangle())
    }
}

/// Notes tab. Renders by the session's persisted processing state, so the user
/// can tell not-run / running / failed apart without waiting on a one-shot alert.
struct NotesView: View {
    let notes: MeetingNotes?
    var state: ProcessingState = .idle
    var lastError: String?
    var failedChunks: Int = 0

    /// Failure text for the UI: the persisted reason, or a generic line when a
    /// step failed without a captured message.
    private var reason: String {
        if let e = lastError, !e.isEmpty { return e }
        return "an unexpected error occurred"
    }

    var body: some View {
        if let n = notes, !n.isEmpty {
            VStack(alignment: .leading, spacing: 14) {
                if state == .failed {
                    Text("Last attempt failed: \(reason). The saved content below was kept.")
                        .font(.footnote).foregroundStyle(Ink.accentText)
                }
                if !n.summary.isEmpty { Text(n.summary).font(.body) }
                ForEach(n.topics, id: \.title) { t in
                    VStack(alignment: .leading, spacing: 4) {
                        Text(t.title).font(.headline)
                        ForEach(t.points, id: \.self) { Text("• \($0)").font(.body) }
                    }
                }
                if !n.conclusions.isEmpty {
                    Text("Conclusions").font(.headline)
                    ForEach(n.conclusions, id: \.self) { Text("• \($0)") }
                }
                if !n.actionItems.isEmpty {
                    Text("Action items").font(.headline)
                    ForEach(n.actionItems, id: \.text) { a in
                        Text("☐ \(a.text)\(a.owner.map { " (\($0))" } ?? "")")
                    }
                }
            }.padding(.top, 16)
        } else if state == .running {
            HStack(spacing: 8) {
                ProgressView()
                Text("Generating…").foregroundStyle(Ink.secondary)
            }.padding(.top, 24)
        } else if state == .failed {
            VStack(alignment: .leading, spacing: 6) {
                Text("Generation failed: \(reason)").foregroundStyle(Ink.accentText)
                if failedChunks > 0 {
                    Text("\(failedChunks) audio chunk(s) could not be transcribed and were skipped.").font(.footnote).foregroundStyle(Ink.secondary)
                }
                Text("Tap ↻ to retry.").foregroundStyle(Ink.secondary)
            }.padding(.top, 24)
        } else if state == .done {
            // The pipeline ran and nothing failed, but the model's notes came
            // back empty (e.g. only blank fields) — not the same as never-run.
            Text("Notes came back empty. Tap ↻ to retry.").foregroundStyle(Ink.secondary).padding(.top, 24)
        } else {
            Text("No notes yet. Tap ▶ to transcribe and write notes.").foregroundStyle(Ink.secondary).padding(.top, 24)
        }
    }
}

// MARK: - Settings

/// Folder picking through UIKit's document picker, presented from the
/// topmost view controller. Two presentation layers failed on the target
/// setup (iOS 16.4, TrollStore install): SwiftUI's .fileImporter never
/// presented at all, and the picker embedded as .sheet content opened but
/// its Open button did nothing (an embedded picker's bar buttons rely on
/// its own presentation). Presenting it the classic UIKit way is the route
/// that works.
final class FolderPickerPresenter: NSObject, UIDocumentPickerDelegate {
    static let shared = FolderPickerPresenter()
    private var onPick: ((URL) -> Void)?

    func present(onPick: @escaping (URL) -> Void) {
        self.onPick = onPick
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.folder], asCopy: false)
        picker.delegate = self
        picker.allowsMultipleSelection = false
        topViewController()?.present(picker, animated: true)
    }

    private func topViewController() -> UIViewController? {
        let scene = UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .first { $0.activationState == .foregroundActive }
            ?? UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first
        var top = scene?.windows.first { $0.isKeyWindow }?.rootViewController
        while let presented = top?.presentedViewController { top = presented }
        return top
    }

    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        if let url = urls.first { onPick?(url) }
        onPick = nil
    }

    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) { onPick = nil }
}

struct SettingsView: View {
    @EnvironmentObject var model: AppModel
    @State private var baseURL = AISettings.baseURL
    @State private var keyDraft = ""
    @State private var chatModel = AISettings.chatModel
    @State private var transcribeModel = AISettings.transcribeModel
    @State private var targetLang = AISettings.targetLang
    @State private var simultaneousInterpretation = AISettings.simultaneousInterpretation
    // Optional notes service (item 2): empty = use the main service's fields.
    @State private var notesBaseURL = AISettings.notesBaseURL
    @State private var notesKeyDraft = ""
    @State private var notesChatModel = AISettings.notesChatModel
    // Auto-save folder (item 3).
    var body: some View {
        Form {
            Section("AI Service (OpenAI-compatible)") {
                TextField("Base URL", text: $baseURL, prompt: Text("https://api.example.com/v1"))
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                    .onChange(of: baseURL, perform: { _ in AISettings.baseURL = baseURL })
                SecureField("API key", text: $keyDraft, prompt: Text("sk-…"))
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                    .onSubmit(saveKey)
                // First-time setup must have a way to save: the button used to show
                // only when a key already existed, so the very first paste was lost
                // unless the keyboard's return key was pressed. Shown whenever the
                // field has content, saved or not.
                if !keyDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Button("Save key") { saveKey() }
                }
                HStack {
                    // The check's own result (AIClient.status), not mere key
                    // existence: a refused key and a working one used to look
                    // identical ("Connected" meant "a key is stored").
                    checkStatusView
                    Spacer()
                    Button("Check") { Task { _ = await model.ai.checkKey() } }
                        .disabled(checkRunning)
                }
                if case .failed(let reason) = model.ai.status {
                    Text(reason).font(.footnote).foregroundStyle(Ink.accentText)
                }
                TextField("Chat model", text: $chatModel).onChange(of: chatModel, perform: { _ in AISettings.chatModel = chatModel })
                TextField("Transcription model", text: $transcribeModel).onChange(of: transcribeModel, perform: { _ in AISettings.transcribeModel = transcribeModel })
                Picker("Translate to", selection: $targetLang) {
                    Text("English").tag("en"); Text("Chinese").tag("zh"); Text("Japanese").tag("ja"); Text("Korean").tag("ko")
                }.onChange(of: targetLang, perform: { _ in AISettings.targetLang = targetLang })
                Toggle("同声传译", isOn: $simultaneousInterpretation)
                    .onChange(of: simultaneousInterpretation, perform: { _ in AISettings.simultaneousInterpretation = simultaneousInterpretation })
                if AISettings.hasKey { Button("Remove key", role: .destructive) { Keychain.remove(for: AISettings.keyKey); keyDraft = "" } }
            }
            Section("Notes service (optional)") {
                // Upstream v1.9.0 parity: notes can run on their own service.
                // Empty fields mean "same as the main service" — that is the
                // documented behaviour, not a missing setting.
                TextField("Base URL", text: $notesBaseURL, prompt: Text("Empty = the main service"))
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                    .onChange(of: notesBaseURL) { _ in
                        AISettings.notesBaseURL = notesBaseURL
                        // The check row appears/disappears with the address;
                        // a half-configured state must not keep a stale result.
                        if !AISettings.notesApart { model.ai.notesStatus = .idle }
                    }
                SecureField("API key", text: $notesKeyDraft, prompt: Text("Empty = the main key"))
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                    .onSubmit(saveNotesKey)
                if !notesKeyDraft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    Button("Save key") { saveNotesKey() }
                }
                // The check action only exists when the notes service is
                // configured (a notes address is set), mirroring the main one.
                if AISettings.notesApart {
                    HStack {
                        notesCheckStatusView
                        Spacer()
                        Button("Check") { Task { _ = await model.ai.checkNotes() } }
                            .disabled(notesCheckRunning)
                    }
                    if case .failed(let reason) = model.ai.notesStatus {
                        Text(reason).font(.footnote).foregroundStyle(Ink.accentText)
                    }
                }
                TextField("Notes model", text: $notesChatModel, prompt: Text("Empty = the chat model"))
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                    .onChange(of: notesChatModel) { _ in AISettings.notesChatModel = notesChatModel }
                if AISettings.hasNotesKey {
                    Button("Remove notes key", role: .destructive) {
                        Keychain.remove(for: AISettings.notesKeyKey); notesKeyDraft = ""
                    }
                }
            }
            Section("Auto-save to folder") {
                // The chosen folder's name, or the offer to pick one. The
                // bookmark (the actual grant) lives in FolderSettings; this
                // row only shows what the user picked.
                if FolderSettings.hasFolder {
                    Text("Saving to “\(FolderExporter.folderDisplayName)”")
                        .font(.footnote).foregroundStyle(Ink.secondary)
                    Button("Choose a different folder") { pickFolder() }
                    Button("Stop auto-saving", role: .destructive) { model.exporter.forget() }
                } else {
                    Text("After processing, the meeting's Markdown is written into a folder you pick — a notes vault, a synced folder, or anywhere in Files.")
                        .font(.footnote).foregroundStyle(Ink.secondary)
                    Button("Choose folder…") { pickFolder() }
                }
            }
            Section("Privacy") {
                Text("Recordings and notes stay on this device. Audio is sent only to the service you configure above.")
                    .font(.footnote).foregroundStyle(Ink.secondary)
            }
            Section {
                Text("Cicada 1.0 · iOS 16.4+ · arm64").font(.footnote).foregroundStyle(Ink.tertiary)
                    .frame(maxWidth: .infinity, alignment: .center)
            }
        }
        .navigationTitle("Settings")

    }
    /// Folder pick for auto-save: the picker grants access to the URL for
    /// this callback only; choose() turns it into a security-scoped
    /// bookmark, which is what persists the grant across launches.
    private func pickFolder() {
        FolderPickerPresenter.shared.present { url in
            model.exporter.choose(url)
            model.exporter.retryPending(sessions: model.store.sessions)
        }
    }

    private func saveKey() {
        let k = keyDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !k.isEmpty else { return }
        Keychain.set(k, for: AISettings.keyKey); keyDraft = ""
        Task { _ = await model.ai.checkKey() }
    }
    /// Same shape as saveKey(), for the notes service's own key.
    private func saveNotesKey() {
        let k = notesKeyDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !k.isEmpty else { return }
        Keychain.set(k, for: AISettings.notesKeyKey); notesKeyDraft = ""
        if AISettings.notesApart { Task { _ = await model.ai.checkNotes() } }
    }

    private var checkRunning: Bool { model.ai.status == .checking }
    private var notesCheckRunning: Bool { model.ai.notesStatus == .checking }

    /// The status dot + word for the check result: checking spinner, OK, or a
    /// plain "not set up" only when nothing is configured at all. The failure
    /// reason prints on its own row below (checkStatusReason).
    @ViewBuilder private var checkStatusView: some View {
        switch model.ai.status {
        case .checking:
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Checking…").foregroundStyle(Ink.secondary)
            }
        case .ok:
            HStack(spacing: 6) {
                Circle().fill(Ink.ok).frame(width: 8, height: 8)
                Text("Connected").foregroundStyle(Ink.secondary)
            }
        case .failed:
            HStack(spacing: 6) {
                Circle().fill(Ink.accent).frame(width: 8, height: 8)
                Text("Check failed").foregroundStyle(Ink.secondary)
            }
        case .idle:
            HStack(spacing: 6) {
                Circle().fill(AISettings.isConfigured ? Ink.ok : Color.gray).frame(width: 8, height: 8)
                Text(AISettings.isConfigured ? "Key saved" : "Not set up").foregroundStyle(Ink.secondary)
            }
        }
    }

    /// The notes check's status, mirroring checkStatusView. Only shown when a
    /// notes address is configured (the check pings that address, with the
    /// notes key when one was saved, else the main key).
    @ViewBuilder private var notesCheckStatusView: some View {
        switch model.ai.notesStatus {
        case .checking:
            HStack(spacing: 6) {
                ProgressView().controlSize(.small)
                Text("Checking…").foregroundStyle(Ink.secondary)
            }
        case .ok:
            HStack(spacing: 6) {
                Circle().fill(Ink.ok).frame(width: 8, height: 8)
                Text("Connected").foregroundStyle(Ink.secondary)
            }
        case .failed:
            HStack(spacing: 6) {
                Circle().fill(Ink.accent).frame(width: 8, height: 8)
                Text("Check failed").foregroundStyle(Ink.secondary)
            }
        case .idle:
            HStack(spacing: 6) {
                Circle().fill(Color.gray).frame(width: 8, height: 8)
                Text("Not checked yet").foregroundStyle(Ink.secondary)
            }
        }
    }
}
