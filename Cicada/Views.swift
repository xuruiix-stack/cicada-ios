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
    enum Tab { case records, settings }

    var selected: Session? { store.sessions.first { $0.id == selection } }
    /// The session currently being recorded (kept out of `selection` so the user
    /// stays where the Stop control is; `selection` is set only when it finishes).
    @Published var recordingId: String?

    /// Set by the background-task expiration handler: runPipeline checks it at
    /// each step boundary and saves the partial progress reached so far.
    private var pipelineCancelled = false
    /// Active UIApplication background task for the pipeline, ended when it
    /// finishes (or left to the expiration handler if the app is torn down).
    private var backgroundTaskId: UIBackgroundTaskIdentifier = .invalid
    /// True while a pipeline is in flight (set synchronously here, unlike
    /// `busy`, which runPipeline sets after its first await — too late to close
    /// the overlap race between two process() calls).
    private var pipelineActive = false

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
        }
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
        guard !pipelineActive else { return }
        guard AISettings.isConfigured else { return }
        // Mark them all up front: a later re-entrant call (the monitor can
        // fire more than once per window) must not double-book a session.
        let todo = store.sessions.filter {
            $0.endedAt != nil && $0.audioFile != nil && !$0.transcribed && !autoRetriedIds.contains($0.id)
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
            // No navigation here: the user stays on the list, where Stop and the
            // live elapsed / level / captions are. The detail page opens on Stop.
        }
        catch { message = "Recording could not start: \(error.localizedDescription)" }
    }
    func stopRecording() {
        _ = recorder.stop()
        // Look up by recordingId, not `selected`: stopping happens on the list,
        // where selection is nil and the old lookup silently dropped endedAt.
        guard let id = recordingId, var s = store.sessions.first(where: { $0.id == id }) else {
            recordingId = nil; return
        }
        recordingId = nil
        s.endedAt = Date()
        // Fallback transcript: keep the on-device live captions as one line so
        // there is text even if server transcription fails. process() replaces it
        // with the full transcription when that succeeds. Clearing here keeps a
        // stale caption from leaking into the next session.
        let live = recorder.liveText; recorder.liveText = ""
        if !live.isEmpty {
            s.lines = [TranscriptLine(t: 0, speaker: 1, src: live)]
        }
        // Same for the caption failure reason: with it on the session, the
        // transcript page can say why there are no lines at all (offline
        // model not downloaded) instead of showing bare "No transcript yet."
        s.liveCaptionNote = recorder.recognitionError
        recorder.recognitionError = nil
        store.save(s)
        selection = s.id
        // Speaker separation first, then the pipeline: applySpeakers groups the voices
        // heard in this recording and lays the transcript over the segments, and the
        // pipeline's server transcription must not race it. process() is async and
        // sets `busy`, so this returns immediately and Stop stays responsive. Both run
        // under the same Task so their saves stay ordered.
        Task {
            await self.applySpeakers(s)
            await self.process(s)
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

    /// The server transcript laid over the segment lines: each line keeps its time and
    /// voice and gets the words said in it (RecordRefiner's edit-distance alignment,
    /// bounded window per segment). Lines whose caption text was too short to align get
    /// their proportional share of the transcript by speaking time.
    private func refineTranscript(_ text: String, onto lines: [TranscriptLine]) -> [TranscriptLine] {
        let record = Array(text)
        guard !record.isEmpty, !lines.isEmpty else { return lines }
        // Each line's provisional range: its start's share of the transcript, plus
        // one segment's worth (8 s). The alignment moves the boundaries; this only
        // bounds its windows and backs up the lines with no caption text.
        let total = max(lines.last?.t ?? 0, 1)
        var segs: [RecordRefiner.Segment] = []
        for l in lines {
            let a = Int(Double(record.count) * max(0, l.t) / total)
            let b = min(record.count, Int(Double(record.count) * min(1, (l.t + 8) / total)))
            segs.append(RecordRefiner.Segment(text: l.src, provisional: a..<max(a, b)))
        }
        let ends = RecordRefiner.ends(segments: segs, record: record)
        var out = lines
        var from = 0
        for i in out.indices {
            let to = i == out.count - 1 ? record.count : (i < ends.count ? ends[i] : record.count)
            out[i].src = String(record[from..<max(from, to)]).trimmingCharacters(in: .whitespacesAndNewlines)
            from = max(from, to)
        }
        // A line the alignment left with no text (all its words went to a neighbour)
        // keeps its caption rather than showing empty.
        for i in out.indices where out[i].src.isEmpty { out[i].src = lines[i].src }
        return out
    }

    /// AI pipeline for a finished recording: transcribe → translate → notes.
    /// Wraps runPipeline(_:) in a UIApplication background task so the pipeline
    /// can finish after the user locks the screen or switches away; every caller
    /// (Stop and the manual ↻ button) gets the background protection.
    func process(_ s: Session) async {
        // Serialize: two pipelines at once (stop a second recording while the
        // first still processes) would interleave saves on the two sessions.
        // The dropped run leaves the session's persisted state as-is, so the
        // Notes tab shows the honest hint / error and ↻ retries.
        guard !pipelineActive else {
            message = "Still processing the previous recording — try again in a moment."
            return
        }
        pipelineActive = true
        defer { pipelineActive = false }
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
        pipelineCancelled = false
        await runPipeline(s)
        if backgroundTaskId == id {
            backgroundTaskId = .invalid
            if id != .invalid { UIApplication.shared.endBackgroundTask(id) }
        }
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
    func runPipeline(_ s: Session) async {
        busy = true; defer { busy = false }
        guard AISettings.isConfigured else { message = "Set the API base URL and key in Settings › AI Service."; return }
        var s = s
        var errors: [String] = []
        s.processingState = .running
        s.lastError = nil
        store.save(s)

        // Step 1 — transcribe. Skipped when a server transcription has already
        // succeeded once: a refresh after a notes-only failure must not re-upload
        // every audio chunk. The liveText fallback line counts as transcript text
        // for the notes step, but does not set `transcribed` — a server
        // transcription can still improve on it.
        if !s.transcribed, let url = store.audioURL(for: s) {
            do {
                let text: String
                var failedChunks = 0
                if ai.baseURLHost.contains("ant-ling.com") {
                    let r = try await ai.transcribeChunked(audioURL: url)
                    text = r.text; failedChunks = r.failedChunks
                } else {
                    text = try await ai.transcribe(audioURL: url)
                }
                if !text.isEmpty {
                    // Speaker separation: when the recording's segments were laid out
                    // as lines (applySpeakers — every line has a voice or a caption),
                    // the server transcript takes each line's place, aligned by its
                    // caption text (RecordRefiner, upstream 4cb8512's boundary fix
                    // included). Without segments (the model unavailable, or a
                    // recording with no voiced audio) the whole transcript stays one
                    // line, as before. The liveText fallback line never has a voice,
                    // so it takes the one-line path.
                    let segmented = s.lines.count > 1 && s.lines.allSatisfy { $0.voice != nil || !$0.src.isEmpty }
                    s.lines = segmented ? refineTranscript(text, onto: s.lines)
                        : [TranscriptLine(t: 0, speaker: 1, src: text)]
                    s.transcribed = true
                    s.failedChunkCount = failedChunks
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
            } catch { errors.append("Transcription failed: \(error.localizedDescription)") }
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
        if AISettings.simultaneousInterpretation && !s.lines.isEmpty && AISettings.targetLang != "auto" {
            for i in s.lines.indices {
                do { s.lines[i].dst = try await ai.translate(s.lines[i].src, to: AISettings.targetLang) }
                catch { errors.append("Translation failed: \(error.localizedDescription)"); break }
            }
            store.save(s)
        }
        if pipelineCancelled {
            s.processingState = .failed
            s.lastError = errors.isEmpty ? "Processing was interrupted." : errors.joined(separator: " · ")
            store.save(s); return
        }

        // Step 3 — notes. Runs whenever ANY transcript text exists (server
        // transcript or the liveText fallback line); a step-1/2 error above
        // does not skip it. Speaker labels match the transcript rows: an
        // unidentified voice is "Unknown", not a false "Speaker 1".
        let numbers = SpeakerLabels.voiceNumbers(s.lines)
        let transcript = s.lines.map { l -> String in
            let who: String
            if let v = l.voice, let n = numbers[v] { who = "Speaker \(n)" }
            else if numbers.isEmpty { who = "Speaker \(l.speaker)" }
            else { who = "Unknown" }
            return "\(who): \(l.src)"
        }.joined(separator: "\n")
        if !transcript.isEmpty {
            do { s.notes = try await ai.meetingNotes(transcript: transcript) }
            catch { errors.append("Notes failed: \(error.localizedDescription)") }
        } else if s.audioFile?.isEmpty == false {
            errors.append("No transcript text to write notes from.")
        }

        s.lastError = errors.isEmpty ? nil : errors.joined(separator: " · ")
        s.processingState = (s.lastError == nil) ? .done : .failed
        store.save(s)
        if let e = s.lastError { message = e }
        // Auto-save Markdown (item 3): once the transcript + notes are saved on
        // the session, its Markdown goes into the user's chosen folder. Strictly
        // after the final save, and its own do/catch: a stale bookmark or a write
        // error is logged and surfaced by the exporter, never a pipeline failure
        // or a block on the session that just finished. No folder chosen: nothing.
        if !s.lines.isEmpty { exporter.write(s) }
    }

    func exportMarkdown(_ s: Session) -> String { TranscriptExport.markdown(s) }
}

// MARK: - Views

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
                        Button("Stop", role: .destructive) { model.stopRecording() }
                    } else {
                        Button { Task { await model.startRecording() } } label: {
                            Image(systemName: "mic.fill").font(.title2)
                                .foregroundStyle(.white).frame(width: 52, height: 52)
                                .background(Ink.accent, in: Circle())
                        }
                    }
                }
                .padding(.vertical, 4)
            }
            Section("Meetings") {
                ForEach(model.store.sessions) { s in
                    Button { model.selection = s.id } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(TranscriptExport.displayTitle(s)).font(.body.weight(.medium)).foregroundStyle(Ink.ink)
                            Text("\(Int(s.duration/60)) min · \(s.lines.count) lines").font(.footnote).foregroundStyle(Ink.secondary)
                        }
                    }
                }
                .onDelete { if let i = $0.first, i < model.store.sessions.count { model.store.delete(model.store.sessions[i]); model.selection = nil } }
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
                    Picker("Page", selection: $page) {
                        Text("Transcript").tag(0); Text("Notes").tag(1)
                    }.pickerStyle(.segmented).padding(.top, 16)
                    if page == 0 {
                        // Speaker labels need the whole session's voice numbers (a voice
                        // gets a number only once two voices are on show), so compute
                        // them here once instead of per row.
                        let numbers = SpeakerLabels.voiceNumbers(session.lines)
                        ForEach(session.lines) { l in
                            TranscriptRow(line: l, numbers: numbers, playing: l.id == playing)
                                .id(l.id)
                                // Tapping a line starts playback from its timestamp
                                // (the official app's seek-on-line-tap).
                                .contentShape(Rectangle())
                                .onTapGesture { following = true; player.seek(l.t) }
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
                    Button { Task { await model.process(session) } } label: {
                        model.busy ? AnyView(ProgressView()) : AnyView(Image(systemName: "arrow.triangle.2.circlepath"))
                    }.disabled(model.busy || model.recorder.isRecording)
                    ShareLink(item: model.exportMarkdown(session), subject: Text(TranscriptExport.displayTitle(session))) {
                        Image(systemName: "square.and.arrow.up")
                    }
                }
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
        FolderPickerPresenter.shared.present { url in model.exporter.choose(url) }
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
