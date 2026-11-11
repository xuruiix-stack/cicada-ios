import Foundation

struct TranscriptLine: Identifiable, Codable, Equatable {
    var id: String = UUID().uuidString
    var t: Double            // seconds from session start
    var speaker: Int = 1     // 1-based speaker number (speaker separation)
    var speakerName: String? // user-assigned name, if any
    var src: String          // original-language text
    var dst: String?         // translated text
    var srcLang: String = "auto"
    var dstLang: String = "en"
    /// Which voice the voiceprint engine grouped this line under (SpeakerID.swift).
    /// Persisted so the speaker numbers survive a relaunch; nil = not identified
    /// (line too short, model missing, or the voice never reached the shown count).
    var voice: Int?
}

extension TranscriptLine {
    /// Tolerant decoding: `voice` was added after sessions existed without it.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString
        t = try c.decodeIfPresent(Double.self, forKey: .t) ?? 0
        speaker = try c.decodeIfPresent(Int.self, forKey: .speaker) ?? 1
        speakerName = try c.decodeIfPresent(String.self, forKey: .speakerName)
        src = try c.decodeIfPresent(String.self, forKey: .src) ?? ""
        dst = try c.decodeIfPresent(String.self, forKey: .dst)
        srcLang = try c.decodeIfPresent(String.self, forKey: .srcLang) ?? "auto"
        dstLang = try c.decodeIfPresent(String.self, forKey: .dstLang) ?? "en"
        voice = try c.decodeIfPresent(Int.self, forKey: .voice)
    }
    enum CodingKeys: String, CodingKey { case id, t, speaker, speakerName, src, dst, srcLang, dstLang, voice }
}

struct ActionItem: Codable, Equatable { var text: String; var owner: String?; var done: Bool = false }

struct MeetingNotes: Codable, Equatable {
    var summary: String = ""
    var topics: [Topic] = []
    var conclusions: [String] = []
    var actionItems: [ActionItem] = []
    var terms: [String] = []
    var isEmpty: Bool { summary.isEmpty && topics.isEmpty && conclusions.isEmpty && actionItems.isEmpty }
    struct Topic: Codable, Equatable { var title: String; var points: [String] = [] }
}

/// Where a session stands in the transcribe → translate → notes pipeline.
/// Persisted so the Notes tab can tell not-run / running / failed apart after
/// a relaunch, not just while the one-shot alert is on screen.
enum ProcessingState: String, Codable {
    case idle          // nothing has run yet
    case running       // pipeline in progress ("Generating…")
    case done          // notes (or at least the transcript) completed
    case failed        // a step failed; lastError holds the reason
}

struct Session: Identifiable, Codable, Equatable {
    var id: String = UUID().uuidString
    var title: String = ""
    var startedAt: Date = Date()
    var endedAt: Date?
    var lines: [TranscriptLine] = []
    var notes: MeetingNotes?
    var audioFile: String?   // relative filename under Documents/Recordings
    // Pipeline state, persisted with the session. Synthesized Codable ignores
    // default values (a missing key throws for non-optional fields), which would
    // make every pre-existing session JSON fail to decode and disappear from
    // the list — hence the hand-written init(from:) in the extension below.
    var processingState: ProcessingState = .idle
    /// True once a server transcription has succeeded for this session — lets a
    /// refresh skip re-uploading all audio chunks and go straight to notes.
    var transcribed = false
    /// Persisted reason for the last failed step, shown on the Notes tab.
    var lastError: String?
    /// Why the on-device live captions never came (model not downloaded, locale
    /// unsupported…), captured at stop when the fallback transcript is empty
    /// because of it. Shown on the session's transcript page so "no lines"
    /// is never mistaken for silence. The pipeline's own errors live in
    /// `lastError`; this one only ever explains the missing live captions.
    var liveCaptionNote: String?
    /// Chunks that failed transcription even after their retry (Ling path).
    var failedChunkCount = 0
    var duration: Double { endedAt.map { $0.timeIntervalSince(startedAt) } ?? 0 }
}

extension Session {
    enum CodingKeys: String, CodingKey {
        case id, title, startedAt, endedAt, lines, notes, audioFile
        case processingState, transcribed, lastError, liveCaptionNote, failedChunkCount
    }
    /// Tolerant decoding: every field falls back to its default, so session
    /// files written before the pipeline-state fields existed keep loading.
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decodeIfPresent(String.self, forKey: .id) ?? UUID().uuidString
        title = try c.decodeIfPresent(String.self, forKey: .title) ?? ""
        startedAt = try c.decodeIfPresent(Date.self, forKey: .startedAt) ?? Date()
        endedAt = try c.decodeIfPresent(Date.self, forKey: .endedAt)
        lines = try c.decodeIfPresent([TranscriptLine].self, forKey: .lines) ?? []
        notes = try c.decodeIfPresent(MeetingNotes.self, forKey: .notes)
        audioFile = try c.decodeIfPresent(String.self, forKey: .audioFile)
        processingState = try c.decodeIfPresent(ProcessingState.self, forKey: .processingState) ?? .idle
        transcribed = try c.decodeIfPresent(Bool.self, forKey: .transcribed) ?? false
        lastError = try c.decodeIfPresent(String.self, forKey: .lastError)
        liveCaptionNote = try c.decodeIfPresent(String.self, forKey: .liveCaptionNote)
        failedChunkCount = try c.decodeIfPresent(Int.self, forKey: .failedChunkCount) ?? 0
    }
    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(title, forKey: .title)
        try c.encode(startedAt, forKey: .startedAt)
        try c.encodeIfPresent(endedAt, forKey: .endedAt)
        try c.encode(lines, forKey: .lines)
        try c.encodeIfPresent(notes, forKey: .notes)
        try c.encodeIfPresent(audioFile, forKey: .audioFile)
        try c.encode(processingState, forKey: .processingState)
        try c.encode(transcribed, forKey: .transcribed)
        try c.encodeIfPresent(lastError, forKey: .lastError)
        try c.encodeIfPresent(liveCaptionNote, forKey: .liveCaptionNote)
        try c.encode(failedChunkCount, forKey: .failedChunkCount)
    }
}

enum TranscriptExport {
    static func markdown(_ s: Session) -> String {
        var b = "# \(displayTitle(s))\n\n"
        let df = DateFormatter(); df.dateStyle = .medium; df.timeStyle = .short
        b += "\(df.string(from: s.startedAt)) · \(Int(s.duration/60)) min\n\n"
        if let n = s.notes, !n.isEmpty {
            b += "## Summary\n\n\(n.summary)\n\n"
            for t in n.topics { b += "### \(t.title)\n\n"; for p in t.points { b += "- \(p)\n" }; b += "\n" }
            if !n.conclusions.isEmpty { b += "### Conclusions\n\n"; for c in n.conclusions { b += "- \(c)\n" }; b += "\n" }
            if !n.actionItems.isEmpty { b += "### Action items\n\n"; for a in n.actionItems { b += "- [\(a.done ? "x" : " ")] \(a.text)\(a.owner.map { " (\($0))" } ?? "")\n" }; b += "\n" }
        }
        b += "## Transcript\n\n"
        for l in s.lines {
            let who = l.speakerName ?? "Speaker \(l.speaker)"
            b += "**[\(clock(l.t))] \(who)**  \n\(l.src)\n"
            if let d = l.dst, d != l.src { b += "> \(d)\n" }
            b += "\n"
        }
        return b
    }
    static func displayTitle(_ s: Session) -> String {
        if !s.title.isEmpty { return s.title }
        let df = DateFormatter(); df.dateFormat = "M月d日 HH:mm"
        return df.string(from: s.startedAt)
    }
    static func clock(_ t: Double) -> String {
        let s = Int(max(0, t)); return String(format: "%02d:%02d", s/60, s%60)
    }
}
