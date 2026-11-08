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

struct Session: Identifiable, Codable, Equatable {
    var id: String = UUID().uuidString
    var title: String = ""
    var startedAt: Date = Date()
    var endedAt: Date?
    var lines: [TranscriptLine] = []
    var notes: MeetingNotes?
    var audioFile: String?   // relative filename under Documents/Recordings
    var duration: Double { endedAt.map { $0.timeIntervalSince(startedAt) } ?? 0 }
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
