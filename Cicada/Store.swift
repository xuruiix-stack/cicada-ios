import Foundation

/// Original tuning constants (linux/cicada/core/*.py, voiceprint.c). See MECHANICS.md.
enum CicadaTuning {
    // Audio (audio.py / segmenter.py)
    static let sampleRate = 16000.0
    static let captureChunk = 0.1            // 100 ms
    static let gateThreshold: Float = 0.01
    static let pauseEndsSegment = 0.6
    static let maxSegmentLength = 8.0
    static let preRoll = 0.15
    // Speaker separation (speakers.py / voiceprint.c)
    static let voiceprintDims = 256
    static let minEmbedSeconds = 0.5
    static let maxEmbedSeconds = 20.0
    static let minEmbedFrames = 50
    static let joinThreshold = 0.5
    static let startNewThreshold = 0.5
    static let newVoiceMinSeconds = 2.0
    static let fbankBins = 80
    static let fbankWindow = 400
    static let fbankShift = 160
    static let fbankFFT = 512
    static let preEmphasis: Float = 0.97
    static let shownAfter = 5
    // Notes (notes.py)
    static let notesInterval = 45.0
    static let tickerTick = 5.0
    static let firstNotesLines = 3
    static let firstNotesWait = 20.0
    static let editableTopics = 2
    static let notesMaxTokens = 6000
    static let notesTimeout = 90.0
    static let maxTopics = 8
    static let liveMaxTopics = 12
    static let topicSummaryLimit = 160
    static let glossaryTerms = 30
    static let splitMinutes = 100.0
    static let partMinutes = 60.0
    // AI (providers.py / ling.py)
    static let spacingLing = 0.6
    static let spacingAggregator = 0.1
    // Storage (session_store.py)
    static let saveDebounce = 1.5
}

/// Local storage: sessions as JSON + audio files under Documents/Cicada.
final class SessionStore: ObservableObject {
    @Published var sessions: [Session] = []
    private let dir: URL
    private let audioDir: URL

    init() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        dir = docs.appendingPathComponent("Cicada", isDirectory: true)
        audioDir = dir.appendingPathComponent("Recordings", isDirectory: true)
        try? FileManager.default.createDirectory(at: audioDir, withIntermediateDirectories: true)
        load()
    }

    func load() {
        guard let files = try? FileManager.default.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil)
            .filter({ $0.pathExtension == "json" }) else { sessions = []; return }
        sessions = files.compactMap { f in
            guard let d = try? Data(contentsOf: f) else { return nil }
            return try? JSONDecoder().decode(Session.self, from: d)
        }.sorted { $0.startedAt > $1.startedAt }
    }

    func save(_ s: Session) {
        if let i = sessions.firstIndex(where: { $0.id == s.id }) { sessions[i] = s }
        else { sessions.insert(s, at: 0) }
        if let d = try? JSONEncoder().encode(s) {
            try? d.write(to: dir.appendingPathComponent("\(s.id).json"), options: .atomic)
        }
    }
    func delete(_ s: Session) {
        sessions.removeAll { $0.id == s.id }
        try? FileManager.default.removeItem(at: dir.appendingPathComponent("\(s.id).json"))
        if let f = s.audioFile { try? FileManager.default.removeItem(at: audioDir.appendingPathComponent(f)) }
    }
    func audioURL(for s: Session) -> URL? { s.audioFile.map { audioDir.appendingPathComponent($0) } }

    /// Speaker separation (SpeakerID.swift): a line counts towards its voice's number only
    /// when substantive. Original rule (speakers.py substantive/_cjk): >=3 words, or >=4 CJK chars.
    static func substantive(_ text: String) -> Bool {
        // Original rule (speakers.py substantive/_cjk): >=3 words, or >=4 CJK chars.
        let words = text.split(whereSeparator: { $0.isWhitespace })
            .filter { $0.contains(where: { $0.isLetter }) }
        if words.count >= 3 { return true }
        return text.unicodeScalars.filter { isCJK($0) }.count >= 4
    }
    static func isCJK(_ s: UnicodeScalar) -> Bool {
        let o = s.value
        return (0x3400...0x4DBF).contains(o) || (0x4E00...0x9FFF).contains(o)
            || (0xF900...0xFAFF).contains(o) || (0x20000...0x3FFFF).contains(o)
            || (0x3040...0x30FF).contains(o) || (0xAC00...0xD7A3).contains(o)
            || o == 0x3005 || o == 0x3006 || o == 0x3007
    }
}
