import Foundation
import Combine
import AVFoundation

/// User-configurable OpenAI-compatible AI service. No key is ever hardcoded:
/// base URL, key and model names come only from Settings (UserDefaults + Keychain).
enum AISettings {
    static let baseURLKey = "ai.baseURL"
    static let keyKey = "ai.apiKey"          // stored in Keychain, not UserDefaults
    static let chatModelKey = "ai.chatModel"
    static let transcribeModelKey = "ai.transcribeModel"
    static let targetLangKey = "ai.targetLang"

    static var baseURL: String {
        get { UserDefaults.standard.string(forKey: baseURLKey) ?? "" }
        set { UserDefaults.standard.set(newValue, forKey: baseURLKey) }
    }
    static var chatModel: String {
        get { UserDefaults.standard.string(forKey: chatModelKey) ?? "gpt-4o-mini" }
        set { UserDefaults.standard.set(newValue, forKey: chatModelKey) }
    }
    static var transcribeModel: String {
        get { UserDefaults.standard.string(forKey: transcribeModelKey) ?? "whisper-1" }
        set { UserDefaults.standard.set(newValue, forKey: transcribeModelKey) }
    }
    static var targetLang: String {
        get { UserDefaults.standard.string(forKey: targetLangKey) ?? "en" }
        set { UserDefaults.standard.set(newValue, forKey: targetLangKey) }
    }
    static var hasKey: Bool { Keychain.string(for: keyKey) != nil }
    static var isConfigured: Bool { !baseURL.isEmpty && hasKey }
}

enum Keychain {
    static func string(for key: String) -> String? {
        var q = [kSecClass as String: kSecClassGenericPassword,
                 kSecAttrAccount as String: key,
                 kSecAttrService as String: "com.wquguru.cicada"] as [String: Any]
        q[kSecReturnData as String] = true; q[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        guard SecItemCopyMatching(q as CFDictionary, &item) == errSecSuccess,
              let d = item as? Data else { return nil }
        return String(data: d, encoding: .utf8)
    }
    static func set(_ value: String, for key: String) {
        let data = Data(value.utf8)
        let base: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                   kSecAttrAccount as String: key,
                                   kSecAttrService as String: "com.wquguru.cicada"]
        if SecItemCopyMatching((base as [String: Any]) as CFDictionary, nil) == errSecSuccess {
            SecItemUpdate(base as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        } else {
            var add = base; add[kSecValueData as String] = data
            SecItemAdd(add as CFDictionary, nil)
        }
    }
    static func remove(for key: String) {
        let q = [kSecClass as String: kSecClassGenericPassword,
                 kSecAttrAccount as String: key,
                 kSecAttrService as String: "com.wquguru.cicada"] as [String: Any]
        SecItemDelete(q as CFDictionary)
    }
}

enum AIError: LocalizedError {
    case notConfigured, invalidResponse, http(Int, String?), underlying(Error)
    var errorDescription: String? {
        switch self {
        case .notConfigured: return "Set the API base URL and key in Settings › AI Service."
        case .invalidResponse: return "The service returned an unexpected response."
        // Show the server's own reason (Ling/OpenRouter reject wrong models, formats
        // and over-long audio with 400 + message) instead of a bare status code.
        case .http(let c, let detail):
            if let d = detail, !d.isEmpty { return "Service error (HTTP \(c)): \(d)" }
            return "Service error (HTTP \(c))."
        case .underlying(let e): return e.localizedDescription
        }
    }
}

/// OpenAI-compatible client: POST {base}/chat/completions and {base}/audio/transcriptions.
final class AIClient: ObservableObject {
    @Published var status: String = AISettings.isConfigured ? "Connected" : "Not set up"

    private func endpoint(_ path: String) throws -> URL {
        guard AISettings.isConfigured, let key = Keychain.string(for: AISettings.keyKey) else { throw AIError.notConfigured }
        _ = key
        var b = AISettings.baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        if !b.contains("://") { b = "https://" + b }
        if !b.hasSuffix("/") { b += "/" }
        guard let u = URL(string: b + path) else { throw AIError.notConfigured }
        return u
    }
    private func request(to url: URL) -> URLRequest {
        var r = URLRequest(url: url); r.timeoutInterval = CicadaTuning.notesTimeout
        r.setValue("Bearer \(Keychain.string(for: AISettings.keyKey) ?? "")", forHTTPHeaderField: "Authorization")
        r.setValue("application/json", forHTTPHeaderField: "Content-Type")
        return r
    }

    /// Best-effort reason from a non-200 response body: the JSON "message"/"error"
    /// field if present, else the raw body. Capped so an HTML error page can't
    /// flood the alert.
    private func bodyDetail(_ data: Data) -> String? {
        if let j = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let m = j["message"] as? String { return String(m.prefix(500)) }
            if let e = j["error"] as? String { return String(e.prefix(500)) }
            if let e = j["error"] as? [String: Any], let m = e["message"] as? String {
                return String(m.prefix(500))
            }
        }
        let s = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return s.isEmpty ? nil : String(s.prefix(500))
    }

    private func throwHTTP(_ resp: URLResponse, body data: Data) -> AIError {
        let code = (resp as? HTTPURLResponse)?.statusCode ?? 0
        return .http(code, bodyDetail(data))
    }

    func chat(system: String, user: String, json: Bool = false) async throws -> String {
        let url = try endpoint("chat/completions")
        var r = request(to: url); r.httpMethod = "POST"
        let body: [String: Any] = ["model": AISettings.chatModel,
            "messages": [["role": "system", "content": system], ["role": "user", "content": user]],
            "temperature": 0.3, "max_tokens": CicadaTuning.notesMaxTokens]
        r.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, resp) = try await URLSession.shared.data(for: r)
        guard let h = resp as? HTTPURLResponse, h.statusCode == 200 else {
            throw throwHTTP(resp, body: data) }
        guard let j = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = j["choices"] as? [[String: Any]],
              let msg = choices.first?["message"] as? [String: Any],
              let content = msg["content"] as? String else { throw AIError.invalidResponse }
        _ = json; return content
    }

    /// Speech-to-text via /audio/transcriptions (multipart file upload), made
    /// provider-aware (facts verified against the providers' docs, see below).
    func transcribe(audioURL: URL, language: String? = nil) async throws -> String {
        if baseURLHost.contains("ant-ling.com") {
            // Ling (api.ant-ling.com) /v1/audio/transcriptions accepts ONLY model
            // Ming-3.0-asr-tiny, ONLY .wav (not .m4a), ≤10 MB and ≤30 s per call.
            // Convert the recorded .m4a to 16 kHz mono LinearPCM .wav on device,
            // split into ≤30 s chunks, transcribe each, join the texts.
            let chunks = try lingWavChunks(from: audioURL)
            defer {
                for c in chunks { try? FileManager.default.removeItem(at: c) }
                if let d = chunks.first?.deletingLastPathComponent() {
                    try? FileManager.default.removeItem(at: d)
                }
            }
            var texts: [String] = []
            for c in chunks {
                let t = try await transcribeFile(c, model: lingTranscribeModel, language: language)
                if !t.isEmpty { texts.append(t) }
            }
            return texts.joined(separator: " ")
        }
        return try await transcribeFile(audioURL, model: transcribeModelForProvider, language: language)
    }

    /// Host of the configured base URL (scheme optional, as typed in Settings).
    private var baseURLHost: String {
        var b = AISettings.baseURL.trimmingCharacters(in: .whitespacesAndNewlines)
        if !b.contains("://") { b = "https://" + b }
        return (URL(string: b)?.host ?? b).lowercased()
    }

    /// Model sent to /audio/transcriptions for the configured provider.
    /// OpenRouter requires namespaced models: bare whisper-1 → openai/whisper-1.
    private var transcribeModelForProvider: String {
        let m = AISettings.transcribeModel
        if baseURLHost.contains("openrouter.ai") && m == "whisper-1" { return "openai/whisper-1" }
        return m
    }

    /// Ling's transcriptions endpoint accepts only this model, regardless of the
    /// settings field (whisper-1 / Ming-3.0-audio-flash are rejected with HTTP 400).
    private let lingTranscribeModel = "Ming-3.0-asr-tiny"
    private let lingChunkSeconds = 30.0

    /// One multipart POST /audio/transcriptions call with a file already on disk.
    private func transcribeFile(_ fileURL: URL, model: String, language: String?) async throws -> String {
        let url = try endpoint("audio/transcriptions")
        var r = request(to: url); r.httpMethod = "POST"
        let boundary = UUID().uuidString
        r.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        func field(_ name: String, _ v: String) {
            body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(v)\r\n".data(using: .utf8)!) }
        field("model", model); field("response_format", "json")
        if let language { field("language", language) }
        let file = try Data(contentsOf: fileURL)
        let mime = fileURL.pathExtension.lowercased() == "wav" ? "audio/wav" : "audio/m4a"
        body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"\(fileURL.lastPathComponent)\"\r\nContent-Type: \(mime)\r\n\r\n".data(using: .utf8)!)
        body.append(file); body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        r.httpBody = body
        let (data, resp) = try await URLSession.shared.data(for: r)
        guard let h = resp as? HTTPURLResponse, h.statusCode == 200 else {
            throw throwHTTP(resp, body: data) }
        guard let j = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = j["text"] as? String else { throw AIError.invalidResponse }
        return text
    }

    /// .m4a → 16 kHz mono LinearPCM .wav chunk files of ≤30 s each (Ling's limits),
    /// via AVAudioFile read + write and AVAudioConverter when rates differ — the
    /// same AVFoundation APIs Recorder.swift already uses. Caller deletes the files.
    private func lingWavChunks(from audioURL: URL) throws -> [URL] {
        let src = try AVAudioFile(forReading: audioURL)
        guard let out = AVAudioFormat(commonFormat: .pcmFormatFloat32,
                                      sampleRate: CicadaTuning.sampleRate, channels: 1, interleaved: false),
              src.processingFormat.sampleRate > 0 else { throw AIError.invalidResponse }
        let conv = src.processingFormat == out ? nil
            : AVAudioConverter(from: src.processingFormat, to: out)
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("cicada-asr-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var chunks: [URL] = []
        let pcmSettings: [String: Any] = [AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: CicadaTuning.sampleRate, AVNumberOfChannelsKey: 1,
            AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false, AVLinearPCMIsNonInterleaved: false]
        func newChunk() throws -> AVAudioFile {
            let u = dir.appendingPathComponent("chunk\(chunks.count).wav")
            chunks.append(u)
            return try AVAudioFile(forWriting: u, settings: pcmSettings)
        }
        // If anything throws below, delete the partial chunk files instead of
        // leaking them in tmp.
        var completed = false
        defer {
            if !completed {
                for u in chunks { try? FileManager.default.removeItem(at: u) }
                try? FileManager.default.removeItem(at: dir)
            }
        }
        var f: AVAudioFile?
        var inChunk = 0
        let chunkFrames = Int(CicadaTuning.sampleRate * lingChunkSeconds)
        let readCap = AVAudioFrameCount(16384)
        while src.framePosition < src.length {
            guard let inBuf = AVAudioPCMBuffer(pcmFormat: src.processingFormat, frameCapacity: readCap) else { break }
            try src.read(into: inBuf)
            guard inBuf.frameLength > 0 else { break }
            var converted = inBuf
            if let conv {
                let ratio = out.sampleRate / src.processingFormat.sampleRate
                let cap = AVAudioFrameCount((Double(inBuf.frameLength) * ratio).rounded(.up)) + 32
                guard let outBuf = AVAudioPCMBuffer(pcmFormat: out, frameCapacity: cap) else { break }
                var fed = false; var cerr: NSError?
                conv.convert(to: outBuf, error: &cerr) { _, status in
                    if fed { status.pointee = .noDataNow; return (nil, .noDataNow) }
                    fed = true; status.pointee = .haveData
                    return (inBuf, .haveData)
                }
                guard fed, cerr == nil, outBuf.frameLength > 0 else { break }
                converted = outBuf
            }
            // Distribute the converted frames across ≤30 s chunk files. Chunk
            // files are created lazily so a zero-length recording yields no
            // chunks (the liveText fallback then survives) and audio ending
            // exactly on a boundary leaves no trailing empty file.
            var offset = 0
            while offset < Int(converted.frameLength) {
                if f == nil || inChunk >= chunkFrames { f = try newChunk(); inChunk = 0 }
                let n = min(chunkFrames - inChunk, Int(converted.frameLength) - offset)
                guard n > 0, let w = f,
                      let part = AVAudioPCMBuffer(pcmFormat: out, frameCapacity: AVAudioFrameCount(n)) else { break }
                part.frameLength = AVAudioFrameCount(n)
                if let s = converted.floatChannelData?[0], let d = part.floatChannelData?[0] {
                    d.update(from: UnsafeBufferPointer(start: s + offset, count: n))
                }
                try w.write(from: part)
                inChunk += n; offset += n
            }
        }
        // An empty result (zero-length audio) leaves the temp dir to the defer.
        completed = !chunks.isEmpty
        return chunks
    }

    func translate(_ text: String, to lang: String) async throws -> String {
        try await chat(system: "Translate faithfully; keep speaker names. Reply with only the translation.",
                       user: "Translate to \(lang):\n\(text)")
    }

    /// Structured meeting notes: summary, topics, conclusions, action items.
    func meetingNotes(transcript: String) async throws -> MeetingNotes {
        let raw = try await chat(
            system: "You write structured meeting notes. Reply with JSON only: {\"summary\":string,\"topics\":[{\"title\":string,\"points\":[string]}],\"conclusions\":[string],\"actionItems\":[{\"text\":string,\"owner\":string|null}]}",
            user: transcript, json: true)
        // Chat models often wrap JSON in ```json … ``` fences despite the prompt;
        // strip them before parsing.
        var t = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if t.hasPrefix("```") {
            t = t.split(separator: "\n", omittingEmptySubsequences: false)
                .dropFirst().joined(separator: "\n")
            if t.hasSuffix("```") { t = String(t.dropLast(3)) }
            t = t.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        guard let d = t.data(using: .utf8),
              let j = try JSONSerialization.jsonObject(with: d) as? [String: Any] else { throw AIError.invalidResponse }
        var n = MeetingNotes(); n.summary = j["summary"] as? String ?? ""
        n.conclusions = (j["conclusions"] as? [String]) ?? []
        for t in (j["topics"] as? [[String: Any]] ?? []) {
            n.topics.append(.init(title: t["title"] as? String ?? "", points: t["points"] as? [String] ?? [])) }
        for a in (j["actionItems"] as? [[String: Any]] ?? []) {
            n.actionItems.append(.init(text: a["text"] as? String ?? "", owner: a["owner"] as? String)) }
        return n
    }

    func checkKey() async -> Bool {
        do { _ = try await chat(system: "Reply OK", user: "ping"); status = "Connected"; return true }
        catch { status = "Check failed"; return false }
    }
}
