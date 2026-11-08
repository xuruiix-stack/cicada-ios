import Foundation
import Combine

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
    case notConfigured, invalidResponse, http(Int), underlying(Error)
    var errorDescription: String? {
        switch self {
        case .notConfigured: return "Set the API base URL and key in Settings › AI Service."
        case .invalidResponse: return "The service returned an unexpected response."
        case .http(let c): return "Service error (HTTP \(c))."
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

    func chat(system: String, user: String, json: Bool = false) async throws -> String {
        let url = try endpoint("chat/completions")
        var r = request(to: url); r.httpMethod = "POST"
        let body: [String: Any] = ["model": AISettings.chatModel,
            "messages": [["role": "system", "content": system], ["role": "user", "content": user]],
            "temperature": 0.3, "max_tokens": CicadaTuning.notesMaxTokens]
        r.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, resp) = try await URLSession.shared.data(for: r)
        guard let h = resp as? HTTPURLResponse, h.statusCode == 200 else {
            throw AIError.http((resp as? HTTPURLResponse)?.statusCode ?? 0) }
        guard let j = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let choices = j["choices"] as? [[String: Any]],
              let msg = choices.first?["message"] as? [String: Any],
              let content = msg["content"] as? String else { throw AIError.invalidResponse }
        _ = json; return content
    }

    /// Speech-to-text via /audio/transcriptions (multipart file upload).
    func transcribe(audioURL: URL, language: String? = nil) async throws -> String {
        let url = try endpoint("audio/transcriptions")
        var r = request(to: url); r.httpMethod = "POST"
        let boundary = UUID().uuidString
        r.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")
        var body = Data()
        func field(_ name: String, _ v: String) {
            body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"\(name)\"\r\n\r\n\(v)\r\n".data(using: .utf8)!) }
        field("model", AISettings.transcribeModel); field("response_format", "json")
        if let language { field("language", language) }
        let file = try Data(contentsOf: audioURL)
        body.append("--\(boundary)\r\nContent-Disposition: form-data; name=\"file\"; filename=\"\(audioURL.lastPathComponent)\"\r\nContent-Type: audio/m4a\r\n\r\n".data(using: .utf8)!)
        body.append(file); body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)
        r.httpBody = body
        let (data, resp) = try await URLSession.shared.data(for: r)
        guard let h = resp as? HTTPURLResponse, h.statusCode == 200 else {
            throw AIError.http((resp as? HTTPURLResponse)?.statusCode ?? 0) }
        guard let j = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let text = j["text"] as? String else { throw AIError.invalidResponse }
        return text
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
        guard let d = raw.data(using: .utf8),
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
