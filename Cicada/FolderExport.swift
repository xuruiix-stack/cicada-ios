import Foundation
import UniformTypeIdentifiers

/// 存到文件夹 (folder export): the file a finished pipeline's session becomes
/// (.refs/folder-export.md — the cross-platform contract). The frontmatter and
/// the file name are the contract; the body is 导出 Markdown's
/// (TranscriptExport.markdown), exactly what ShareLink already exports, so a
/// vault (Obsidian, Logseq) or an agent watching the folder picks up the same
/// file the Mac app would have written.
enum FolderExportFormat {
    /// The record's own name: what the user called it, else empty (iOS has no
    /// imported files and no notes title of its own — the contract's
    /// remaining fallbacks don't apply to this build). Never `displayTitle`,
    /// which is localized and carries a date.
    static func name(_ s: Session) -> String {
        s.title.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// A name every system accepts and a vault can link to: no path, wildcard
    /// or link characters, at most 120 UTF-8 bytes (vaults sync between
    /// systems, so the union of their rules). Ported verbatim from the Mac.
    static func safeName(_ name: String) -> String {
        let banned = Set("/\\:*?\"<>|#^[]".unicodeScalars)
        let spaced = String(String.UnicodeScalarView(name.unicodeScalars.map { banned.contains($0) || $0.value < 0x20 ? " " : $0 }))
        var out = trim(spaced.split(whereSeparator: \.isWhitespace).joined(separator: " "))
        if out.utf8.count > 120 {
            var cut = String.UnicodeScalarView()
            var bytes = 0
            for u in out.unicodeScalars {
                bytes += String(u).utf8.count
                if bytes > 120 { break }
                cut.append(u)
            }
            out = trim(String(cut))
        }
        return out
    }

    private static func trim(_ s: String) -> String {
        var scalars = Array(s.unicodeScalars)
        while let f = scalars.first, f == " " || f == "." { scalars.removeFirst() }
        while let l = scalars.last, l == " " || l == "." { scalars.removeLast() }
        return String(String.UnicodeScalarView(scalars))
    }

    /// `<yyyy-MM-dd HHmm>[ <safe name>]`, the start in local time.
    static func stem(_ s: Session, timeZone: TimeZone = .current) -> String {
        let date = format(s.startedAt, "yyyy-MM-dd HHmm", timeZone)
        let safe = safeName(name(s))
        return safe.isEmpty ? date : "\(date) \(safe)"
    }

    /// `stem.md`, or `stem 2.md`, `stem 3.md`… when a name is taken. Cicada
    /// never overwrites a file it didn't write.
    static func fileName(_ stem: String, taken: (String) -> Bool) -> String {
        var n = 1
        while true {
            let name = n == 1 ? "\(stem).md" : "\(stem) \(n).md"
            if !taken(name) { return name }
            n += 1
        }
    }

    /// A JSON-style double-quoted string, which YAML reads as is. Contract:
    /// `\` → `\\`, `"` → `\"`, `\n` `\r` `\t` as such, other characters below
    /// U+0020 as `\u00xx` (lowercase hex), everything else as is.
    static func quote(_ s: String) -> String {
        var out = "\""
        for u in s.unicodeScalars {
            switch u {
            case "\\": out += "\\\\"
            case "\"": out += "\\\""
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            case "\t": out += "\\t"
            case _ where u.value < 0x20: out += String(format: "\\u%04x", u.value)
            default: out.unicodeScalars.append(u)
            }
        }
        return out + "\""
    }

    /// The contract's frontmatter, keys in this order. Notes have no
    /// draft/final distinction in this build, so a non-empty notes counts as
    /// "draft" (the contract allows it; there is no final pass here to gate on).
    static func frontmatter(_ s: Session, timeZone: TimeZone = .current) -> String {
        let ended = s.endedAt ?? s.startedAt
        let languages = Set(s.lines.map(\.srcLang)).sorted {
            Array($0.unicodeScalars.map(\.value)).lexicographicallyPrecedes($1.unicodeScalars.map(\.value))
        }
        let notes = s.notes.map { $0.isEmpty ? "none" : "draft" } ?? "none"
        return """
        ---
        cicada: 1
        id: \(quote(s.id))
        title: \(quote(name(s)))
        started: \(quote(format(s.startedAt, "yyyy-MM-dd'T'HH:mm:ssxxx", timeZone)))
        ended: \(quote(format(ended, "yyyy-MM-dd'T'HH:mm:ssxxx", timeZone)))
        minutes: \(max(0, Int(ended.timeIntervalSince(s.startedAt) / 60)))
        languages: [\(languages.map(quote).joined(separator: ", "))]
        source: \(quote(s.sourceFile == nil ? "meeting" : "file"))
        notes: \(quote(notes))
        tags: ["cicada"]
        ---

        """
    }

    static func document(_ s: Session) -> String { frontmatter(s) + TranscriptExport.markdown(s) }

    private static func format(_ date: Date, _ pattern: String, _ tz: TimeZone) -> String {
        let f = DateFormatter()
        f.locale = Locale(identifier: "en_US_POSIX")
        f.calendar = Calendar(identifier: .gregorian)
        f.timeZone = tz
        f.dateFormat = pattern
        return f.string(from: date)
    }
}

/// Settings for the auto-save folder: the security-scoped bookmark (the only
/// iOS-persistent form of the user's grant) plus the file names this app has
/// already written to the chosen folder, so a later write never overwrites a
/// file that isn't ours (the contract's taken-names rule).
enum FolderSettings {
    static let bookmarkKey = "export.folderBookmark"
    /// Record id → the file name Cicada wrote for it (the index the contract
    /// keeps "next to the records"; on iOS the UserDefaults entry stands in).
    static let filesKey = "export.folderFiles"

    static var bookmark: Data? {
        get { UserDefaults.standard.data(forKey: bookmarkKey) }
        set { UserDefaults.standard.set(newValue, forKey: bookmarkKey) }
    }

    /// The folder's path, for the Settings row ("saving to …"); empty when no
    /// folder is chosen. Storing the path alongside the bookmark is safe: it is
    /// display-only, the bookmark is what grants access.
    static var folderPath: String {
        get { UserDefaults.standard.string(forKey: "\(bookmarkKey).path") ?? "" }
        set { UserDefaults.standard.set(newValue, forKey: "\(bookmarkKey).path") }
    }

    static var hasFolder: Bool { bookmark != nil }

    static func fileName(for id: String) -> String? {
        (UserDefaults.standard.dictionary(forKey: filesKey) as? [String: String])?[id]
    }

    static func setFileName(_ name: String?, for id: String) {
        var files = (UserDefaults.standard.dictionary(forKey: filesKey) as? [String: String]) ?? [:]
        if let name { files[id] = name } else { files[id] = nil }
        UserDefaults.standard.set(files, forKey: filesKey)
    }

    /// The whole index goes when a new folder is chosen: files in the old one
    /// stay (the contract's rule), but they are no longer ours to track.
    static func resetFiles() {
        UserDefaults.standard.removeObject(forKey: filesKey)
    }
}

/// Writes a finished session's Markdown into the user's chosen folder
/// (upstream v1.9.0's 存到文件夹, iOS side). Called once from the pipeline's
/// end, after the session with its transcript and notes is saved — never
/// inside it, so a failure here can never fail or block the pipeline itself.
@MainActor
final class FolderExporter: ObservableObject {
    /// The last message surfaced for a folder problem, so the same failure is
    /// alerted once (a stale bookmark fails on every pipeline until fixed; one
    /// telling is enough — the rest stays in debugLog).
    private var lastSurfaced: String?

    /// The Settings row reads this for "saving to …".
    static var folderDisplayName: String {
        let p = FolderSettings.folderPath
        guard !p.isEmpty else { return "" }
        return URL(fileURLWithPath: p).lastPathComponent
    }

    /// A new folder was picked (document picker): store its bookmark and start
    /// a fresh index. The picker's URL grant dies with the picker; the
    /// bookmark is what persists it.
    func choose(_ url: URL) {
        stopAccess()
        let scoped = url // document-picker URLs are already security-scoped
        let gotAccess = scoped.startAccessingSecurityScopedResource()
        defer { if gotAccess { scoped.stopAccessingSecurityScopedResource() } }
        do {
            // iOS bookmarks are inherently security-scoped: creation takes no
            // options (.withSecurityScope is macOS-only and does not compile here).
            // lose the user's grant across launches).
            let data = try scoped.bookmarkData(options: [])
            FolderSettings.bookmark = data
            FolderSettings.folderPath = url.path
            FolderSettings.resetFiles()
            lastSurfaced = nil
            debugLog("folder export: chose \(url.path)")
        } catch {
            debugLog("folder export: bookmark failed: \(error.localizedDescription)")
        }
        // The Settings row flips between "choose" and "saving to …".
        objectWillChange.send()
    }

    func forget() {
        stopAccess()
        FolderSettings.bookmark = nil
        FolderSettings.folderPath = ""
        FolderSettings.resetFiles()
        lastSurfaced = nil
        objectWillChange.send()
    }

    // MARK: - Writing

    /// The resolved folder, with its scope held for the duration of a write.
    /// Resolved at write time, never cached across writes: a bookmark can go
    /// stale between meetings (folder deleted, iCloud off) and iOS only
    /// reports it when resolved.
    private var heldFolder: URL?
    private var heldAccess = false

    private func stopAccess() {
        if heldAccess { heldFolder?.stopAccessingSecurityScopedResource() }
        heldAccess = false
        heldFolder = nil
    }

    /// Resolve the bookmark and take the scope. Returns nil and surfaces once
    /// when it cannot be done (stale bookmark — the folder is gone or the
    /// grant was lost; the user must re-pick it in Settings).
    private func resolveFolder() -> URL? {
        guard let data = FolderSettings.bookmark else { return nil }
        var stale = false
        let resolved: URL?
        do {
            resolved = try URL(resolvingBookmarkData: data, options: [],
                               relativeTo: nil, bookmarkDataIsStale: &stale)
        } catch {
            debugLog("folder export: bookmark unresolvable: \(error.localizedDescription)")
            surface("The auto-save folder is no longer accessible. Pick it again in Settings › Auto-save to folder.")
            return nil
        }
        guard let url = resolved else {
            surface("The auto-save folder is no longer accessible. Pick it again in Settings › Auto-save to folder.")
            debugLog("folder export: bookmark stale or unresolvable")
            return nil
        }
        if stale {
            // iOS says the bookmark still resolves but was rebuilt (the user
            // re-created the folder, or the file provider remounted): refresh
            // the stored bookmark so the next write resolves cleanly.
            if let fresh = try? url.bookmarkData(options: []) {
                FolderSettings.bookmark = fresh
            }
        }
        heldFolder = url
        heldAccess = url.startAccessingSecurityScopedResource()
        return url
    }

    /// Write one session's Markdown into the chosen folder. Called at the end
    /// of a successful pipeline; every failure path ends in debugLog + one
    /// surfaced message and returns — the pipeline's result on the session is
    /// already saved and is never touched by this.
    func write(_ s: Session) {
        guard FolderSettings.hasFolder else { return }
        guard let folder = resolveFolder() else { stopAccess(); return }
        defer { stopAccess() }
        do {
            let stem = FolderExportFormat.stem(s)
            let mine = (UserDefaults.standard.dictionary(forKey: FolderSettings.filesKey) as? [String: String]) ?? [:]
            let old = FolderSettings.fileName(for: s.id)
            let name = FolderExportFormat.fileName(stem) { candidate in
                if candidate == old { return false }
                if mine.values.contains(candidate) { return true }
                return FileManager.default.fileExists(atPath: folder.appendingPathComponent(candidate).path)
            }
            let url = folder.appendingPathComponent(name)
            let text = FolderExportFormat.document(s)
            try Data(text.utf8).write(to: url, options: .atomic)
            FolderSettings.setFileName(name, for: s.id)
            // A rename moved the file: the old one goes, only if it is the one
            // Cicada wrote (the index says so).
            if let old, old != name {
                try? FileManager.default.removeItem(at: folder.appendingPathComponent(old))
            }
            debugLog("folder export: wrote \(name)")
        } catch {
            debugLog("folder export: write failed: \(error.localizedDescription)")
            surface("Could not save the Markdown to the auto-save folder: \(error.localizedDescription)")
        }
    }

    /// One alert per problem text, not one per pipeline run: the message goes
    /// through the app's own alert channel, which the caller has already
    /// moved on past (the pipeline is done by the time this runs).
    private func surface(_ text: String) {
        guard text != lastSurfaced else { return }
        lastSurfaced = text
        onSurface?(text)
    }

    /// Set by AppModel at init: hands the message to the app alert.
    var onSurface: ((String) -> Void)?
}
