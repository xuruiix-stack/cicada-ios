#!/usr/bin/env python3
"""Run extracted production Swift logic locally; no iOS SDK or network required.

SWIFT_FRONTEND=/absolute/path/to/swift-frontend python3 Tests/run-text-regressions.py
Storage/HTTP boundaries are test doubles; this is not an iOS typecheck or UI test.
"""
from pathlib import Path
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = os.environ.get("SWIFT_FRONTEND", "swift-frontend")
AI = (ROOT / "Cicada/AIClient.swift").read_text()
FOLDER = (ROOT / "Cicada/FolderExport.swift").read_text()
MODELS = (ROOT / "Cicada/Models.swift").read_text()
STORE = (ROOT / "Cicada/Store.swift").read_text()


def between(source, start, end):
    return source[source.index(start):source.index(end, source.index(start))]


PREFIX = r'''
import Foundation
let suiteName = "cicada-text-regressions-" + UUID().uuidString
let testDefaults = UserDefaults(suiteName: suiteName)!
defer { testDefaults.removePersistentDomain(forName: suiteName) }
'''


def run(name, source):
    with tempfile.TemporaryDirectory(prefix="cicada-text-regression-") as directory:
        path = Path(directory) / "main.swift"
        path.write_text(source)
        print(f"=== {name} ===", flush=True)
        subprocess.run([FRONTEND, "-interpret", str(path)], check=True)


def notes_tests():
    settings = between(AI, "enum AISettings {", "\nenum Keychain {")
    settings = settings.replace("UserDefaults.standard", "testDefaults")
    side = between(AI, "    enum Side {", "\n    private func endpoint")
    method = between(AI, "    func meetingNotes(", "\n    /// Settings › Check")
    models = between(MODELS, "struct ActionItem:", "\n/// Where a session stands")
    return PREFIX + r'''
enum Keychain {
 static var values: [String: String] = [:]
 static func string(for key: String) -> String? { values[key] }
}
enum AIError: Error { case unparsableNotes(host: String) }
''' + settings + models + "\nfinal class Harness {\n" + side + r'''
 var observed: (String, String, String?)?
 func chat(system: String, user: String, json: Bool, on side: Side) async throws -> String {
  observed = (side.baseURL, side.chatModel, side.key)
  return "{\"summary\":\"ok\"}"
 }
 func extractNotesJSON(from raw: String) -> [String: Any]? {
  try? JSONSerialization.jsonObject(with: Data(raw.utf8)) as? [String: Any]
 }
''' + method + r'''
}
func verify(_ label: String, url: String, model: String, notesKey: String?, expectedURL: String, expectedModel: String, expectedKey: String?) async throws {
 AISettings.baseURL = "https://api.ant-ling.com/v1"
 AISettings.chatModel = "Ling-2.5-1T"
 AISettings.notesBaseURL = url
 AISettings.notesChatModel = model
 Keychain.values = [AISettings.keyKey: "main-secret"]
 Keychain.values[AISettings.notesKeyKey] = notesKey
 let h = Harness()
 let n = try await h.meetingNotes(transcript: "test")
 guard let o = h.observed else { fatalError("no call") }
 precondition(o.0 == expectedURL && o.1 == expectedModel && o.2 == expectedKey, label)
 precondition(n.summary == "ok")
 precondition(Harness.Side.main.baseURL == AISettings.baseURL && Harness.Side.main.chatModel == AISettings.chatModel && Harness.Side.main.key == "main-secret", "main side changed")
 print("PASS " + label)
}
let mainURL = "https://api.ant-ling.com/v1"
try await verify("default Ling configuration unchanged", url: "", model: "", notesKey: nil, expectedURL: mainURL, expectedModel: "Ling-2.5-1T", expectedKey: "main-secret")
try await verify("model-only override", url: "", model: "notes-large", notesKey: nil, expectedURL: mainURL, expectedModel: "notes-large", expectedKey: "main-secret")
try await verify("model-only ignores stale notes key", url: "", model: "notes-large", notesKey: "stale-secret", expectedURL: mainURL, expectedModel: "notes-large", expectedKey: "main-secret")
try await verify("separate URL model key", url: "https://notes.example/v1", model: "notes-large", notesKey: "notes-secret", expectedURL: "https://notes.example/v1", expectedModel: "notes-large", expectedKey: "notes-secret")
try await verify("separate URL falls back to main key", url: "https://notes.example/v1", model: "notes-large", notesKey: nil, expectedURL: "https://notes.example/v1", expectedModel: "notes-large", expectedKey: "main-secret")
try await verify("separate URL falls back to main model", url: "https://notes.example/v1", model: "", notesKey: "notes-secret", expectedURL: "https://notes.example/v1", expectedModel: "Ling-2.5-1T", expectedKey: "notes-secret")
try await verify("removed notes URL ignores stale key", url: "", model: "", notesKey: "stale-secret", expectedURL: mainURL, expectedModel: "Ling-2.5-1T", expectedKey: "main-secret")
Keychain.values[AISettings.keyKey] = nil
precondition(!Harness.Side.notes.isConfigured, "stale notes key must not configure main service")
print("PASS stale notes key cannot replace missing main key")
'''


def folder_tests():
    fmt = between(FOLDER, "enum FolderExportFormat {", "\n/// Settings for the auto-save folder")
    settings = between(FOLDER, "enum FolderSettings {", "\n/// Writes a finished session")
    methods = between(FOLDER, "    /// Retry only failed/pending writes", "    /// One alert per problem text")
    settings = settings.replace("UserDefaults.standard", "testDefaults")
    methods = methods.replace("UserDefaults.standard", "testDefaults")
    models = between(MODELS, "struct TranscriptLine:", "\nenum TranscriptExport {")
    tuning = between(STORE, "enum CicadaTuning {", "\n/// Local storage")
    substantive = STORE[STORE.index("    static func substantive("):STORE.rfind("\n}")]
    store_helpers = tuning + "\nfinal class SessionStore {\n" + substantive + "\n}\n"
    return PREFIX + "func debugLog(_ s: String) {}\n" + store_helpers + models + r'''
enum TranscriptExport {
 static func markdown(_ s: Session) -> String { s.lines.map(\.src).joined(separator: "\n") }
}
''' + fmt + settings + r'''
final class Harness {
 var folder: URL?
 var surfaces = 0
 func resolveFolder() -> URL? { folder }
 func stopAccess() {}
 func surface(_ s: String) { surfaces += 1 }
''' + methods + r'''
}
func session(_ id: String) -> Session {
 var s = Session(); s.id = id; s.title = id; s.endedAt = Date(); s.processingState = .done
 s.lines = [TranscriptLine(t: 0, src: "hello " + id)]
 return s
}
let base = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
try FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
defer { try? FileManager.default.removeItem(at: base) }
let h = Harness()
let a = session("a"), b = session("b"), historical = session("historical")
h.folder = base
h.write(a)
precondition(FolderSettings.pendingIDs.isEmpty && FolderSettings.fileName(for: a.id) == nil)
print("PASS disabled folder does not queue/export")
FolderSettings.bookmark = Data([1])
h.folder = base.appendingPathComponent("missing/leaf")
h.write(a)
precondition(FolderSettings.pendingIDs == [a.id] && FolderSettings.fileName(for: a.id) == nil)
print("PASS write failure persists only record ID")
h.folder = nil
h.write(b)
precondition(FolderSettings.pendingIDs == [a.id,b.id])
print("PASS unresolved folder preserves pending")
let unrelated = FolderExportFormat.stem(a) + ".md"
try Data("user file".utf8).write(to: base.appendingPathComponent(unrelated))
h.folder = base
h.retryPending(sessions: [a,b,historical])
precondition(FolderSettings.pendingIDs.isEmpty)
precondition(FolderSettings.fileName(for: a.id) == FolderExportFormat.stem(a) + " 2.md")
precondition(FolderSettings.fileName(for: b.id) != nil && FolderSettings.fileName(for: historical.id) == nil)
let preserved = try String(contentsOf: base.appendingPathComponent(unrelated), encoding: .utf8)
precondition(preserved == "user file")
print("PASS recovery exports only pending, clears queue, preserves foreign file")
FolderSettings.setPending(true, for: "deleted")
h.retryPending(sessions: [a,b])
precondition(FolderSettings.pendingIDs.isEmpty)
print("PASS deleted session dequeued")
var running = session("running"); running.processingState = .running
var live = session("live"); live.endedAt = nil
var blank = session("blank"); blank.lines[0].src = "   "
for s in [running,live,blank] { FolderSettings.setPending(true, for: s.id) }
h.retryPending(sessions: [running,live,blank])
precondition(FolderSettings.pendingIDs == [running.id,live.id,blank.id])
precondition(FolderSettings.fileName(for: running.id) == nil)
print("PASS live/running/blank pending records are not exported early")
FolderSettings.resetFiles()
precondition(FolderSettings.pendingIDs.isEmpty && FolderSettings.fileName(for: a.id) == nil)
print("PASS folder reset clears pending and successful index")
FolderSettings.setPending(true, for: "same")
FolderSettings.setPending(true, for: "same")
precondition(FolderSettings.pendingIDs == ["same"])
print("PASS repeated failure deduplicates record IDs")
FolderSettings.resetFiles()
var delayed = session("delayed"); delayed.lines = []
h.markPending(delayed)
h.retryPending(sessions: [delayed])
precondition(FolderSettings.pendingIDs == [delayed.id] && FolderSettings.fileName(for: delayed.id) == nil)
delayed.lines = [TranscriptLine(t: 0, src: "text arrived later")]
h.retryPending(sessions: [delayed])
precondition(FolderSettings.pendingIDs.isEmpty && FolderSettings.fileName(for: delayed.id) != nil)
print("PASS mark before text preserves job until later usable text")
h.markPending(live)
h.write(live)
precondition(FolderSettings.pendingIDs.isEmpty && FolderSettings.fileName(for: live.id) == nil)
print("PASS unfinished session cannot queue or write")
FolderSettings.folderPath = base.path
FolderSettings.setPending(true, for: a.id)
FolderSettings.setFileName("owned.md", for: a.id)
FolderSettings.choseFolder(base.appendingPathComponent("unused/.."), bookmark: Data([2]))
precondition(FolderSettings.pendingIDs == [a.id] && FolderSettings.fileName(for: a.id) == "owned.md")
precondition(FolderSettings.bookmark == Data([2]))
print("PASS same normalized folder reauthorization preserves jobs and ownership")
FolderSettings.choseFolder(base.appendingPathComponent("different"), bookmark: Data([3]))
precondition(FolderSettings.pendingIDs.isEmpty && FolderSettings.fileName(for: a.id) == nil)
precondition(FolderSettings.bookmark == Data([3]))
print("PASS different folder clears old jobs and ownership")
FolderSettings.bookmark = nil
h.markPending(a)
precondition(FolderSettings.pendingIDs.isEmpty)
print("PASS disabled folder ignores early pending marks")
'''


if __name__ == "__main__":
    run("notes routing: 8 cases", notes_tests())
    run("folder pending: 13 cases", folder_tests())
    print("PASS all 21 text regressions")
