import SwiftUI
import AVFoundation

// MARK: - App model wiring record / transcribe / translate / notes

@MainActor
final class AppModel: ObservableObject {
    @Published var store = SessionStore()
    @Published var ai = AIClient()
    @Published var recorder = Recorder()
    @Published var tab: Tab = .records
    @Published var selection: String?
    @Published var busy = false
    @Published var message: String?
    enum Tab { case records, settings }

    var selected: Session? { store.sessions.first { $0.id == selection } }

    func startRecording() async {
        guard await recorder.requestPermissions() else { message = "Microphone / speech permission denied."; return }
        var s = Session(); s.startedAt = Date()
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
            .appendingPathComponent("Cicada/Recordings/\(s.id).m4a")
        s.audioFile = url.lastPathComponent
        do { try recorder.start(audioURL: url); store.save(s); selection = s.id }
        catch { message = error.localizedDescription }
    }
    func stopRecording() {
        _ = recorder.stop()
        if var s = selected { s.endedAt = Date(); store.save(s) }
    }

    /// AI pipeline for a finished recording: transcribe → translate → notes.
    func process(_ s: Session) async {
        busy = true; defer { busy = false }
        guard AISettings.isConfigured else { message = "Set the API base URL and key in Settings › AI Service."; return }
        do {
            var s = s
            if let f = s.audioFile, let url = store.audioURL(for: s) {
                let text = try await ai.transcribe(audioURL: url)
                if !text.isEmpty {
                    s.lines = [TranscriptLine(t: 0, speaker: 1, src: text)]
                }
            }
            if !s.lines.isEmpty && AISettings.targetLang != "auto" {
                for i in s.lines.indices {
                    s.lines[i].dst = try await ai.translate(s.lines[i].src, to: AISettings.targetLang)
                }
            }
            let transcript = s.lines.map { "Speaker \($0.speaker): \($0.src)" }.joined(separator: "\n")
            if !transcript.isEmpty { s.notes = try await ai.meetingNotes(transcript: transcript) }
            store.save(s)
        } catch { message = error.localizedDescription }
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
                        Text(model.recorder.isRecording ? "Recording… \(TranscriptExport.clock(model.recorder.elapsed))" : "Record, transcribe, translate and write notes")
                            .font(.footnote).foregroundStyle(Ink.secondary)
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
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 0) {
                Text(TranscriptExport.displayTitle(session)).font(.title.bold())
                Text("\(Int(session.duration/60)) min · \(session.lines.count) lines")
                    .font(.footnote).foregroundStyle(Ink.secondary).padding(.top, 4)
                Picker("Page", selection: $page) {
                    Text("Transcript").tag(0); Text("Notes").tag(1)
                }.pickerStyle(.segmented).padding(.top, 16)
                if page == 0 {
                    ForEach(session.lines) { l in TranscriptRow(line: l) }
                    if session.lines.isEmpty { Text("No transcript yet.").foregroundStyle(Ink.secondary).padding(.top, 24) }
                } else {
                    NotesView(notes: session.notes)
                }
            }
            .padding(.horizontal, 20).padding(.top, 8)
        }
        .navigationTitle("").navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItemGroup(placement: .topBarTrailing) {
                Button { Task { await model.process(session) } } label: {
                    model.busy ? AnyView(ProgressView()) : AnyView(Image(systemName: "arrow.triangle.2.circlepath"))
                }.disabled(model.busy)
                ShareLink(item: model.exportMarkdown(session), subject: Text(TranscriptExport.displayTitle(session))) {
                    Image(systemName: "square.and.arrow.up")
                }
            }
        }
    }
}

struct TranscriptRow: View {
    let line: TranscriptLine
    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Text("S\(line.speaker)").font(.caption.bold())
                .foregroundStyle(SpeakerInk.ink(line.speaker))
                .padding(.horizontal, 8).padding(.vertical, 4)
                .background(SpeakerInk.tint(line.speaker), in: Capsule())
            VStack(alignment: .leading, spacing: 2) {
                Text(line.src).font(.body)
                if let d = line.dst, d != line.src { Text(d).font(.body).foregroundStyle(Ink.secondary) }
                Text(TranscriptExport.clock(line.t)).font(.caption).foregroundStyle(Ink.tertiary)
            }
        }.padding(.vertical, 6)
    }
}

struct NotesView: View {
    let notes: MeetingNotes?
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
        } else { Text("No notes yet. Tap ▶ to transcribe and write notes.").foregroundStyle(Ink.secondary).padding(.top, 24) }
    }
}

// MARK: - Settings

struct SettingsView: View {
    @EnvironmentObject var model: AppModel
    @State private var baseURL = AISettings.baseURL
    @State private var keyDraft = ""
    @State private var chatModel = AISettings.chatModel
    @State private var transcribeModel = AISettings.transcribeModel
    @State private var targetLang = AISettings.targetLang
    var body: some View {
        Form {
            Section("AI Service (OpenAI-compatible)") {
                TextField("Base URL", text: $baseURL, prompt: Text("https://api.example.com/v1"))
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                    .onChange(of: baseURL, perform: { _ in AISettings.baseURL = baseURL })
                SecureField("API key", text: $keyDraft, prompt: Text("sk-…"))
                    .textInputAutocapitalization(.never).autocorrectionDisabled()
                    .onSubmit(saveKey)
                if AISettings.hasKey { Button("Save key") { saveKey() } }
                HStack {
                    Circle().fill(AISettings.isConfigured ? Ink.ok : Color.gray).frame(width: 8, height: 8)
                    Text(AISettings.isConfigured ? "Connected" : "Not set up").foregroundStyle(Ink.secondary)
                    Spacer()
                    Button("Check") { Task { _ = await model.ai.checkKey() } }
                }
                TextField("Chat model", text: $chatModel).onChange(of: chatModel, perform: { _ in AISettings.chatModel = chatModel })
                TextField("Transcription model", text: $transcribeModel).onChange(of: transcribeModel, perform: { _ in AISettings.transcribeModel = transcribeModel })
                Picker("Translate to", selection: $targetLang) {
                    Text("English").tag("en"); Text("Chinese").tag("zh"); Text("Japanese").tag("ja"); Text("Korean").tag("ko")
                }.onChange(of: targetLang, perform: { _ in AISettings.targetLang = targetLang })
                if AISettings.hasKey { Button("Remove key", role: .destructive) { Keychain.remove(for: AISettings.keyKey); keyDraft = "" } }
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
    private func saveKey() {
        let k = keyDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !k.isEmpty else { return }
        Keychain.set(k, for: AISettings.keyKey); keyDraft = ""
        Task { _ = await model.ai.checkKey() }
    }
}
