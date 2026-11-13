import AVFoundation
import SwiftUI
import Foundation

/// Plays a finished session's recording (the official iOS app's RecordPlayer
/// is the model): AVAudioPlayer over the .m4a in Documents/Cicada/Recordings,
/// a 0.25 s timer drives the playhead the transcript highlights, and the
/// transcript's line times seek. Owned by the record detail screen; its life
/// is the screen's — stop() runs on leaving, so no player outlives its page.
@MainActor
final class RecordPlayer: ObservableObject {
    @Published private(set) var now: Double = 0
    @Published private(set) var playing = false
    private(set) var duration: Double = 0
    private var audio: AVAudioPlayer?
    private var timer: Timer?

    var available: Bool { audio != nil }

    private var loaded: String?

    /// Set by the owning screen: true when the recorder is NOT running. The
    /// player and the recorder share one AVAudioSession; while the recorder
    /// owns it (.playAndRecord) the player must not switch the category —
    /// that would drop the microphone mid-recording. Playback still works
    /// over the recorder's session (it carries .defaultToSpeaker), just
    /// without touching it.
    var mayOwnSession: () -> Bool = { true }

    /// The session's recording; another session's replaces it (the detail page
    /// is reused when another record is picked).
    func load(store: SessionStore, session: Session) {
        guard loaded != session.id else { return }
        stop()
        loaded = session.id
        audio = nil
        now = 0
        duration = 0
        objectWillChange.send()
        guard let url = store.audioURL(for: session),
              FileManager.default.fileExists(atPath: url.path) else { return }
        guard let a = try? AVAudioPlayer(contentsOf: url) else { return }
        audio = a
        duration = a.duration
        objectWillChange.send()
    }

    func toggle() {
        guard let audio else { return }
        if audio.isPlaying { audio.pause(); playing = false; timer?.invalidate(); return }
        // Playback, not playAndRecord — but only when the recorder isn't
        // mid-recording (see mayOwnSession). Failing to activate is
        // best-effort: the player still plays, silently, which is the honest
        // outcome of a broken session rather than a crashed screen.
        if mayOwnSession() {
            try? AVAudioSession.sharedInstance().setCategory(.playback)
        }
        try? AVAudioSession.sharedInstance().setActive(true)
        audio.play(); playing = true
        // The playhead the transcript highlights. The timer fires on the main
        // run loop; hopping through a MainActor Task keeps the @MainActor
        // isolation honest on iOS 16 (no MainActor.assumeIsolated there).
        timer = Timer.scheduledTimer(withTimeInterval: 0.25, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }

    private func tick() {
        guard let a = audio else { timer?.invalidate(); return }
        now = a.currentTime
        if !a.isPlaying { playing = false; timer?.invalidate() }
    }

    /// Tapping a transcript line starts playback from its timestamp (the
    /// official app's seek; a paused player starts too).
    func seek(_ t: Double) {
        guard let audio else { return }
        audio.currentTime = min(max(0, t), duration)
        now = audio.currentTime
        if !audio.isPlaying { toggle() }
    }

    func stop() { audio?.stop(); playing = false; timer?.invalidate() }

    /// The last line that started at or before the playhead, once playing has
    /// begun (the official app's rule, +0.2 s tolerance).
    func lineID(in s: Session) -> String? {
        guard playing || now > 0 else { return nil }
        return s.lines.last(where: { $0.t <= now + 0.2 })?.id
    }
}

/// The record detail screen's player bar, modeled on the official app's
/// PlayerBar: play / pause, a progress line, the elapsed and total time.
struct PlayerBar: View {
    @ObservedObject var player: RecordPlayer

    var body: some View {
        HStack(spacing: 12) {
            Button(action: player.toggle) {
                Image(systemName: player.playing ? "pause.fill" : "play.fill")
                    .font(.system(size: 16, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 46, height: 46)
                    .background(Ink.ink, in: Circle())
            }
            .accessibilityLabel(player.playing ? Text("Pause") : Text("Play"))
            VStack(spacing: 6) {
                GeometryReader { g in
                    ZStack(alignment: .leading) {
                        Capsule().fill(Ink.line)
                        Capsule().fill(Ink.accent)
                            .frame(width: g.size.width * (player.duration > 0 ? min(1, player.now / player.duration) : 0))
                    }
                }
                .frame(height: 4)
                HStack {
                    Text(TranscriptExport.clock(player.now))
                    Spacer()
                    Text(TranscriptExport.clock(player.duration))
                }
                .font(.caption2.monospacedDigit())
                .foregroundStyle(Ink.secondary)
            }
        }
        .padding(.leading, 8)
        .padding(.trailing, 18)
        .frame(height: 62)
        .padding(.horizontal, 12)
        .background(.thinMaterial, in: Capsule())
    }
}
