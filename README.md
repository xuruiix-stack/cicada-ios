# Cicada (iOS) — native Swift / SwiftUI

Native iOS app, iOS 16.4+, arm64. Xcode project: open `Cicada.xcodeproj` on a Mac with Xcode.

Features:
- Record audio from the microphone (AVAudioEngine → AAC `.m4a`)
- On-device live captions (SFSpeechRecognizer) + cloud speech-to-text
- Translation, AI meeting notes (structured summary, topics, conclusions, action items)
- Speaker separation (1-based numbers in first-appearance order, shown at ≥5 substantive lines)
- Local storage of recordings and notes (`Documents/Cicada/`)
- Export as Markdown (Share sheet)

## AI configuration (no hardcoded keys)
Settings › AI Service: enter an OpenAI-compatible **Base URL** and **API key** (stored in Keychain),
plus chat / transcription model names. Endpoints used: `POST {base}/chat/completions`,
`POST {base}/audio/transcriptions`.

## Linux preview
This sandbox is Linux (no Swift/Xcode). `Preview/` is a Rust/egui mirror of the same screens,
built to verify layout via `--screenshot`. It is not the iOS deliverable.
