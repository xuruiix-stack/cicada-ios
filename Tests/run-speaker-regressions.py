#!/usr/bin/env python3
"""Run the actual file VAD/cancellation helpers without AVFoundation or Core ML.

Usage: python3 Tests/run-speaker-regressions.py
       SWIFT_FRONTEND=/path/to/swift-frontend python3 Tests/run-speaker-regressions.py

The Swift source is extracted from SpeakerID.swift and tuning from Store.swift;
no copied implementation or generated source file is kept in the repository.
These tests exercise pure gate state, sample-clock timing, bounded emitted audio
and the cancellation token only. They do NOT test AVAudioConverter, Core ML,
Task cancellation-handler delivery, UIApplication background expiry or real RSS.

Xcode/device checks still required:
* Decode 16k CAF and 44.1/48k stereo m4a; compare lengths/timestamps and EOF tail.
  Put different voices on left/right (including right-only speech) and confirm
  both survive the mono downmix; a mere channel remap must not pass this test.
* Missing/damaged input must throw, leaving the old transcript/notes untouched.
* Cancel before queue entry, during decode/inference and at background expiry;
  no partial skeleton may be committed and subsequent live embeddings must run.
* Two-hour audio: profile peak RSS; raw PCM must not grow with recording length.
  Final TranscriptLine metadata and voice centroids legitimately grow with it.
* Run two files and a live recording: no voice/session identity can cross jobs.
"""

import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def extract_type(source: str, declaration: str) -> str:
    start = source.index(declaration)
    opening = source.index("{", start)
    depth = 0
    # These two pure types have no braces in strings/comments. Fail closed if
    # their declarations disappear rather than silently testing a stale copy.
    for offset in range(opening, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[start:offset + 1]
    raise ValueError(f"Unclosed Swift declaration: {declaration}")


def frontend_path() -> str:
    configured = os.environ.get("SWIFT_FRONTEND")
    if configured:
        return configured
    installed = shutil.which("swift-frontend")
    if installed:
        return installed
    local = ROOT.parent / ".swift-toolchain/usr/bin/swift-frontend"
    if local.is_file():
        return str(local)
    if shutil.which("xcrun"):
        found = subprocess.run(["xcrun", "--find", "swift-frontend"],
                               text=True, capture_output=True)
        if found.returncode == 0 and found.stdout.strip():
            return found.stdout.strip()
    raise RuntimeError("swift-frontend not found; set SWIFT_FRONTEND to its absolute path")


def main() -> int:
    speaker = (ROOT / "Cicada/SpeakerID.swift").read_text()
    store = (ROOT / "Cicada/Store.swift").read_text()
    names = ("sampleRate", "captureChunk", "gateThreshold", "preRoll",
             "pauseEndsSegment", "maxSegmentLength")
    constants = []
    for name in names:
        match = re.search(rf"^\s*(static let {name}(?:\s*:\s*\w+)?\s*=\s*[^\n/]+)",
                          store, re.MULTILINE)
        if match is None:
            raise ValueError(f"Missing tuning constant: {name}")
        constants.append(match.group(1).strip())
    source = "import Foundation\nenum CicadaTuning {\n" + "\n".join(constants) + "\n}\n"
    source += extract_type(speaker, "private struct FileVoiceGate {") + "\n"
    source += extract_type(speaker, "private final class FileSpeakerCancellation:") + "\n"
    source += r'''
func run(_ parts: [(Double, Float)], batch: Int = 1023) -> [(Double, Int)] {
    var gate = FileVoiceGate()
    var result: [(Double, Int)] = []
    let emit = { (t: Double, samples: [Float]) in result.append((t, samples.count)) }
    for (seconds, value) in parts {
        let n = Int((seconds * CicadaTuning.sampleRate).rounded())
        var from = 0
        while from < n {
            let count = min(batch, n - from)
            Array(repeating: value, count: count).withUnsafeBufferPointer {
                gate.append($0, emit: emit)
            }
            from += count
        }
    }
    gate.finish(emit: emit)
    let finishedCount = result.count
    gate.finish(emit: emit)
    assert(result.count == finishedCount, "EOF flush must be idempotent")
    return result
}
let rate = CicadaTuning.sampleRate
let quiet = run([(30, 0)])
assert(quiet.isEmpty, "Silence must not create a segment")
assert(run([(2, CicadaTuning.gateThreshold)]).isEmpty,
       "The gate comparison is strict, not >= threshold")
let tail = run([(0.49, 0.2)])
assert(tail.count == 1 && tail[0].1 == 7840, "Unclosed short EOF tail must survive")
let joined = run([(2, 0.2), (0.5, 0), (2, 0.2)])
assert(joined.count == 1 && joined[0].1 == 64000,
       "A sub-threshold pause joins speech, without retaining its quiet audio")
let split = run([(2, 0.2), (0.6, 0), (2, 0.2)])
assert(split.count == 2 && split.map { $0.1 } == [32000, 32000])
assert(abs(split[1].0 - (2.6 - CicadaTuning.preRoll)) < 0.00001,
       "Segment timestamps must count decoded silence, not just voiced samples")
let bounded = run([(25.1, 0.2)])
let maximum = Int(rate * CicadaTuning.maxSegmentLength)
assert(bounded.count == 4 && bounded.allSatisfy { $0.1 <= maximum })
assert(bounded.reduce(0) { $0 + $1.1 } == 401600, "Max-length closure must not lose samples")
let unaligned = run([(2, 0.2), (0.6, 0), (2, 0.2)], batch: 37)
assert(split.count == unaligned.count)
assert(zip(split, unaligned).allSatisfy { $0.0 == $1.0 && $0.1 == $1.1 },
       "Codec/read chunk partitioning must not change gate output")
let late = run([(5, 0), (2, 0.2)])
assert(late.count == 1 && abs(late[0].0 - (5 - CicadaTuning.preRoll)) < 0.00001)
let halfSecond = run([(0.5, 0.2)])
let twoSeconds = run([(2, 0.2)])
assert(halfSecond.first?.1 == 8000 && twoSeconds.first?.1 == 32000,
       "The gate must not discard embedding-threshold candidates itself")
let long = run([(600, 0.2)], batch: 4093)
assert(long.count == 75 && long.allSatisfy { $0.1 <= maximum })
assert(long.reduce(0) { $0 + $1.1 } == 9_600_000)
assert(run([(2, 0.2)]).first?.0 == 0, "A new gate must not inherit a prior clock")
private let cancellation = FileSpeakerCancellation()
try cancellation.check()
DispatchQueue.global().sync { cancellation.cancel() }
var wasCancelled = false
do { try cancellation.check() }
catch is CancellationError { wasCancelled = true }
assert(wasCancelled, "Cancellation must be visible to queue-confined decode work")
cancellation.cancel()
wasCancelled = false
do { try cancellation.check() }
catch is CancellationError { wasCancelled = true }
assert(wasCancelled, "Cancellation must remain set and be idempotent")
print("PASS: actual file VAD, EOF, threshold, pause merge/split, sample-clock, chunk invariance, 8-second bound, cancellation token")
print("NOT COVERED: AVAudioConverter/CoreML/iOS Task/background/RSS integration; see this test's docstring")
'''
    # Typecheck the ACTUAL actor/queue methods, with only Apple-only model/file
    # APIs stubbed. Parse alone misses illegal MainActor property access from a
    # DispatchQueue's @Sendable closure, even in Swift 5 language mode.
    tagger = extract_type(speaker, "@MainActor\nfinal class SpeakerTagger {")
    file_reader = extract_type(tagger, "    private nonisolated static func fileLines(")
    tagger = tagger.replace(file_reader, """    private nonisolated static func fileLines(
        _ file: AVAudioFile, model: Voiceprints,
        cancellation: FileSpeakerCancellation) throws -> [TranscriptLine] {
        try cancellation.check()
        return []
    }""", 1)
    actor_source = """import Foundation
final class Voiceprints: @unchecked Sendable {
    static func load() throws -> Voiceprints { Voiceprints() }
    func embed(_ samples: [Float]) -> [Float]? { samples }
}
final class AVAudioFile { init(forReading url: URL) throws {} }
struct TranscriptLine {}
func debugLog(_ message: String) {}
"""
    actor_source += extract_type(speaker, "private final class FileSpeakerCancellation:") + "\n"
    actor_source += extract_type(speaker, "private final class SpeakerWorkerState:") + "\n"
    actor_source += tagger + "\n"
    frontend = frontend_path()
    env = os.environ.copy()
    deps = ROOT.parent / ".swift-toolchain/deps/usr/lib/x86_64-linux-gnu"
    if deps.is_dir():
        env["LD_LIBRARY_PATH"] = str(deps) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    subprocess.run([frontend, "-typecheck", "-swift-version", "5", "-"],
                   input=actor_source, text=True, env=env, check=True)
    print("PASS: actual SpeakerTagger queue closures Swift 5 typecheck (Apple-only APIs stubbed)", flush=True)
    subprocess.run([frontend, "-interpret", "-swift-version", "5", "-"],
                   input=source, text=True, env=env, check=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        sys.exit(1)
