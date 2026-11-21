#!/usr/bin/env python3
"""Compile and exercise Cicada's pure push-to-talk policy without an Apple SDK."""
import argparse
from pathlib import Path
import subprocess
import tempfile


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--swiftc", default="swiftc")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
source = (root / "Cicada" / "FaceToFace.swift").read_text(encoding="utf-8")
policy = source[source.index("enum FaceToFacePolicy {"):source.index("\nstruct FaceTurn:")]
checks = r'''
var count = 0
func check(_ condition: @autoclosure () -> Bool, _ name: String) {
    guard condition() else { fatalError("FAIL: \(name)") }
    count += 1
}
func accepts(_ seconds: Double, _ rms: Double, _ signalSeconds: Double) -> Bool {
    FaceToFacePolicy.acceptsAudio(.init(seconds: seconds, rms: rms, signalSeconds: signalSeconds))
}
check(accepts(0.5, FaceToFacePolicy.minimumRMS, 0.12), "inclusive audio thresholds")
check(!accepts(0.499, 0.01, 0.2), "short press")
check(!accepts(1, 0, 0), "silence")
check(!accepts(1, 0.02, 0.05), "isolated loud impulse")
check(!accepts(.nan, 0.02, 0.3), "invalid duration")
check(!accepts(1, .infinity, 0.3), "invalid RMS")
check(!accepts(1, 0.02, .nan), "invalid signal time")
check(FaceToFacePolicy.allowsDirection(source: "auto", target: "en"), "detect source")
check(FaceToFacePolicy.allowsDirection(source: "zh", target: "ja"), "explicit direction")
check(!FaceToFacePolicy.allowsDirection(source: "en", target: "en"), "same language")
check(!FaceToFacePolicy.allowsDirection(source: "ja", target: "auto"), "auto target")
check(!FaceToFacePolicy.allowsDirection(source: "invalid", target: "en"), "unsupported source")
for invalid in ["\n ", "…?!", "[BLANK_AUDIO]", "(No speech detected.)", "未检测到语音",
                "音声が検出されませんでした", "음성이 감지되지 않았습니다", "Error: timeout",
                "{\"error\":\"bad request\"}", String(repeating: "a", count: 4001)] {
    check(!FaceToFacePolicy.containsSpeech(invalid), "invalid transcript: \(invalid.prefix(50))")
}
check(FaceToFacePolicy.containsSpeech("明日、会いましょう。"), "non-Latin text")
check(FaceToFacePolicy.containsSpeech("There was no speech after the introduction."), "sentence containing sentinel words")
check(FaceToFacePolicy.containsSpeech(String(repeating: "a", count: 4000)), "inclusive text limit")
check(FaceToFacePolicy.usableTranslation("你好！", source: "Hello!"), "distinct translation")
check(!FaceToFacePolicy.usableTranslation("HELLO … world!", source: "Hello, world."), "case and punctuation echo")
check(!FaceToFacePolicy.usableTranslation("No translation available.", source: "Hello!"), "translation sentinel")
check(!FaceToFacePolicy.usableTranslation("你好！", source: "[BLANK_AUDIO]"), "invalid source")
print("PASS: \(count) face-to-face policy checks")
'''
with tempfile.TemporaryDirectory() as directory:
    swift = Path(directory) / "main.swift"
    executable = Path(directory) / "face-policy"
    swift.write_text(f"import Foundation\n{policy}\n{checks}", encoding="utf-8")
    subprocess.run([args.swiftc, "-swift-version", "5", str(swift), "-o", str(executable)], check=True)
    subprocess.run([str(executable)], check=True)
