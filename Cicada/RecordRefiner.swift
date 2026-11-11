import Foundation
import NaturalLanguage

/// Lays the post-recording transcript over the recording's VAD segments: each line keeps
/// its segment's start time and voice and gets the part of the transcript said in it.
/// Ported from the upstream Mac app's Sources/Cicada/RecordRefiner.swift, which streams
/// the live recognizer's lines against the record recognizer's sentences as they arrive.
/// Here both texts exist in full once the recording stops, and one whole-transcript
/// alignment would be O(N²) in the meeting's length, so each segment's on-device caption
/// text is aligned against the transcript one bounded window at a time, left to right.
/// Includes upstream commit 4cb8512's boundary fix: a line the speaker ended a sentence
/// on was not cut, so snapping its end to the next sentence mark must not take the next
/// speaker's first word with it.
struct RecordRefiner {
    /// One segment's knowns: its caption text ("" when the recognizer gave none) and where
    /// its share of the speaking time puts it in the transcript, as a character range —
    /// provisional; the text alignment decides the real boundary.
    struct Segment {
        let text: String
        let provisional: Range<Int>
    }

    /// Where each segment's part of `record` ends (monotonic character offsets; the last
    /// segment ends at the record's end). A segment with caption text ends where that
    /// text ends in the record; one without ends at its provisional share.
    static func ends(segments: [Segment], record: [Character]) -> [Int] {
        var out: [Int] = []
        var cursor = 0
        for (i, seg) in segments.enumerated() {
            if i == segments.count - 1 { out.append(record.count); continue }
            let cap = seg.text.trimmingCharacters(in: .whitespacesAndNewlines)
            var e: Int
            if cap.count >= 3 {
                // The caption can't end far past its provisional range; the window bounds
                // the edit-distance work to a few hundred characters however long the meeting.
                let to = min(record.count, max(seg.provisional.upperBound + 40,
                                               cursor + cap.count * 3 + 120))
                let window = Array(record[cursor..<max(cursor, to)])
                e = cursor + (split(window, at: [cap], freeTail: true).first ?? 0)
            } else {
                e = min(record.count, seg.provisional.upperBound)
            }
            e = min(record.count, max(e, cursor))
            out.append(e)
            cursor = e
        }
        return out
    }

    /// Where in `record` each of `lines` ends: the record aligned with the lines' text joined, by
    /// edit distance. With `freeTail` the record may run on past the lines (a sentence that goes
    /// into a line not given yet). Verbatim from upstream, including commit 4cb8512's fix.
    static func split(_ record: [Character], at lines: [String], freeTail: Bool) -> [Int] {
        let live = Array(lines.joined())
        let n = live.count, m = record.count
        var bounds: [Int] = []
        var acc = 0
        for t in lines { acc += t.count; bounds.append(acc) }
        guard n > 0, m > 0 else { return bounds.map { _ in 0 } }
        // d[i][j]: cost of live[..<i] against record[..<j].
        var d = Array(repeating: Array(repeating: 0, count: m + 1), count: n + 1)
        for i in 0...n { d[i][0] = i }
        for j in 0...m { d[0][j] = j }
        for i in 1...n {
            for j in 1...m {
                let same = Self.same(live[i - 1], record[j - 1])
                d[i][j] = min(d[i - 1][j - 1] + (same ? 0 : 1), d[i - 1][j] + 1, d[i][j - 1] + 1)
            }
        }
        var j = m
        if freeTail { j = (0...m).min { d[n][$0] < d[n][$1] || d[n][$0] == d[n][$1] && $0 > $1 }! }
        // Walk back, noting the record position at each live position (the furthest one).
        // `low`: the nearest one, before the record words inserted there.
        var at = Array(repeating: 0, count: n + 1)
        var low = at
        var i = n
        at[n] = j
        low[n] = j
        while i > 0 || j > 0 {
            if i > 0, j > 0, d[i][j] == d[i - 1][j - 1] + (Self.same(live[i - 1], record[j - 1]) ? 0 : 1) {
                i -= 1; j -= 1
            } else if i > 0, d[i][j] == d[i - 1][j] + 1 {
                i -= 1
            } else {
                j -= 1
                low[i] = j
                continue
            }
            if at[i] == 0 { at[i] = j }
            low[i] = j
        }
        var ends = bounds.map { b in at[b] }
        // The live lines were often cut mid-sentence for the subtitles: a line ends at the record's
        // nearest punctuation when there is some a few characters away.
        // Otherwise it ends between words: Chinese comes one character per run, and "审 / 计" is no line end.
        let words = Self.wordStarts(record)
        let inside = Self.insideConnectives(record)
        // A line the speaker ended a sentence on was not cut: its end is where the next voice may
        // start, and the record often drops the "." ("lock that in" + " Okay,"): snapping there
        // took the next speaker's word. It only takes the mark that ends its last word. Words the
        // live recognizer lost between the two ("Friday." + " Great." + "Let's…") cost the same on
        // either side: its "." is the record's first one there, and they start the next line.
        let mark: (Character) -> Bool = { "。！？.!?".contains($0) }
        for k in ends.indices {
            let closed = lines[k].last.map(mark) ?? false
            var e = ends[k]
            if !closed { e = Self.snap(e, in: record) }
            else if bounds[k] > 0, case let p = low[bounds[k] - 1], p < e - 1, mark(record[p]) { e = p + 1 }
            if e == ends[k], e > 0, e < m, !words.contains(e) {
                let after = (e...m).first { words.contains($0) || $0 == m } ?? m
                let before = (0...e).last { words.contains($0) } ?? 0
                e = after - e <= e - before ? after : before
            }
            if closed, e > 0, e < m, mark(record[e]) { e += 1 }
            // "所 / 以": the tokenizer splits some connectives; the line ends before the whole word.
            if let start = inside[e] { e = start }
            ends[k] = max(e, k > 0 ? ends[k - 1] : 0)
        }
        return ends
    }

    /// Character offsets where a word starts (and the end).
    static func wordStarts(_ record: [Character]) -> Set<Int> {
        let text = String(record)
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        var starts: Set<Int> = [0, record.count]
        tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { range, _ in
            starts.insert(text.distance(from: text.startIndex, to: range.lowerBound))
            starts.insert(text.distance(from: text.startIndex, to: range.upperBound))
            return true
        }
        return starts
    }

    /// Chinese words that start a clause (upstream HeardText.connectives): the transcript's
    /// tokenizer may split them, and no line ends inside one ("所" "以").
    static let connectives: Set<String> = ["所以", "但是", "因为", "然后", "可是", "而且", "不过", "其实", "反正",
                                           "如果", "虽然", "或者", "另外", "接下来", "总之", "因此", "于是", "就是说"]

    /// Offsets inside a Chinese connective, each with where it starts: no line ends there, even
    /// where the tokenizer splits the word ("所" "以").
    static func insideConnectives(_ chars: [Character]) -> [Int: Int] {
        var out: [Int: Int] = [:]
        for c in connectives {
            let w = Array(c)
            guard chars.count >= w.count else { continue }
            for p in 0...(chars.count - w.count) where Array(chars[p..<(p + w.count)]) == w {
                for i in (p + 1)..<(p + w.count) { out[i] = p }
            }
        }
        return out
    }

    static let snapReach = 6

    /// `e`, or just after the closest sentence mark within `snapReach` characters.
    static func snap(_ e: Int, in record: [Character]) -> Int {
        func mark(_ c: Character) -> Bool { "，。？！；、,.?!;".contains(c) }
        guard e > 0, e < record.count, !mark(record[e - 1]) else { return e }
        for d in 0..<snapReach {
            if e + d < record.count, mark(record[e + d]) { return e + d + 1 }
            if e - 1 - d > 0, mark(record[e - 1 - d]) { return e - d }
        }
        return e
    }

    private static func same(_ a: Character, _ b: Character) -> Bool {
        a == b || a.lowercased() == b.lowercased()
    }
}
