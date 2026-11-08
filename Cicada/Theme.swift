import SwiftUI

/// Cicada palette, from the original app's Theme: grey ground, white cards, orange accent.
enum Ink {
    static let ground = Color(red: 0xF3/255, green: 0xF4/255, blue: 0xF6/255)
    static let ink = Color(red: 0x15/255, green: 0x17/255, blue: 0x1C/255)
    static let accent = Color(red: 0xE8/255, green: 0x55/255, blue: 0x2B/255)
    static let accentText = Color(red: 0xC8/255, green: 0x43/255, blue: 0x1D/255)
    static let secondary = Color(red: 0x5A/255, green: 0x5F/255, blue: 0x6B/255)
    static let tertiary = Color(red: 0x6B/255, green: 0x70/255, blue: 0x80/255)
    static let line = Color(red: 0xE2/255, green: 0xE4/255, blue: 0xE9/255)
    static let accentWash = Color(red: 0xE8/255, green: 0x55/255, blue: 0x2B/255).opacity(0.12)
    static let ok = Color(red: 0x2F/255, green: 0x9E/255, blue: 0x5B/255)
    static let link = Color(red: 0.70, green: 0.23, blue: 0.08)
}

enum SpeakerInk {
    private static let pairs: [(ink: UInt32, tint: UInt32)] = [
        (0x1F4E8C, 0xDCE7F7), (0x8A3A16, 0xF6E2D6), (0x1E6B45, 0xDDEFE4),
        (0x5B3A8C, 0xE9E1F5), (0x3F434C, 0xE7E9EE),
    ]
    static func ink(_ n: Int) -> Color { Color(hex: pairs[(max(1,n)-1) % pairs.count].ink) }
    static func tint(_ n: Int) -> Color { Color(hex: pairs[(max(1,n)-1) % pairs.count].tint) }
}

extension Color {
    init(hex: UInt32) {
        self.init(red: Double(hex >> 16 & 0xFF)/255, green: Double(hex >> 8 & 0xFF)/255, blue: Double(hex & 0xFF)/255)
    }
}
