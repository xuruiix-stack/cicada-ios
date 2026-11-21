import UIKit
import UniformTypeIdentifiers

/// UIKit's document picker works reliably in the supported iOS 16.4 setup.
/// Access is copied into the app sandbox as soon as the user picks a file.
final class MediaPickerPresenter: NSObject, UIDocumentPickerDelegate {
    static let shared = MediaPickerPresenter()
    private var onPick: ((URL) -> Void)?

    func present(onPick: @escaping (URL) -> Void) {
        self.onPick = onPick
        let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.audio, .movie], asCopy: false)
        picker.delegate = self
        picker.allowsMultipleSelection = false
        topViewController()?.present(picker, animated: true)
    }

    private func topViewController() -> UIViewController? {
        let scene = UIApplication.shared.connectedScenes
            .compactMap { $0 as? UIWindowScene }
            .first { $0.activationState == .foregroundActive }
            ?? UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.first
        var top = scene?.windows.first { $0.isKeyWindow }?.rootViewController
        while let presented = top?.presentedViewController { top = presented }
        return top
    }

    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        if let url = urls.first { onPick?(url) }
        onPick = nil
    }

    func documentPickerWasCancelled(_ controller: UIDocumentPickerViewController) { onPick = nil }
}
