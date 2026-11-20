import SwiftUI

/// The scan helper's live hints over the camera: warnings (move slower, light) and the
/// angle checklist chip for the hard spot in view
struct ScanHelperHUD: View {
  @ObservedObject var helper: ScanHelper

  static let purple = Color(red: 0.72, green: 0.33, blue: 0.95)

  var body: some View {
    VStack(spacing: 6) {
      ForEach(helper.warnings, id: \.self) { w in
        Label(w.text, systemImage: w.icon)
          .font(.caption.weight(.semibold))
          .multilineTextAlignment(.center)
          .foregroundStyle(.white)
          .padding(.horizontal, 10)
          .padding(.vertical, 6)
          .background(.orange.opacity(0.9), in: Capsule())
      }
      if let chip = helper.chip {
        VStack(alignment: .leading, spacing: 2) {
          Label(chip.title, systemImage: "cube.transparent.fill")
            .font(.caption.weight(.semibold))
            .foregroundStyle(Self.purple)
          Text(chip.detail).font(.caption)
        }
        .padding(8)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .combine)
      }
    }
    .animation(.easeInOut(duration: 0.2), value: helper.chip)
    .animation(.easeInOut(duration: 0.2), value: helper.warnings)
  }
}
