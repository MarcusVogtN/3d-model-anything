import SwiftUI

/// One instruction at a time over the camera: the scan helper's hint if it has one
/// (too dark > move slower > bright window > hard spot), else the app's own coverage tips
struct ScanHelperHUD: View {
  @ObservedObject var helper: ScanHelper
  /// the capture screen's existing tips, shown when the helper has nothing to say
  var fallback: [String]

  static let purple = Color(red: 0.72, green: 0.33, blue: 0.95)

  var body: some View {
    Group {
      if let h = helper.hint {
        if h.warning {
          Label(h.text, systemImage: h.icon)
            .font(.caption.weight(.semibold))
            .multilineTextAlignment(.center)
            .foregroundStyle(.white)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(.orange.opacity(0.9), in: Capsule())
        } else {
          Label(h.text, systemImage: h.icon)
            .font(.caption.weight(.semibold))
            .multilineTextAlignment(.center)
            .padding(8)
            .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
            .overlay(RoundedRectangle(cornerRadius: 10).stroke(Self.purple, lineWidth: 1.5))
        }
      } else if let first = fallback.first {
        Text(first)
          .font(.caption)
          .multilineTextAlignment(.center)
          .padding(8)
          .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 10))
      }
    }
    .accessibilityElement(children: .combine)
    .animation(.easeInOut(duration: 0.2), value: helper.hint)
  }
}

/// "Show me": a pulsing ring on the spot, or a chevron at the screen edge pointing to it
struct ScanHelperGuide: View {
  @ObservedObject var helper: ScanHelper
  @State private var pulse = false

  var body: some View {
    GeometryReader { geo in
      if helper.guiding {
        if let p = helper.guidePoint {
          Circle()
            .stroke(ScanHelperHUD.purple, lineWidth: 4)
            .frame(width: 70, height: 70)
            .scaleEffect(pulse ? 1.25 : 0.85)
            .opacity(pulse ? 0.5 : 1)
            .position(p)
            .onAppear {
              withAnimation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true)) { pulse = true }
            }
        } else if let a = helper.guideAngle {
          let size = geo.size
          let rx = size.width / 2 - 44, ry = size.height / 2 - 140
          Image(systemName: "chevron.right.circle.fill")
            .font(.system(size: 52, weight: .bold))
            .symbolRenderingMode(.palette)
            .foregroundStyle(.white, ScanHelperHUD.purple)
            .rotationEffect(.radians(a))
            .position(x: size.width / 2 + rx * CGFloat(cos(a)), y: size.height / 2 + ry * CGFloat(sin(a)))
        }
      }
    }
    .allowsHitTesting(false)
    .ignoresSafeArea()
  }
}
