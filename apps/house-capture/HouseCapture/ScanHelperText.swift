import Foundation

/// Every UI string of the scan helper, in one place so it can be translated (Danish next).
enum ScanHelperText {
  // warnings
  static let moveSlower = "Move slower"
  static let tooDark = "Too dark: turn on more lights"
  static let startDark = "Turn on the lights for the best result"
  static let glare = "Bright window/lamp in view: film at an angle or close the curtain"

  // hard spot steps
  static let fromAbove = "Hold the phone above it and look down"
  static let low = "Crouch down and film low"
  static let closeUp = "Move within 1 m and hold still for a close-up"
  static let holdStill = "Hold still for a close-up"
  static let stepCloser = "Step closer to it"
  static let pointAtIt = "Point the camera straight at it"
  static let farSide = "Walk around to the far side"
  static let rightSide = "Step to the right side"
  static let leftSide = "Step to the left side"
  static let walkToSpot = "Walk to the purple spot"
  static func hardSpot(_ action: String) -> String { "Purple spot: " + action }

  // finishing
  static func almostDone(spots: Int) -> String {
    spots == 0 ? "Almost done: some areas need more photos"
      : spots == 1 ? "Almost done: 1 spot needs a closer look"
      : "Almost done: \(spots) spots need a closer look"
  }
  static let almostDoneDetail = "Purple spots are hard for the scanner. The app shows you where to go and takes the photos itself."
  static let showMe = "Show me"
  static let finishAnyway = "Finish anyway"
  static let allDone = "All done"
  static let legendHardSpot = "hard spot"
}
