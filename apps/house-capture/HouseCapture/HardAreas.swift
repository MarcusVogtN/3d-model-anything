import Foundation
import simd

/// Pure logic of the live scan helper's "hard areas": places the LiDAR is unsure about
/// (thin bars, black or shiny things, glass, clutter) and which camera angles of each one
/// the saved photos already cover. No ARKit here, so -selftest can check it.
///
/// The thresholds mirror the PC check (Python) exactly; change both together.
enum HardAreas {
  // MARK: voxels
  /// edge of one voxel (m)
  static let voxelSize: Float = 0.1
  /// depth samples outside this range are ignored (m)
  static let minDepth: Float = 0.2
  static let maxDepth: Float = 3.5
  /// sample every 2nd depth pixel in x and y
  static let pixelStep = 2
  /// a voxel is hard when unsure / total > this ...
  static let unsureRatio: Float = 0.5
  /// ... and it has more than this many samples
  static let minSamples: Int32 = 40
  /// voxels this close to a RoomPlan wall are ignored (m)
  static let wallMargin: Float = 0.12
  /// voxels lower than floor + this are ignored (m)
  static let floorMargin: Float = 0.06
  /// clusters (26-neighbour connected) need at least this many voxels
  static let minClusterVoxels = 15

  // MARK: angle checklist
  /// a photo counts for a cluster when its camera is this close to the centre (m) ...
  static let viewDistance: Float = 2.5
  /// ... and the centre is inside this half-angle around the view direction (~60° cone)
  static let viewHalfAngle: Float = 30 * .pi / 180
  static let sectorCount = 8
  /// "from above": camera at least this far above the cluster top (m) and pitched down more than abovePitch
  static let aboveHeight: Float = 0.3
  static let abovePitch: Float = 25 * .pi / 180
  /// "from above" is only needed when the cluster top is lower than floor + this (m)
  static let aboveNeededBelow: Float = 1.7
  /// "low": camera lower than floor + this (m); only needed when the cluster bottom is below floor + lowNeededBelow
  static let lowCameraHeight: Float = 0.8
  static let lowNeededBelow: Float = 0.5
  /// a photo (auto or manual) this close (m) is the close-up
  static let closeupDistance: Float = 1.0
  /// sectors needed when more are reachable
  static let minSectors = 3
  /// where someone stands for a sector, from the cluster centre (m)
  static let standRadii: [Float] = [0.7, 1.0, 1.3, 1.6]
  /// a stand point is blocked by mesh geometry within this distance (m) ...
  static let standClearance: Float = 0.2
  /// ... between these heights above the floor (m)
  static let obstacleMinHeight: Float = 0.1
  static let obstacleMaxHeight: Float = 1.9
  /// cell size of the obstacle grid (m)
  static let obstacleCell: Float = 0.1
  /// the HUD chip shows for clusters closer than this (m)
  static let chipDistance: Float = 3

  typealias Key = SIMD3<Int32>
  typealias Cell = SIMD2<Int32>

  static func key(_ p: SIMD3<Float>) -> Key {
    let q = (p / voxelSize).rounded(.down)
    return Key(Int32(q.x), Int32(q.y), Int32(q.z))
  }

  static func center(_ k: Key) -> SIMD3<Float> {
    let f = SIMD3<Float>(Float(k.x), Float(k.y), Float(k.z)) + 0.5
    return f * voxelSize
  }

  static func cell(_ x: Float, _ z: Float) -> Cell {
    Cell(Int32((x / obstacleCell).rounded(.down)), Int32((z / obstacleCell).rounded(.down)))
  }

  struct Count {
    var total: Int32 = 0
    var unsure: Int32 = 0
  }

  /// A RoomPlan wall as a rectangle (centre transform, width × height)
  struct Wall {
    let inverse: simd_float4x4
    let halfWidth: Float
    let halfHeight: Float
    init(transform: simd_float4x4, width: Float, height: Float) {
      inverse = transform.inverse
      halfWidth = width / 2
      halfHeight = height / 2
    }
    /// distance from p to the wall rectangle (m)
    func distance(_ p: SIMD3<Float>) -> Float {
      let l = inverse * SIMD4<Float>(p, 1)
      let dx = max(0, abs(l.x) - halfWidth)
      let dy = max(0, abs(l.y) - halfHeight)
      let dz = abs(l.z)
      return (dx * dx + dy * dy + dz * dz).squareRoot()
    }
  }

  /// total + unsure depth samples per 10 cm voxel
  struct Grid {
    private(set) var counts: [Key: Count] = [:]

    mutating func add(_ p: SIMD3<Float>, unsure: Bool) {
      let k = HardAreas.key(p)
      var c = counts[k] ?? Count()
      c.total += 1
      if unsure { c.unsure += 1 }
      counts[k] = c
    }

    static func isHard(_ c: Count) -> Bool {
      c.total > HardAreas.minSamples && Float(c.unsure) > HardAreas.unsureRatio * Float(c.total)
    }

    /// hard voxels, without those near walls or near the floor
    func hardKeys(walls: [Wall], floorY: Float?) -> [Key] {
      var out: [Key] = []
      for (k, c) in counts where Self.isHard(c) {
        let p = HardAreas.center(k)
        if let floorY, p.y < floorY + HardAreas.floorMargin { continue }
        if walls.contains(where: { $0.distance(p) <= HardAreas.wallMargin }) { continue }
        out.append(k)
      }
      return out
    }
  }

  /// 26-neighbour connected components with at least minSize voxels, biggest first
  static func components(_ keys: [Key], minSize: Int = minClusterVoxels) -> [[Key]] {
    var remaining = Set(keys)
    var out: [[Key]] = []
    var offsets: [Key] = []
    for dx in -1...1 {
      for dy in -1...1 {
        for dz in -1...1 where dx != 0 || dy != 0 || dz != 0 {
          offsets.append(Key(Int32(dx), Int32(dy), Int32(dz)))
        }
      }
    }
    for seed in keys where remaining.contains(seed) {
      remaining.remove(seed)
      var group: [Key] = [seed]
      var i = 0
      while i < group.count {
        let k = group[i]
        i += 1
        for o in offsets {
          let n = k &+ o
          if remaining.remove(n) != nil { group.append(n) }
        }
      }
      if group.count >= minSize { out.append(group) }
    }
    return out.sorted { $0.count > $1.count }
  }

  // MARK: clusters + angles

  enum Kind: String {
    case thinBars = "thin bars"
    case shiny = "shiny or dark"
  }

  /// A saved photo as the checklist needs it
  struct View {
    var position: SIMD3<Float>
    /// unit view direction (camera −Z in world)
    var forward: SIMD3<Float>
    var manual: Bool
  }

  struct Angles: Equatable {
    var sectors = [Bool](repeating: false, count: HardAreas.sectorCount)
    var reachable = [Bool](repeating: true, count: HardAreas.sectorCount)
    var above = false
    var low = false
    var closeup = false
  }

  enum Step: Equatable {
    case sector(Int)
    case above
    case low
    case closeup
  }

  struct Status: Equatable {
    var have = 0
    var total = 0
    var done = false
    /// missing angles, in the order the HUD asks for them (sectors: reachable ones only)
    var missing: [Step] = []
  }

  struct Cluster {
    var min: SIMD3<Float>
    var max: SIMD3<Float>
    var center: SIMD3<Float>
    var voxels: Int
    var kind: Kind
    var angles = Angles()
    var status = Status()
  }

  static func cluster(_ keys: [Key]) -> Cluster {
    var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
    var klo = Key(repeating: .max), khi = Key(repeating: .min)
    for k in keys {
      let c = center(k)
      lo = simd_min(lo, c - voxelSize / 2)
      hi = simd_max(hi, c + voxelSize / 2)
      klo = pointwiseMin(klo, k)
      khi = pointwiseMax(khi, k)
    }
    let span = khi &- klo &+ Key(repeating: 1)
    let box = Float(span.x) * Float(span.y) * Float(span.z)
    let fill = Float(keys.count) / Swift.max(1, box)
    // sparse (a rack, a frame) or a bar (two thin directions) = thin bars; else a solid shiny/dark thing
    let thinAxes = (span.x <= 2 ? 1 : 0) + (span.y <= 2 ? 1 : 0) + (span.z <= 2 ? 1 : 0)
    let kind: Kind = fill < 0.25 || thinAxes >= 2 ? .thinBars : .shiny
    return Cluster(min: lo, max: hi, center: (lo + hi) / 2, voxels: keys.count, kind: kind)
  }

  /// azimuth sector (0..<8) of point p around centre c; sector 0 starts at +X and runs toward +Z
  static func sector(of p: SIMD3<Float>, around c: SIMD3<Float>) -> Int {
    var a = atan2(p.z - c.z, p.x - c.x)
    if a < 0 { a += 2 * .pi }
    let s = Int(a / (2 * .pi) * Float(sectorCount))
    return Swift.min(sectorCount - 1, Swift.max(0, s))
  }

  /// unit x/z direction through the middle of a sector
  static func sectorDirection(_ s: Int) -> SIMD2<Float> {
    let a = (Float(s) + 0.5) / Float(sectorCount) * 2 * .pi
    return SIMD2(cos(a), sin(a))
  }

  /// where to stand for a sector at a given distance from the centre (x/z)
  static func standPoint(_ c: SIMD3<Float>, sector s: Int, radius: Float) -> SIMD2<Float> {
    SIMD2(c.x, c.z) + sectorDirection(s) * radius
  }

  /// true when the photo's camera is near the centre and looks at it
  static func sees(_ v: View, _ c: SIMD3<Float>) -> Bool {
    let to = c - v.position
    let d = simd_length(to)
    guard d > 0.05, d <= viewDistance else { return false }
    return simd_dot(to / d, v.forward) >= cos(viewHalfAngle)
  }

  /// Which angles these photos cover (reachable is left as given)
  static func observe(_ cluster: Cluster, photos: [View], floorY: Float?, reachable: [Bool]) -> Angles {
    var a = Angles()
    a.reachable = reachable
    let c = cluster.center
    for v in photos where sees(v, c) {
      a.sectors[sector(of: v.position, around: c)] = true
      if v.position.y >= cluster.max.y + aboveHeight, v.forward.y < -sin(abovePitch) { a.above = true }
      if let floorY, v.position.y - floorY < lowCameraHeight { a.low = true }
      if simd_distance(v.position, c) <= closeupDistance { a.closeup = true } // any photo, auto or manual
    }
    return a
  }

  /// Would a photo from this view add an angle the cluster still needs (a new sector,
  /// "from above" or "low")? Used to take more auto photos near hard spots.
  static func addsAngle(_ cluster: Cluster, view v: View, floorY: Float?) -> Bool {
    guard !cluster.status.done, sees(v, cluster.center) else { return false }
    let a = cluster.angles
    if !a.sectors[sector(of: v.position, around: cluster.center)] { return true }
    let missing = cluster.status.missing
    if missing.contains(.above), v.position.y >= cluster.max.y + aboveHeight, v.forward.y < -sin(abovePitch) { return true }
    if missing.contains(.low), let floorY, v.position.y - floorY < lowCameraHeight { return true }
    if missing.contains(.closeup), simd_distance(v.position, cluster.center) <= closeupDistance { return true }
    return false
  }

  static func inside(_ poly: [SIMD2<Float>], _ p: SIMD2<Float>) -> Bool {
    var hit = false
    var j = poly.count - 1
    for i in poly.indices {
      let a = poly[i], b = poly[j]
      if (a.y > p.y) != (b.y > p.y) {
        let x = a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y)
        if p.x < x { hit.toggle() }
      }
      j = i
    }
    return hit
  }

  /// mesh geometry (obstacle cells) within standClearance of p
  static func blocked(_ p: SIMD2<Float>, obstacles: Set<Cell>) -> Bool {
    guard !obstacles.isEmpty else { return false }
    let r = Int32((standClearance / obstacleCell).rounded(.up))
    let c = cell(p.x, p.y)
    for dx in -r...r {
      for dz in -r...r {
        let n = Cell(c.x + dx, c.y + dz)
        guard obstacles.contains(n) else { continue }
        let mid = (SIMD2<Float>(Float(n.x), Float(n.y)) + 0.5) * obstacleCell
        if simd_distance(mid, p) <= standClearance { return true }
      }
    }
    return false
  }

  /// A sector is reachable when one of its stand points is inside the room outline (when known)
  /// and clear of mesh geometry between obstacleMinHeight and obstacleMaxHeight
  static func reachability(_ c: SIMD3<Float>, outline: [SIMD2<Float>], obstacles: Set<Cell>) -> [Bool] {
    (0..<sectorCount).map { s in
      standRadii.contains { r in
        let p = standPoint(c, sector: s, radius: r)
        if outline.count >= 3, !inside(outline, p) { return false }
        return !blocked(p, obstacles: obstacles)
      }
    }
  }

  /// Done = enough sectors (all reachable ones, or at least minSectors), from above when the
  /// top is low, low when the bottom is low, and one close-up
  static func status(_ cluster: Cluster, _ a: Angles, floorY: Float?) -> Status {
    let reachableCount = a.reachable.filter { $0 }.count
    let neededSectors = reachableCount == 0 ? 1 : Swift.min(minSectors, reachableCount)
    let covered = a.sectors.filter { $0 }.count
    var s = Status()
    s.total = neededSectors + 1
    s.have = Swift.min(covered, neededSectors) + (a.closeup ? 1 : 0)
    if covered < neededSectors {
      for i in 0..<sectorCount where a.reachable[i] && !a.sectors[i] { s.missing.append(.sector(i)) }
      if s.missing.isEmpty { // nothing reachable is known: ask for any side
        for i in 0..<sectorCount where !a.sectors[i] { s.missing.append(.sector(i)) }
      }
    }
    var extrasOK = a.closeup
    if let floorY, cluster.max.y - floorY < aboveNeededBelow {
      s.total += 1
      if a.above { s.have += 1 } else { s.missing.append(.above); extrasOK = false }
    }
    if let floorY, cluster.min.y - floorY < lowNeededBelow {
      s.total += 1
      if a.low { s.have += 1 } else { s.missing.append(.low); extrasOK = false }
    }
    if !a.closeup { s.missing.append(.closeup) }
    s.done = covered >= neededSectors && extrasOK
    return s
  }

  /// The next missing angle to ask for: the missing sector closest to where the camera is now
  /// (around the centre), else the first other missing step
  static func next(_ s: Status, cameraSector: Int) -> Step? {
    var best: (Step, Int)?
    for step in s.missing {
      guard case .sector(let i) = step else { continue }
      let raw = abs(i - cameraSector) % sectorCount
      let d = Swift.min(raw, sectorCount - raw)
      if best == nil || d < best!.1 { best = (step, d) }
    }
    if let best { return best.0 }
    return s.missing.first
  }

  /// The step as an instruction, for a camera at camera with right vector right (world)
  static func action(_ step: Step, cluster c: Cluster, camera: SIMD3<Float>, right: SIMD3<Float>) -> String {
    switch step {
    case .above: return ScanHelperText.fromAbove
    case .low: return ScanHelperText.low
    case .closeup:
      let d = simd_distance(camera, c.center)
      return d <= closeupDistance ? ScanHelperText.holdStill : ScanHelperText.closeUp
    case .sector(let i):
      let here = sector(of: camera, around: c.center)
      let raw = abs(i - here) % sectorCount
      let steps = Swift.min(raw, sectorCount - raw)
      if steps == 0 {
        return simd_distance(camera, c.center) > viewDistance ? ScanHelperText.stepCloser : ScanHelperText.pointAtIt
      }
      if steps >= 3 { return ScanHelperText.farSide }
      let target = standPoint(c.center, sector: i, radius: 1.2)
      let to = SIMD2<Float>(target.x - camera.x, target.y - camera.z)
      let side = to.x * right.x + to.y * right.z
      return side >= 0 ? ScanHelperText.rightSide : ScanHelperText.leftSide
    }
  }

  // MARK: export (helper/hard_areas.json)

  struct File: Codable {
    struct Angles: Codable {
      var sectors: [Bool]
      var reachable: [Bool]
      var above: Bool
      var low: Bool
      var closeup: Bool
    }
    struct Cluster: Codable {
      var center: [Float]
      var min: [Float]
      var max: [Float]
      var voxels: Int
      var angles: Angles
      var done: Bool
    }
    struct Warnings: Codable {
      var slowSeconds: Double
      var darkSeconds: Double
      var glareSeconds: Double
    }
    var version = 1
    var room: Int
    var voxelSize: Float = HardAreas.voxelSize
    var clusters: [Cluster]
    var warnings: Warnings
  }

  static func file(room: Int, clusters: [HardAreas.Cluster], warnings: File.Warnings) -> File {
    func mm(_ v: SIMD3<Float>) -> [Float] { [v.x, v.y, v.z].map { ($0 * 1000).rounded() / 1000 } }
    let list = clusters.map { c in
      File.Cluster(
        center: mm(c.center), min: mm(c.min), max: mm(c.max), voxels: c.voxels,
        angles: File.Angles(sectors: c.angles.sectors, reachable: c.angles.reachable, above: c.angles.above, low: c.angles.low, closeup: c.angles.closeup),
        done: c.status.done)
    }
    return File(room: room, clusters: list, warnings: warnings)
  }
}
