import ARKit
import RoomPlan
import SceneKit
import SwiftUI

/// Camera speed from consecutive ARFrame poses, smoothed over ~0.3 s
struct MotionMeter {
  /// "Move slower" above these
  static let maxAngular: Float = 60 * .pi / 180 // rad/s
  static let maxLinear: Float = 0.6 // m/s
  static let smoothing: Double = 0.3 // s
  private(set) var angular: Float = 0
  private(set) var linear: Float = 0
  private var last: (time: TimeInterval, transform: simd_float4x4)?

  var tooFast: Bool { angular > Self.maxAngular || linear > Self.maxLinear }

  mutating func update(time: TimeInterval, transform: simd_float4x4) {
    let previous = last
    last = (time, transform)
    guard let previous else { return }
    let dt = time - previous.time
    guard dt > 0 else { return }
    guard dt < 0.5 else { // a gap (pause, lost frames): start over
      angular = 0
      linear = 0
      return
    }
    let q0 = simd_quatf(previous.transform), q1 = simd_quatf(transform)
    let dq = q0.inverse * q1
    let angle: Float = 2 * acos(min(1, abs(dq.real)))
    let moved: Float = simd_distance(previous.transform.columns.3, transform.columns.3)
    let a = Float(1 - exp(-dt / Self.smoothing))
    let w = angle / Float(dt), v = moved / Float(dt)
    angular += (w - angular) * a
    linear += (v - linear) * a
  }
}

/// Light thresholds (checked every ~0.5 s)
enum LightCheck {
  static let interval: TimeInterval = 0.5
  /// too dark: ambient intensity under this (lumen-ish, 1000 = well lit) ...
  static let darkAmbient: Float = 250
  /// ... or the camera ISO over this, for darkHold seconds
  static let darkISO: Float = 800
  static let darkHold: TimeInterval = 2
  /// glare: more than glareFraction of the luma samples at or above glareLuma
  static let glareLuma: UInt8 = 250
  static let glareFraction: Float = 0.08
  static let lumaStep = 12

  /// fraction of sampled Y (luma) pixels at or above glareLuma
  static func brightFraction(_ image: CVPixelBuffer) -> Float {
    guard CVPixelBufferGetPlaneCount(image) >= 1 else { return 0 }
    CVPixelBufferLockBaseAddress(image, .readOnly)
    defer { CVPixelBufferUnlockBaseAddress(image, .readOnly) }
    guard let base = CVPixelBufferGetBaseAddressOfPlane(image, 0) else { return 0 }
    let w = CVPixelBufferGetWidthOfPlane(image, 0), h = CVPixelBufferGetHeightOfPlane(image, 0)
    let row = CVPixelBufferGetBytesPerRowOfPlane(image, 0)
    var total = 0, bright = 0
    var y = lumaStep / 2
    while y < h {
      let line = base.advanced(by: y * row).assumingMemoryBound(to: UInt8.self)
      var x = lumaStep / 2
      while x < w {
        total += 1
        if line[x] >= glareLuma { bright += 1 }
        x += lumaStep
      }
      y += lumaStep
    }
    return total > 0 ? Float(bright) / Float(total) : 0
  }
}

/// One depth frame, subsampled (every HardAreas.pixelStep pixel), for the helper queue
struct HelperDepth {
  var depth: [Float32]
  var confidence: [UInt8]
  var width: Int
  var height: Int
  var step: Int
  /// intrinsics at the depth map's full resolution
  var fx: Float, fy: Float, cx: Float, cy: Float
  var transform: simd_float4x4

  static func sample(_ frame: ARFrame) -> HelperDepth? {
    guard let sd = frame.sceneDepth, let conf = sd.confidenceMap else { return nil }
    let depth = sd.depthMap
    CVPixelBufferLockBaseAddress(depth, .readOnly)
    CVPixelBufferLockBaseAddress(conf, .readOnly)
    defer {
      CVPixelBufferUnlockBaseAddress(conf, .readOnly)
      CVPixelBufferUnlockBaseAddress(depth, .readOnly)
    }
    let w = CVPixelBufferGetWidth(depth), h = CVPixelBufferGetHeight(depth)
    guard CVPixelBufferGetWidth(conf) == w, CVPixelBufferGetHeight(conf) == h,
          let db = CVPixelBufferGetBaseAddress(depth), let cb = CVPixelBufferGetBaseAddress(conf) else { return nil }
    let dRow = CVPixelBufferGetBytesPerRow(depth), cRow = CVPixelBufferGetBytesPerRow(conf)
    let step = HardAreas.pixelStep
    let sw = w / step, sh = h / step
    var d = [Float32](), c = [UInt8]()
    d.reserveCapacity(sw * sh)
    c.reserveCapacity(sw * sh)
    for j in 0..<sh {
      let drow = db.advanced(by: j * step * dRow).assumingMemoryBound(to: Float32.self)
      let crow = cb.advanced(by: j * step * cRow).assumingMemoryBound(to: UInt8.self)
      for i in 0..<sw {
        d.append(drow[i * step])
        c.append(crow[i * step])
      }
    }
    let scale = Float(w) / Float(frame.camera.imageResolution.width)
    let k = frame.camera.intrinsics
    return HelperDepth(
      depth: d, confidence: c, width: sw, height: sh, step: step,
      fx: k.columns.0.x * scale, fy: k.columns.1.y * scale, cx: k.columns.2.x * scale, cy: k.columns.2.y * scale,
      transform: frame.camera.transform)
  }
}

/// The hard-area voxels and clusters, on their own serial queue (all state queue-only)
final class HardAreaWorker: @unchecked Sendable {
  struct Overlay {
    var positions: Data
    var colors: Data
    var indices: Data
    var vertexCount: Int
    var triangleCount: Int
  }
  struct Result {
    var generation: Int
    var clusters: [HardAreas.Cluster]
    var floorY: Float?
    var overlay: Overlay?
  }

  private let queue = DispatchQueue(label: "scan-helper", qos: .utility)
  private let lock = NSLock()
  private var _depthBusy = false
  private var _refreshBusy = false

  private var generation = 0
  private var grid = HardAreas.Grid()
  private var walls: [HardAreas.Wall] = []
  private var outline: [SIMD2<Float>] = []
  private var roomFloor: Float?
  private var floorGuess: Float = .infinity
  private var photos: [HardAreas.View] = []
  private var obstacles: Set<HardAreas.Cell> = []

  /// most voxels drawn in the AR view
  static let maxDrawnVoxels = 3000

  var depthBusy: Bool { lock.withLock { _depthBusy } }

  func reset(generation g: Int) {
    queue.async {
      self.generation = g
      self.grid = HardAreas.Grid()
      self.walls = []
      self.outline = []
      self.roomFloor = nil
      self.floorGuess = .infinity
      self.photos = []
      self.obstacles = []
    }
  }

  func setRoom(walls: [HardAreas.Wall], outline: [SIMD2<Float>], floorY: Float?) {
    queue.async {
      self.walls = walls
      self.outline = outline
      self.roomFloor = floorY
    }
  }

  func addPhoto(_ v: HardAreas.View) {
    queue.async { self.photos.append(v) }
  }

  func setObstacles(_ cells: Set<HardAreas.Cell>) {
    queue.async { self.obstacles = cells }
  }

  /// false (frame dropped) while the previous frame is still being integrated
  @discardableResult
  func add(_ frame: HelperDepth) -> Bool {
    let accepted = lock.withLock { () -> Bool in
      if _depthBusy { return false }
      _depthBusy = true
      return true
    }
    guard accepted else { return false }
    queue.async {
      self.integrate(frame)
      self.lock.withLock { self._depthBusy = false }
    }
    return true
  }

  /// Recompute clusters + angles (+ the overlay); skipped while a refresh is still running
  func refresh(done: @escaping (Result) -> Void) {
    let accepted = lock.withLock { () -> Bool in
      if _refreshBusy { return false }
      _refreshBusy = true
      return true
    }
    guard accepted else { return }
    queue.async {
      let (clusters, keys) = self.compute()
      let overlay = Self.overlay(clusters: clusters, keys: keys)
      let result = Result(generation: self.generation, clusters: clusters, floorY: self.floorY, overlay: overlay)
      self.lock.withLock { self._refreshBusy = false }
      done(result)
    }
  }

  /// Writes helper/hard_areas.json (same content to every URL); done(true) when all were written
  func export(room: Int, warnings: HardAreas.File.Warnings, to urls: [URL], done: @escaping (Bool) -> Void) {
    queue.async {
      let (clusters, _) = self.compute()
      let file = HardAreas.file(room: room, clusters: clusters, warnings: warnings)
      let e = JSONEncoder()
      e.outputFormatting = [.prettyPrinted, .sortedKeys]
      var ok = false
      if let data = try? e.encode(file) {
        ok = true
        for url in urls {
          do {
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try data.write(to: url, options: .atomic)
          } catch {
            ok = false
          }
        }
      }
      done(ok)
    }
  }

  // MARK: queue only

  private var floorY: Float? {
    if let roomFloor { return roomFloor }
    return floorGuess.isFinite ? floorGuess : nil
  }

  private func integrate(_ f: HelperDepth) {
    var heights: [Float] = []
    heights.reserveCapacity(f.depth.count / 8 + 1)
    let t = f.transform
    for j in 0..<f.height {
      let v = Float(j * f.step) + 0.5
      let yFactor = -(v - f.cy) / f.fy
      for i in 0..<f.width {
        let index = j * f.width + i
        let z = f.depth[index]
        guard z.isFinite, z >= HardAreas.minDepth, z <= HardAreas.maxDepth else { continue }
        let u = Float(i * f.step) + 0.5
        let xc = (u - f.cx) / f.fx * z
        let yc = yFactor * z
        let world = t * SIMD4<Float>(xc, yc, -z, 1)
        let p = SIMD3<Float>(world.x, world.y, world.z)
        // ARConfidenceLevel: 0 low, 1 medium, 2 high; low/medium = unsure
        grid.add(p, unsure: f.confidence[index] < 2)
        if index % 8 == 0 { heights.append(p.y) }
      }
    }
    // no RoomPlan floor yet: the lowest few percent of what the depth sees
    if heights.count > 200 {
      heights.sort()
      floorGuess = min(floorGuess, heights[heights.count / 50])
    }
  }

  private func compute() -> ([HardAreas.Cluster], [[HardAreas.Key]]) {
    let floor = floorY
    let keys = grid.hardKeys(walls: walls, floorY: floor)
    let groups = HardAreas.components(keys)
    var clusters: [HardAreas.Cluster] = []
    clusters.reserveCapacity(groups.count)
    for g in groups {
      var c = HardAreas.cluster(g)
      let reachable = HardAreas.reachability(c.center, outline: outline, obstacles: obstacles)
      c.angles = HardAreas.observe(c, photos: photos, floorY: floor, reachable: reachable)
      c.status = HardAreas.status(c, c.angles, floorY: floor)
      clusters.append(c)
    }
    return (clusters, groups)
  }

  /// translucent cubes: purple = hard, green = cluster done
  private static func overlay(clusters: [HardAreas.Cluster], keys: [[HardAreas.Key]]) -> Overlay? {
    var positions: [Float] = []
    var colors: [Float] = []
    var indices: [UInt32] = []
    let half: Float = HardAreas.voxelSize * 0.4
    let corners: [SIMD3<Float>] = [
      SIMD3(-1, -1, -1), SIMD3(1, -1, -1), SIMD3(1, 1, -1), SIMD3(-1, 1, -1),
      SIMD3(-1, -1, 1), SIMD3(1, -1, 1), SIMD3(1, 1, 1), SIMD3(-1, 1, 1),
    ]
    let faces: [UInt32] = [
      0, 2, 1, 0, 3, 2, // -z
      4, 5, 6, 4, 6, 7, // +z
      0, 1, 5, 0, 5, 4, // -y
      3, 7, 6, 3, 6, 2, // +y
      0, 4, 7, 0, 7, 3, // -x
      1, 2, 6, 1, 6, 5, // +x
    ]
    let purple: [Float] = [0.72, 0.33, 0.95, 1]
    let green: [Float] = [0.22, 0.9, 0.45, 1]
    var drawn = 0
    for (n, group) in keys.enumerated() where n < clusters.count {
      let color = clusters[n].status.done ? green : purple
      for k in group {
        guard drawn < maxDrawnVoxels else { break }
        drawn += 1
        let c = HardAreas.center(k)
        let first = UInt32(positions.count / 3)
        for corner in corners {
          let p = c + corner * half
          positions.append(contentsOf: [p.x, p.y, p.z])
          colors.append(contentsOf: color)
        }
        for f in faces { indices.append(first + f) }
      }
    }
    guard drawn > 0 else { return nil }
    return Overlay(
      positions: positions.withUnsafeBufferPointer { Data(buffer: $0) },
      colors: colors.withUnsafeBufferPointer { Data(buffer: $0) },
      indices: indices.withUnsafeBufferPointer { Data(buffer: $0) },
      vertexCount: positions.count / 3,
      triangleCount: indices.count / 3)
  }
}

/// Live scan helper: hard areas (+ their angle checklist), "Move slower" and light warnings.
/// Guidance only: it never blocks saving (same policy as CaptureQuality). Heavy work runs on
/// HardAreaWorker's queue, and it backs off with CaptureBudget.
@MainActor
final class ScanHelper: ObservableObject {
  struct Chip: Equatable {
    var title: String
    var detail: String
  }
  struct Warning: Hashable {
    var text: String
    var icon: String
  }

  @Published private(set) var chip: Chip?
  @Published private(set) var moveSlower = false
  @Published private(set) var tooDark = false
  @Published private(set) var glare = false
  @Published private(set) var hardSpots = 0
  @Published private(set) var doneSpots = 0
  private(set) var slowSeconds: Double = 0
  private(set) var darkSeconds: Double = 0
  private(set) var glareSeconds: Double = 0

  var warnings: [Warning] {
    var out: [Warning] = []
    if moveSlower { out.append(Warning(text: "Move slower", icon: "tortoise.fill")) }
    if tooDark { out.append(Warning(text: "Too dark: turn on more lights", icon: "lightbulb.fill")) }
    if glare { out.append(Warning(text: "Bright window/lamp in view: film at an angle or close the curtain", icon: "sun.max.fill")) }
    return out
  }

  let node = SCNNode()
  private let worker = HardAreaWorker()
  private let material: SCNMaterial = {
    let m = SCNMaterial()
    m.lightingModel = .constant
    m.diffuse.contents = UIColor.white // vertex colours carry purple / green
    m.transparency = 0.45
    m.blendMode = .alpha
    m.isDoubleSided = true
    m.readsFromDepthBuffer = true
    m.writesToDepthBuffer = false
    return m
  }()
  private var clusters: [HardAreas.Cluster] = []
  private var floorY: Float?
  private var generation = 0
  private var motion = MotionMeter()
  private var lastTick: TimeInterval?
  private var lastDepth: TimeInterval = 0
  private var lastRefresh: TimeInterval = 0
  private var lastObstacles: TimeInterval = 0
  private var lastLight: TimeInterval = 0
  private var lastChip: TimeInterval = 0
  private var lastHaptic: TimeInterval = -.infinity
  private var lastRoomUpdate: TimeInterval = 0
  private var darkSince: TimeInterval?
  private let haptic = UIImpactFeedbackGenerator(style: .light)

  /// the chip only shows for clusters inside this half-angle of the view
  private static let chipHalfAngle: Float = 35 * .pi / 180

  init() {
    node.renderingOrder = 6
  }

  func attach(to view: ARSCNView) {
    if node.parent == nil { view.scene.rootNode.addChildNode(node) }
  }

  /// A new room: forget the last room's voxels, photos and warning times
  func startRoom() {
    generation += 1
    worker.reset(generation: generation)
    clusters = []
    floorY = nil
    chip = nil
    moveSlower = false
    tooDark = false
    glare = false
    hardSpots = 0
    doneSpots = 0
    slowSeconds = 0
    darkSeconds = 0
    glareSeconds = 0
    motion = MotionMeter()
    lastTick = nil
    darkSince = nil
    lastRoomUpdate = 0
    node.geometry = nil
  }

  /// RoomPlan's live room (throttled): walls to ignore, the outline for reachability, the floor
  func setRoom(_ room: CapturedRoom?) {
    let now = CACurrentMediaTime()
    guard now - lastRoomUpdate > 2 else { return }
    lastRoomUpdate = now
    guard let room else { return }
    let walls = room.walls.map { HardAreas.Wall(transform: $0.transform, width: $0.dimensions.x, height: $0.dimensions.y) }
    let (outline, floor, _) = CoverageEngine.bounds(for: room)
    worker.setRoom(walls: walls, outline: outline, floorY: floor > -9 ? floor : nil)
  }

  /// A photo (auto or manual) was saved
  func photoSaved(position: SIMD3<Float>, forward: SIMD3<Float>, manual: Bool) {
    worker.addPhoto(HardAreas.View(position: position, forward: forward, manual: manual))
  }

  /// Every new ARFrame while scanning and capturing
  func tick(_ frame: ARFrame, budget: CaptureBudget, engine: CoverageEngine, iso: () -> Float?) {
    let ts = frame.timestamp
    let dt = lastTick.map { min(0.1, max(0, ts - $0)) } ?? 0
    lastTick = ts

    motion.update(time: ts, transform: frame.camera.transform)
    let fast = motion.tooFast
    if fast != moveSlower { moveSlower = fast }
    if fast {
      slowSeconds += dt
      if ts - lastHaptic >= 3 {
        lastHaptic = ts
        haptic.impactOccurred()
      }
    }
    if tooDark { darkSeconds += dt }
    if glare { glareSeconds += dt }
    if ts - lastLight >= LightCheck.interval {
      lastLight = ts
      updateLight(frame, iso: iso())
    }

    guard !budget.critical else { return } // hot phone: warnings only
    if ts - lastDepth >= max(0.2, budget.depthInterval), !worker.depthBusy, let sample = HelperDepth.sample(frame) {
      if worker.add(sample) { lastDepth = ts }
    }
    if ts - lastRefresh >= (budget.constrained ? 2 : 1) {
      lastRefresh = ts
      worker.refresh { [weak self] result in
        Task { @MainActor in self?.apply(result) }
      }
    }
    if !clusters.isEmpty, let floor = floorY, ts - lastObstacles >= (budget.constrained ? 6 : 3) {
      lastObstacles = ts
      let worker = worker
      engine.occupiedCells(minY: floor + HardAreas.obstacleMinHeight, maxY: floor + HardAreas.obstacleMaxHeight, cell: HardAreas.obstacleCell) { cells in
        worker.setObstacles(cells)
      }
    }
    if ts - lastChip >= 0.5 {
      lastChip = ts
      updateChip(frame.camera.transform)
    }
  }

  /// Writes the room's hard_areas.json; true when written
  func export(room: Int, to urls: [URL]) async -> Bool {
    let warnings = HardAreas.File.Warnings(
      slowSeconds: (slowSeconds * 10).rounded() / 10,
      darkSeconds: (darkSeconds * 10).rounded() / 10,
      glareSeconds: (glareSeconds * 10).rounded() / 10)
    let worker = worker
    return await withCheckedContinuation { k in
      worker.export(room: room, warnings: warnings, to: urls) { k.resume(returning: $0) }
    }
  }

  /// The camera looks at a hard spot that isn't done (within HardAreas.viewDistance)
  func viewsOpenSpot(_ t: simd_float4x4) -> Bool {
    let v = PhotoView(transform: t)
    let view = HardAreas.View(position: v.position, forward: v.forward, manual: false)
    return clusters.contains { !$0.status.done && HardAreas.sees(view, $0.center) }
  }

  /// A photo from here would add a missing angle (sector / from above / low) to a hard spot
  func addsNewAngle(_ t: simd_float4x4) -> Bool {
    guard !clusters.isEmpty else { return false }
    let v = PhotoView(transform: t)
    let view = HardAreas.View(position: v.position, forward: v.forward, manual: false)
    let floor = floorY
    return clusters.contains { HardAreas.addsAngle($0, view: view, floorY: floor) }
  }

  // MARK: private

  private func apply(_ result: HardAreaWorker.Result) {
    guard result.generation == generation else { return } // from the previous room
    clusters = result.clusters
    floorY = result.floorY
    let done = clusters.filter { $0.status.done }.count
    if hardSpots != clusters.count { hardSpots = clusters.count }
    if doneSpots != done { doneSpots = done }
    guard let o = result.overlay else {
      node.geometry = nil
      return
    }
    let vertices = SCNGeometrySource(
      data: o.positions, semantic: .vertex, vectorCount: o.vertexCount,
      usesFloatComponents: true, componentsPerVector: 3, bytesPerComponent: 4, dataOffset: 0, dataStride: 12)
    let colors = SCNGeometrySource(
      data: o.colors, semantic: .color, vectorCount: o.vertexCount,
      usesFloatComponents: true, componentsPerVector: 4, bytesPerComponent: 4, dataOffset: 0, dataStride: 16)
    let element = SCNGeometryElement(data: o.indices, primitiveType: .triangles, primitiveCount: o.triangleCount, bytesPerIndex: 4)
    let geometry = SCNGeometry(sources: [vertices, colors], elements: [element])
    geometry.materials = [material]
    node.geometry = geometry
  }

  private func updateLight(_ frame: ARFrame, iso: Float?) {
    let ts = frame.timestamp
    var darkNow = false
    if let e = frame.lightEstimate, Float(e.ambientIntensity) < LightCheck.darkAmbient { darkNow = true }
    if let iso, iso > LightCheck.darkISO { darkNow = true }
    if darkNow {
      if darkSince == nil { darkSince = ts }
    } else {
      darkSince = nil
    }
    let dark = darkSince.map { ts - $0 >= LightCheck.darkHold } ?? false
    if dark != tooDark { tooDark = dark }
    let bright = LightCheck.brightFraction(frame.capturedImage) > LightCheck.glareFraction
    if bright != glare { glare = bright }
  }

  private func updateChip(_ t: simd_float4x4) {
    let camera = simd_make_float3(t.columns.3)
    let forward = -simd_normalize(simd_make_float3(t.columns.2))
    var best: HardAreas.Cluster?
    var bestDistance = Float.infinity
    for c in clusters where !c.status.done {
      let to = c.center - camera
      let d = simd_length(to)
      guard d > 0.05, d < HardAreas.chipDistance, d < bestDistance else { continue }
      guard simd_dot(to / d, forward) >= cos(Self.chipHalfAngle) else { continue }
      best = c
      bestDistance = d
    }
    guard let c = best,
          let step = HardAreas.next(c.status, cameraSector: HardAreas.sector(of: camera, around: c.center)) else {
      if chip != nil { chip = nil }
      return
    }
    // right of the view, level (independent of how the phone is held)
    let flat = simd_cross(forward, SIMD3<Float>(0, 1, 0))
    let right = simd_length(flat) > 1e-4 ? simd_normalize(flat) : simd_make_float3(t.columns.0)
    let title = "Hard spot (\(c.kind.rawValue)): \(c.status.have) of \(c.status.total) angles"
    let detail = "Next: " + HardAreas.action(step, cluster: c, camera: camera, right: right)
    let next = Chip(title: title, detail: detail)
    if chip != next { chip = next }
  }
}
