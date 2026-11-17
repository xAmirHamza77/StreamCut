import AppKit
import Foundation
import Vision

/// Native macOS Apple Vision face & upper-body detector for StreamCut.
/// Usage: streamcut-vision-detect <img1.jpg> [<img2.jpg> ...]
/// Prints JSON array of { path, width, height, faces: [...], humans: [...] }
/// using top-left pixel coordinates matching ffmpeg's crop filter.
let args = Array(CommandLine.arguments.dropFirst())
var results: [[String: Any]] = []

for path in args {
    guard let img = NSImage(contentsOfFile: path),
          let cgImg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        continue
    }
    let w = Double(cgImg.width)
    let h = Double(cgImg.height)
    let faceReq = VNDetectFaceRectanglesRequest()
    let humanReq = VNDetectHumanRectanglesRequest()
    if #available(macOS 12.0, *) {
        humanReq.upperBodyOnly = true
    }
    let handler = VNImageRequestHandler(cgImage: cgImg, options: [:])
    try? handler.perform([faceReq, humanReq])

    let faces = (faceReq.results ?? []).map { obs -> [String: Any] in
        let b = obs.boundingBox
        return [
            "x": b.origin.x * w,
            "y": (1.0 - b.origin.y - b.height) * h,
            "w": b.width * w,
            "h": b.height * h,
            "conf": obs.confidence,
        ]
    }
    let humans = (humanReq.results ?? []).map { obs -> [String: Any] in
        let b = obs.boundingBox
        return [
            "x": b.origin.x * w,
            "y": (1.0 - b.origin.y - b.height) * h,
            "w": b.width * w,
            "h": b.height * h,
            "conf": obs.confidence,
        ]
    }
    results.append([
        "path": path,
        "width": w,
        "height": h,
        "faces": faces,
        "humans": humans,
    ])
}

let json = try JSONSerialization.data(withJSONObject: results, options: [])
if let str = String(data: json, encoding: .utf8) {
    print(str)
} else {
    print("[]")
}
