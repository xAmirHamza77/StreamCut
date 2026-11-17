"""Find the streamer's webcam in a landscape recording and split the frame: gameplay on top, facecam at the bottom."""
import json
import math
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from . import config
from .media import NO_WINDOW

CAM_POSITIONS = {
    "auto": "Auto-detect face (recommended)",
    "tl": "Top-left",
    "tc": "Top-center",
    "tr": "Top-right",
    "ml": "Mid-left",
    "mr": "Mid-right",
    "bl": "Bottom-left",
    "bc": "Bottom-center",
    "br": "Bottom-right",
}
CAM_STYLES = {
    "split": "Split bar at bottom",
    "center": "Bottom center card",
}
CORNER_PRESETS = ("tl", "tc", "tr", "ml", "mr", "bl", "bc", "br")
TOP_FRAC = 0.625  # 1080x1200 gameplay on top, 1080x720 facecam at the bottom

_SWIFT_SRC = Path(__file__).resolve().parent / "vision_detect.swift"
_YUNET_MODEL = Path(__file__).resolve().parent / "models" / "face_detection_yunet_2023mar.onnx"
_COMPILE_LOCK = threading.Lock()
_TLS = threading.local()


def _clamp_even(n: float, max_val: int) -> int:
    hi = max(2, int(max_val) - (int(max_val) % 2))
    even = int(math.floor(n / 2.0)) * 2
    return max(2, min(hi, even))


def _clamp_even_range(n: float, min_val: float, max_val: int) -> int:
    hi = max(2, int(max_val) - (int(max_val) % 2))
    lo = min(hi, max(2, int(math.ceil(min_val / 2.0)) * 2))
    even = int(math.floor(n / 2.0)) * 2
    return max(lo, min(hi, even))


def _box_iou(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def dedupe_detections(items: list, iou_thresh: float = 0.35) -> list:
    """Keep higher-confidence boxes when the detector reports overlapping rectangles."""
    kept = []
    ordered = sorted(
        (c for c in (items or []) if c and c.get("w", 0) > 0 and c.get("h", 0) > 0),
        key=lambda c: (-float(c.get("conf", 0.0)), -(c["w"] * c["h"])),
    )
    for box in ordered:
        if all(_box_iou(box, prev) <= iou_thresh for prev in kept):
            kept.append(box)
    return kept


def infer_corner_code(cx: float, cy: float, src_w: int, src_h: int) -> str:
    """Short label describing where on the source screen the webcam sits."""
    rx = cx / max(1, src_w)
    ry = cy / max(1, src_h)
    if 0.3 <= ry <= 0.7:
        if rx < 0.42:
            return "ml"
        if rx > 0.58:
            return "mr"
    if 0.36 <= rx <= 0.64:
        return "tc" if ry < 0.5 else "bc"
    return ("t" if ry < 0.5 else "b") + ("l" if rx < 0.5 else "r")


def gameplay_crop_rect(src_w: int, src_h: int, top_frac: float = TOP_FRAC) -> tuple[int, int, dict]:
    """Aspect-preserving centre crop from the source stream for the top gameplay half."""
    out_w, out_h = config.OUT_W, config.OUT_H
    top_h = max(400, min(out_h - 400, int(round((out_h * top_frac) / 2.0)) * 2))
    bot_h = out_h - top_h
    game_w = min(src_w, round(src_h * (out_w / top_h)))
    game_h = round(game_w * (top_h / out_w))
    if game_h > src_h:
        game_h = src_h
        game_w = round(game_h * (out_w / top_h))
    gw = _clamp_even(game_w, src_w)
    gh = _clamp_even(game_h, src_h)
    gx = max(0, min(src_w - gw, (src_w - gw) // 2))
    gy = max(0, min(src_h - gh, (src_h - gh) // 2))
    return top_h, bot_h, {"x": gx, "y": gy, "w": gw, "h": gh}


def compute_inset_cam_rect(box: dict, src_w: int, src_h: int, top_frac: float = TOP_FRAC, is_human: bool = False) -> dict:
    """Build exact non-stretched gameplay (top) and webcam (bottom) crop rectangles around a detected face/body."""
    top_h, bot_h, game = gameplay_crop_rect(src_w, src_h, top_frac)
    bot_aspect = config.OUT_W / bot_h  # 1.5 for 1080x720

    max_w_by_h = int(math.floor(src_h * bot_aspect))
    desired_w = max(box["w"] * 1.45, src_w * 0.20) if is_human else max(box["w"] * 3.5, src_w * 0.19)
    max_allowed_w = min(src_w, max_w_by_h, round(src_w * (0.38 if is_human else 0.36)))
    min_allowed_w = min(max_allowed_w, round(src_w * 0.16))

    cw = _clamp_even_range(desired_w, min_allowed_w, max_allowed_w)
    ch = _clamp_even(round(cw / bot_aspect), src_h)
    cw = _clamp_even(round(ch * bot_aspect), src_w)

    cx_mid = box["x"] + box["w"] / 2.0
    cy_mid = box["y"] + box["h"] / 2.0
    raw_x = cx_mid - cw / 2.0
    # Hug screen left/right edge slightly for corner cams so adjacent gameplay isn't pulled in.
    edge_x = src_w * 0.032
    if raw_x <= edge_x and raw_x > 0 and (box["x"] + box["w"] * 1.15) <= cw:
        raw_x = max(0.0, raw_x - min(raw_x, src_w * 0.015))
    elif raw_x + cw >= src_w - edge_x and (box["x"] - box["w"] * 0.15) >= src_w - cw:
        raw_x = min(float(src_w - cw), raw_x + min(src_w - (raw_x + cw), src_w * 0.015))
    cx = max(0, min(src_w - cw, int(math.floor(raw_x / 2.0)) * 2))

    if is_human:
        raw_y = box["y"] - box["h"] * 0.08
    else:
        # Keep eyes/nose comfortably above middle (~42% from top) without slicing the forehead or headset;
        # when a webcam sits right at the top corner, hug upward so overlays underneath (chat/tips) stay out.
        ideal_y = cy_mid - ch * 0.42
        headroom = max(0.0, box["y"] - box["h"] * 0.38)
        raw_y = min(ideal_y, headroom)
        if box["y"] <= src_h * 0.15 and raw_y <= src_h * 0.085:
            hug_y = max(0.0, box["y"] - box["h"] * 0.90)
            if hug_y + ch >= box["y"] + box["h"] * 1.35:
                raw_y = min(raw_y, hug_y)
    cy = max(0, min(src_h - ch, int(math.floor(raw_y / 2.0)) * 2))

    return {
        "mode": "gamer",
        "corner": infer_corner_code(cx_mid, cy_mid, src_w, src_h),
        "top_h": top_h,
        "bot_h": bot_h,
        "game": game,
        "cam": {"x": cx, "y": cy, "w": cw, "h": ch},
    }


def corner_preset_rect(src_w: int, src_h: int, corner: str, top_frac: float = TOP_FRAC) -> dict:
    """Fixed webcam crop (corners, bottom/top center, mid-left/right) when overriding auto face detection."""
    top_h, bot_h, game = gameplay_crop_rect(src_w, src_h, top_frac)
    cam_w_raw = round(src_w * 0.25)
    cam_h_raw = min(src_h, round(cam_w_raw * (bot_h / config.OUT_W)))
    cw = _clamp_even(cam_w_raw, src_w)
    ch = _clamp_even(cam_h_raw, src_h)
    pad_x = round(src_w * 0.015)
    pad_y = round(src_h * 0.02)
    h_pos = corner[-1]
    v_pos = corner[0]
    cx = max(0, src_w - cw - pad_x) if h_pos == "r" else ((src_w - cw) // 2 if h_pos == "c" else min(src_w - cw, pad_x))
    cy = max(0, src_h - ch - pad_y) if v_pos == "b" else ((src_h - ch) // 2 if v_pos == "m" else min(src_h - ch, pad_y))
    cx = max(0, min(src_w - cw, (cx // 2) * 2))
    cy = max(0, min(src_h - ch, (cy // 2) * 2))
    return {
        "mode": "gamer",
        "corner": corner,
        "top_h": top_h,
        "bot_h": bot_h,
        "game": game,
        "cam": {"x": cx, "y": cy, "w": cw, "h": ch},
    }


def _is_enclosing_large_person(face: dict, humans: list, src_h: int) -> bool:
    if not humans:
        return False
    fcx = face["x"] + face["w"] / 2.0
    fcy = face["y"] + face["h"] / 2.0
    return any(
        h["h"] > src_h * 0.42
        and h["x"] - h["w"] * 0.15 <= fcx <= h["x"] + h["w"] * 1.15
        and h["y"] - h["h"] * 0.15 <= fcy <= h["y"] + h["h"] * 1.1
        for h in humans
    )


def _normalize_frame(sample: dict, src_w: int, src_h: int) -> dict:
    sx = src_w / sample["width"] if sample.get("width", 0) > 0 else 1.0
    sy = src_h / sample["height"] if sample.get("height", 0) > 0 else 1.0
    faces = dedupe_detections([
        {"x": f["x"] * sx, "y": f["y"] * sy, "w": f["w"] * sx, "h": f["h"] * sy, "conf": float(f.get("conf", 0.8))}
        for f in (sample.get("faces") or [])
        if float(f.get("conf", 0.8)) >= 0.46 and f["w"] * sx >= src_w * 0.025 and f["h"] * sy >= src_h * 0.03
    ])
    humans = dedupe_detections([
        {"x": h["x"] * sx, "y": h["y"] * sy, "w": h["w"] * sx, "h": h["h"] * sy, "conf": float(h.get("conf", 0.7))}
        for h in (sample.get("humans") or [])
        if float(h.get("conf", 0.7)) >= 0.50 and h["w"] * sx >= src_w * 0.05 and h["h"] * sy >= src_h * 0.07
    ])
    return {"faces": faces, "humans": humans}


def _pick_stationary_cluster(candidates: list, readable: int, src_w: int, src_h: int, is_human: bool) -> list:
    if len(candidates) < 2:
        return []
    tol_x, tol_y = src_w * 0.065, src_h * 0.085
    min_agree = max(2, math.ceil(readable * 0.3))
    min_req = max(3, math.ceil(readable * 0.65)) if is_human else max(2, min_agree)

    best_score, best_group = None, []
    for seed in candidates:
        per_frame = {}
        for c in candidates:
            if abs(c["cx"] - seed["cx"]) <= tol_x and abs(c["cy"] - seed["cy"]) <= tol_y:
                prev = per_frame.get(c["fi"])
                if not prev or c["box"].get("conf", 0.0) > prev["box"].get("conf", 0.0):
                    per_frame[c["fi"]] = c
        group = list(per_frame.values())
        if len(group) < min_req:
            continue

        med_cx = statistics.median(c["cx"] for c in group) / max(1, src_w)
        med_cy = statistics.median(c["cy"] for c in group) / max(1, src_h)
        med_h = statistics.median(c["box"]["h"] for c in group) / max(1, src_h)
        edge_dist = min(med_cx, 1.0 - med_cx, med_cy, 1.0 - med_cy)
        corner_bonus = max(0.0, 0.35 - edge_dist) + (0.15 if med_h <= 0.21 else 0.0)
        jitter = 0.0
        if len(group) >= 2:
            jitter = statistics.pstdev(c["cx"] / max(1, src_w) for c in group) + statistics.pstdev(c["cy"] / max(1, src_h) for c in group)
        score = (len(group), round(corner_bonus - jitter * 4.0, 4))
        if best_score is None or score > best_score:
            best_score, best_group = score, group
    return best_group


def detect_webcam_inset(samples: list, src_w: int, src_h: int, top_frac: float = TOP_FRAC) -> dict | None:
    """Find a stationary webcam inset across sampled frames and return the split crop plan, or None."""
    if not samples or src_w <= src_h:
        return None
    frames = [_normalize_frame(s, src_w, src_h) for s in samples]
    readable = len(frames)

    face_candidates, human_candidates = [], []
    for fi, fr in enumerate(frames):
        # Collect every eligible peripheral face per frame so a higher-confidence NPC face cannot hide the webcam.
        for f in fr["faces"]:
            if f["h"] > src_h * 0.28 or f["w"] > src_w * 0.24:
                continue
            if _is_enclosing_large_person(f, fr["humans"], src_h):
                continue
            cx_n = (f["x"] + f["w"] / 2.0) / src_w
            cy_n = (f["y"] + f["h"] / 2.0) / src_h
            off_center = abs(cx_n - 0.5) >= 0.16
            near_edge = cx_n <= 0.25 or cx_n >= 0.75 or cy_n <= 0.26 or cy_n >= 0.70
            center_tb = cy_n <= 0.26 or cy_n >= 0.70
            small_enough = off_center or (
                f["h"] <= src_h * (0.24 if center_tb else 0.16)
                and f["w"] <= src_w * (0.20 if center_tb else 0.12)
            )
            if (off_center or near_edge) and small_enough:
                face_candidates.append({"fi": fi, "cx": f["x"] + f["w"] / 2.0, "cy": f["y"] + f["h"] / 2.0, "box": f})

        for h in fr["humans"]:
            if h["h"] > src_h * 0.38 or h["w"] > src_w * 0.35:
                continue
            cx_n = (h["x"] + h["w"] / 2.0) / src_w
            off_center = abs(cx_n - 0.5) >= 0.18
            near_edge = (h["x"] <= src_w * 0.15 or h["x"] + h["w"] >= src_w * 0.85
                         or h["y"] <= src_h * 0.15 or h["y"] + h["h"] >= src_h * 0.85)
            if off_center and near_edge:
                human_candidates.append({"fi": fi, "cx": h["x"] + h["w"] / 2.0, "cy": h["y"] + h["h"] / 2.0, "box": h})

    pool = _pick_stationary_cluster(face_candidates, readable, src_w, src_h, is_human=False)
    use_face = bool(pool)
    if not pool:
        pool = _pick_stationary_cluster(human_candidates, readable, src_w, src_h, is_human=True)
    if not pool:
        return None

    med = {k: statistics.median(c["box"][k] for c in pool) for k in ("x", "y", "w", "h")}
    return compute_inset_cam_rect(med, src_w, src_h, top_frac=top_frac, is_human=not use_face)


def vision_binary() -> Path | None:
    """Compile the macOS Apple Vision helper once into the data folder. Returns None off macOS or without swiftc."""
    if sys.platform != "darwin" or not _SWIFT_SRC.is_file():
        return None
    swiftc = "/usr/bin/swiftc" if Path("/usr/bin/swiftc").is_file() else shutil.which("swiftc")
    if not swiftc:
        return None
    out = config.DATA_DIR / "bin" / "streamcut-vision-detect"
    with _COMPILE_LOCK:
        try:
            if out.is_file() and out.stat().st_mtime >= _SWIFT_SRC.stat().st_mtime:
                return out
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            p = subprocess.run([swiftc, "-O", str(_SWIFT_SRC), "-o", str(tmp)],
                               capture_output=True, text=True, timeout=60, creationflags=NO_WINDOW)
            if p.returncode != 0 or not tmp.is_file():
                tmp.unlink(missing_ok=True)
                return None
            tmp.chmod(0o755)
            tmp.replace(out)
            return out
        except Exception:  # noqa: BLE001
            return None


def _yunet_detector(w: int, h: int):
    import cv2
    if not _YUNET_MODEL.is_file():
        return None
    det = getattr(_TLS, "yunet", None)
    if det is None:
        det = cv2.FaceDetectorYN.create(str(_YUNET_MODEL), "", (w, h), score_threshold=0.45, nms_threshold=0.3, top_k=20)
        _TLS.yunet = det
    else:
        det.setInputSize((w, h))
    return det


def _haar_detect(cv2, bgr):
    casc = getattr(_TLS, "haar", None)
    if casc is None:
        base = Path(cv2.data.haarcascades)
        casc = [cv2.CascadeClassifier(str(base / name))
                for name in ("haarcascade_frontalface_default.xml", "haarcascade_profileface.xml")
                if (base / name).is_file()]
        _TLS.haar = casc
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    faces = []
    for cl in casc:
        for (x, y, fw, fh) in cl.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(28, 28)):
            faces.append({"x": float(x), "y": float(y), "w": float(fw), "h": float(fh), "conf": 0.65})
    return faces


def _run_opencv(frame_paths: list[Path]) -> list:
    """Cross-platform face detection via bundled OpenCV YuNet ONNX (Haar fallback if model absent)."""
    try:
        import cv2
    except Exception:  # noqa: BLE001
        return []
    out = []
    for fp in frame_paths:
        bgr = cv2.imread(str(fp))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        faces = []
        try:
            det = _yunet_detector(w, h)
            if det is not None:
                _, raw = det.detect(bgr)
                if raw is not None:
                    for row in raw:
                        faces.append({
                            "x": float(row[0]), "y": float(row[1]),
                            "w": float(row[2]), "h": float(row[3]),
                            "conf": float(row[-1]),
                        })
            else:
                faces = _haar_detect(cv2, bgr)
        except Exception:  # noqa: BLE001
            faces = _haar_detect(cv2, bgr)
        out.append({"path": str(fp), "width": w, "height": h, "faces": faces, "humans": []})
    return out


def run_vision(frame_paths: list[Path]) -> list:
    """Detect faces & upper bodies across frames (Apple Vision on macOS, OpenCV YuNet elsewhere or fallback)."""
    if not frame_paths:
        return []
    exe = vision_binary()
    if exe:
        try:
            p = subprocess.run([str(exe), *(str(x) for x in frame_paths)],
                               capture_output=True, text=True, timeout=20, creationflags=NO_WINDOW)
            if p.returncode == 0:
                data = json.loads(p.stdout or "[]")
                if isinstance(data, list) and data:
                    return data
        except Exception:  # noqa: BLE001
            pass
    return _run_opencv(frame_paths)


def _probe_upscaled_zones(frames: list[Path], probe_dir: Path, src_w: int, src_h: int) -> dict | None:
    """Second-pass zoom scan of peripheral webcam zones (2x upscaled + contrast lift) for tiny/dim webcams."""
    try:
        from PIL import Image, ImageEnhance
    except Exception:  # noqa: BLE001
        return None
    pick_idx = sorted({0, len(frames) // 3, (2 * len(frames)) // 3, len(frames) - 1})[:4]
    zones = [
        ("tl", 0.00, 0.00), ("tr", 0.64, 0.00),
        ("bl", 0.00, 0.62), ("br", 0.64, 0.62),
        ("bc", 0.32, 0.62), ("ml", 0.00, 0.31), ("mr", 0.64, 0.31),
    ]
    z_dir = probe_dir / "zones"
    z_dir.mkdir(exist_ok=True)
    meta, paths = {}, []
    for fi in pick_idx:
        with Image.open(frames[fi]) as im:
            fw, fh = im.size
            zw, zh = max(64, int(round(fw * 0.36))), max(64, int(round(fh * 0.38)))
            for code, rx, ry in zones:
                ox, oy = int(round(fw * rx)), int(round(fh * ry))
                crop = im.crop((ox, oy, min(fw, ox + zw), min(fh, oy + zh)))
                up = ImageEnhance.Contrast(crop.resize((crop.width * 2, crop.height * 2), Image.LANCZOS)).enhance(1.18)
                out = z_dir / f"f{fi}_{code}.jpg"
                up.save(out, quality=92)
                paths.append(out)
                meta[str(out)] = (fi, ox, oy, fw, fh)
    dets = run_vision(paths)
    by_frame = {fi: {"width": 0, "height": 0, "faces": [], "humans": []} for fi in pick_idx}
    for d in dets:
        info = meta.get(d.get("path"))
        if not info:
            continue
        fi, ox, oy, fw, fh = info
        by_frame[fi]["width"], by_frame[fi]["height"] = fw, fh
        for key in ("faces", "humans"):
            for b in d.get(key) or []:
                by_frame[fi][key].append({
                    "x": ox + b["x"] / 2.0, "y": oy + b["y"] / 2.0,
                    "w": b["w"] / 2.0, "h": b["h"] / 2.0, "conf": b.get("conf", 0.7),
                })
    samples = [v for v in by_frame.values() if v["width"] > 0]
    return detect_webcam_inset(samples, src_w, src_h) if samples else None


def plan_gamer_split(src: Path, start: float, end: float, src_w: int, src_h: int, cam_pos: str = "auto") -> dict | None:
    """Resolve top gameplay + bottom facecam crop rectangles for a landscape clip, or None if no cam can be found."""
    if src_w <= 0 or src_h <= 0 or src_w <= src_h:
        return None
    if cam_pos in CORNER_PRESETS:
        return corner_preset_rect(src_w, src_h, cam_pos)

    dur = max(0.5, float(end) - float(start))
    n_samples = max(6, min(14, round(dur / 1.5)))
    fps = n_samples / max(1.0, dur)
    with tempfile.TemporaryDirectory(prefix="pkcam_") as tmp:
        probe_dir = Path(tmp)
        cmd = [
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-ss", f"{max(0.0, float(start)):.3f}", "-t", f"{dur:.3f}", "-i", str(src),
            "-vf", f"fps={fps:.4f},scale='min(1280,iw)':-2",
            "-frames:v", str(n_samples), "-q:v", "4", str(probe_dir / "f_%03d.jpg"),
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=25, creationflags=NO_WINDOW)
            if p.returncode != 0:
                return None
        except Exception:  # noqa: BLE001
            return None
        frames = sorted(probe_dir.glob("f_*.jpg"))
        if not frames:
            return None
        detections = run_vision(frames)
        if detections:
            plan = detect_webcam_inset(detections, src_w, src_h)
            if plan:
                return plan
        if vision_binary() and _YUNET_MODEL.is_file():
            # Backup cross-platform YuNet pass in case Vision skipped dim/tilted cams
            cv_dets = _run_opencv(frames)
            if cv_dets:
                plan = detect_webcam_inset(cv_dets, src_w, src_h)
                if plan:
                    return plan
        return _probe_upscaled_zones(frames, probe_dir, src_w, src_h)
