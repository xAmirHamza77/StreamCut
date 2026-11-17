"""Turns a time range into a vertical clip: 9:16 layout, bouncy word captions, hook title, cover."""
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import config, face_layout, roman
from .media import NO_WINDOW, run

W, H = config.OUT_W, config.OUT_H

# ASS colours are &HBBGGRR. `size` is the ASS font size (line height); _Font turns it into real pixels.
STYLES = {
    "bounce": {"label": "Bounce", "font": "Poppins ExtraBold", "size": 132, "base": "FFFFFF", "active": "00F0FF",
               "strong": "FFFFFF", "outline": 7, "pop": 112, "upper": True, "per": 3},
    "hype": {"label": "Hype", "font": "Anton", "size": 152, "base": "FFFFFF", "active": "3C3CFF",
             "strong": "00D7FF", "outline": 7, "pop": 114, "upper": True, "per": 2},
    "clean": {"label": "Clean", "font": "Poppins ExtraBold", "size": 100, "base": "FFFFFF", "active": "6BD7FF",
              "strong": "FFFFFF", "outline": 5, "pop": 106, "upper": False, "per": 4},
}
HOOK = {"font": "Poppins ExtraBold", "size": 96, "ink": "14100E", "box": "FFFFFF", "pad_x": 36, "pad_y": 22, "radius": 26,
        "max_w": 900}
FONT_FILES = {"Poppins ExtraBold": "Poppins-ExtraBold.ttf", "Anton": "Anton-Regular.ttf", "Archivo Black": "ArchivoBlack-Regular.ttf"}
CAP_MAX_W = 960   # captions never run wider than this, whatever is said


class _Font:
    """Measures text exactly as the renderer will draw it, so every word can be placed by hand: the highlighted
    word can grow without pushing its neighbours, and the heading box fits its text."""
    _cache: dict = {}

    def __init__(self, name):
        import struct
        path = Path(__file__).resolve().parent / "fonts" / FONT_FILES.get(name, FONT_FILES["Poppins ExtraBold"])
        d = path.read_bytes()
        tables = {}
        for i in range(struct.unpack(">H", d[4:6])[0]):
            tag, _, off, _ = struct.unpack(">4sIII", d[12 + 16 * i: 28 + 16 * i])
            tables[tag] = off
        self.upm = struct.unpack(">H", d[tables[b"head"] + 18: tables[b"head"] + 20])[0]
        o = tables[b"OS/2"]
        self.asc, self.desc = struct.unpack(">HH", d[o + 74: o + 78])      # what the renderer sizes the font by
        self.cap = struct.unpack(">h", d[o + 88: o + 90])[0] or int(self.upm * 0.7)
        self.path = str(path)
        self._pil = {}

    @classmethod
    def get(cls, name):
        if name not in cls._cache:
            cls._cache[name] = cls(name)
        return cls._cache[name]

    def em(self, size):
        return size * self.upm / (self.asc + self.desc)

    def width(self, text, size):
        """Pixels wide at this ASS size."""
        try:
            from PIL import ImageFont
            if "f" not in self._pil:
                self._pil["f"] = ImageFont.truetype(self.path, 200)
            return self._pil["f"].getlength(text) * self.em(size) / 200
        except Exception:  # noqa: BLE001 - no measuring library: a fair estimate keeps clips rendering
            return len(text) * self.em(size) * 0.62

    def cap_shift(self, size):
        """How far below the anchor the middle of a capital letter sits when text is centred on a point."""
        return ((self.asc - self.desc) / 2 - self.cap / 2) * size / (self.asc + self.desc)


CAPTION_LEAD = 0.0  # measured: word stamps are within 0.15 s of the voice, so no shift is needed
def install_fonts():
    """Copy the bundled caption fonts next to the data, so captions look the same on every machine and server."""
    src = Path(__file__).resolve().parent / "fonts"
    for f in src.glob("*.ttf"):
        if not (config.FONTS_DIR / f.name).exists():
            shutil.copyfile(f, config.FONTS_DIR / f.name)


LAYOUTS = {
    "blur": "Full frame + blurred fill",
    "zoom": "Zoomed in + blurred fill",
    "crop": "Fill the screen (centre crop)",
    "gamer": "Gamer (gameplay + facecam at bottom)",
}


def _ts(t):
    t = max(0.0, t)
    h, m = int(t // 3600), int(t % 3600 // 60)
    return f"{h}:{m:02d}:{t % 60:05.2f}"


def _clean(text):
    return re.sub(r"[\x00-\x1f\x7f]+", " ", re.sub(r"[{}\\]", "", str(text))).strip()


def _key(word):
    return re.sub(r"[^a-z0-9']", "", word.lower())


def _groups(words, per):
    out, cur = [], []
    for w in words:
        if cur and (len(cur) >= per or sum(len(x["w"]) for x in cur) + len(w["w"]) > 16
                    or w["s"] - cur[-1]["e"] > 0.45 or cur[-1]["w"][-1:] in ".?!"):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def geometry(src_w, src_h, layout, cam_style="split"):
    """Where the video sits in the 1080x1920 frame, and where hook and captions go."""
    if src_h >= src_w and layout != "crop":
        layout = "crop"  # already vertical or square: nothing to gain from a blurred fill
    if layout == "blur":
        fg_h = min(H, int(round(W * src_h / src_w / 2)) * 2)
    elif layout == "zoom":
        fg_h = 1000
    elif layout == "gamer":
        if cam_style == "center":
            return layout, H, 240, 1290
        fg_h = int(round((H * face_layout.TOP_FRAC) / 2.0)) * 2
        return layout, fg_h, 220, fg_h - 30
    else:
        fg_h = H
    top = (H - fg_h) // 2
    if layout == "crop":
        return layout, fg_h, 300, 1380
    return layout, fg_h, max(230, top - 190), min(1640, top + fg_h + 180)


def _hook_spec(hook: str):
    """Balanced 1-or-2-line measurement for the heading box, shared by ASS and PNG fallback renderers."""
    f, size = _Font.get(HOOK["font"]), HOOK["size"]
    words = hook.split()
    if not words:
        return None
    lines = [" ".join(words)]
    if f.width(lines[0], size) > HOOK["max_w"] and len(words) > 1:      # two lines, split where they come out most even
        cut = min(range(1, len(words)), key=lambda i: max(f.width(" ".join(words[:i]), size), f.width(" ".join(words[i:]), size)))
        lines = [" ".join(words[:cut]), " ".join(words[cut:])]
    widest = max(f.width(l, size) for l in lines)
    if widest > HOOK["max_w"]:
        size = max(40, int(size * HOOK["max_w"] / widest))
        widest = max(f.width(l, size) for l in lines)
    line_h = f.em(size) * 1.16
    bw, bh = widest + 2 * HOOK["pad_x"], line_h * len(lines) + 2 * HOOK["pad_y"]
    return lines, f, size, line_h, bw, bh


def _hook_events(hook, dur, y):
    """The heading: dark text in a white rounded box that fits it, on one line or two balanced ones."""
    spec = _hook_spec(hook)
    if not spec:
        return []
    lines, f, size, line_h, bw, bh = spec
    x0, y0, r = (W - bw) / 2, y - bh / 2, min(HOOK["radius"], bh / 2)
    x1, y1 = x0 + bw, y0 + bh
    n = lambda v: f"{v:.0f}"                                                         # noqa: E731
    box = (f"m {n(x0 + r)} {n(y0)} l {n(x1 - r)} {n(y0)} b {n(x1)} {n(y0)} {n(x1)} {n(y0)} {n(x1)} {n(y0 + r)} "
           f"l {n(x1)} {n(y1 - r)} b {n(x1)} {n(y1)} {n(x1)} {n(y1)} {n(x1 - r)} {n(y1)} "
           f"l {n(x0 + r)} {n(y1)} b {n(x0)} {n(y1)} {n(x0)} {n(y1)} {n(x0)} {n(y1 - r)} "
           f"l {n(x0)} {n(y0 + r)} b {n(x0)} {n(y0)} {n(x0)} {n(y0)} {n(x0 + r)} {n(y0)}")
    span = f"{_ts(0)},{_ts(dur)}"
    ev = [f"Dialogue: 3,{span},Hook,,0,0,0,,{{\\an7\\pos(0,0)\\fad(160,0)\\c&H{HOOK['box']}&\\p1}}{box}{{\\p0}}"]
    for i, line in enumerate(lines):
        cy = y - line_h * (len(lines) - 1) / 2 + i * line_h - f.cap_shift(size)
        ev.append(f"Dialogue: 4,{span},Hook,,0,0,0,,{{\\an5\\pos({W // 2},{cy:.0f})\\fad(160,0)\\fs{size}}}{line}")
    return ev


def _caption_groups(words, start, end, st):
    words = [dict(w, s=w["s"] - CAPTION_LEAD, e=w["e"] - CAPTION_LEAD) for w in words]
    inside = [dict(w, w=_clean(w["w"])) for w in words if w["s"] >= start - 0.05 and w["s"] < end]
    inside = sorted((w for w in inside if w["w"]), key=lambda w: w["s"])
    for i in range(len(inside) - 1):
        inside[i]["e"] = max(inside[i]["s"], min(inside[i]["e"], inside[i + 1]["s"]))
    return _groups(inside, st["per"])


def _group_metrics(texts, st, f):
    size = st["size"]
    gap = f.width("n n", size) - 2 * f.width("n", size) + (2 * st["outline"] + 6) * size / st["size"]
    total = sum(f.width(t, size) for t in texts) + gap * (len(texts) - 1)
    if total > CAP_MAX_W:
        size = max(36, int(size * CAP_MAX_W / total))
        gap = f.width("n n", size) - 2 * f.width("n", size) + (2 * st["outline"] + 6) * size / st["size"]
        total = sum(f.width(t, size) for t in texts) + gap * (len(texts) - 1)
    return size, gap, total


def build_ass(words, start, end, hook, style_name, highlight, hook_y, cap_y) -> str:
    st = STYLES.get(style_name, STYLES["bounce"])
    strong = {_key(h) for h in highlight if _key(h)}
    head = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {W}
PlayResY: {H}
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{st['font']},{st['size']},&H00{st['base']},&H00{st['base']},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,{st['outline']},0,5,0,0,0,1
Style: Hook,{HOOK['font']},{HOOK['size']},&H00{HOOK['ink']},&H00{HOOK['ink']},&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,0,0,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    dur = end - start
    ev = _hook_events(_clean(hook), dur, hook_y)
    f = _Font.get(st["font"])
    groups = _caption_groups(words, start, end, st)
    for gi, g in enumerate(groups):
        g_start = max(0.0, g[0]["s"] - start)
        g_end = min(g[-1]["e"] + 0.25, groups[gi + 1][0]["s"] if gi + 1 < len(groups) else end) - start
        if g_end - g_start < 0.02:
            continue
        texts = []
        for x in g:
            t = re.sub(r"[.,;:]+$", "", x["w"])
            texts.append(t.upper() if st["upper"] else t)
        if not all(roman.drawable(t) for t in texts):
            # A script the caption font cannot draw (Japanese, Thai, Chinese...): its words cannot be measured, so
            # they are not placed one by one. The line is drawn whole and centred, in a font the system has for it.
            line = " ".join(texts)
            fs = max(44, min(int(st["size"] * 0.7), int(1500 / max(1, len(line)) * 1.6)))
            ev.append(f"Dialogue: 1,{_ts(g_start)},{_ts(g_end)},Cap,,60,60,0,,{{\\an5\\pos({W // 2},{cap_y:.0f})\\q0\\fs{fs}}}{line}")
            continue
        size, gap, total = _group_metrics(texts, st, f)
        cy = cap_y - f.cap_shift(size)
        x = (W - total) / 2
        for i, (w, t) in enumerate(zip(g, texts)):
            wd = f.width(t, size)
            p = max(104, min(st["pop"], 100 + int(2200 / max(wd, 1))))   # long words grow less, so they never touch their neighbours
            pos = f"\\an5\\pos({x + wd / 2:.0f},{cy:.0f})\\fs{size}"
            x += wd + gap
            a = max(g_start, w["s"] - start)
            b = (g[i + 1]["s"] - start) if i + 1 < len(g) else g_end
            base = st["strong"] if _key(w["w"]) in strong else st["base"]
            if a - g_start >= 0.02:                                                   # waiting to be said
                ev.append(f"Dialogue: 1,{_ts(g_start)},{_ts(a)},Cap,,0,0,0,,{{{pos}\\c&H{base}&}}{t}")
            if b - a >= 0.02:                                                         # being said: colour and a quick pop, in front
                ev.append(f"Dialogue: 2,{_ts(a)},{_ts(b)},Cap,,0,0,0,,{{{pos}\\c&H{st['active']}&\\fscx94\\fscy94"
                          f"\\t(0,90,\\fscx{p}\\fscy{p})\\t(90,200,\\fscx{p - 5}\\fscy{p - 5})}}{t}")
            if g_end - b >= 0.02:                                                     # said
                ev.append(f"Dialogue: 1,{_ts(b)},{_ts(g_end)},Cap,,0,0,0,,{{{pos}\\c&H{base}&\\fscx{p - 5}\\fscy{p - 5}"
                          f"\\t(0,90,\\fscx100\\fscy100)}}{t}")
    return head + "\n".join(ev) + "\n"


def _filter(layout, fg_h, split_plan=None, cam_style="split"):
    if layout == "gamer" and split_plan:
        g, c = split_plan["game"], split_plan["cam"]
        if cam_style == "center":
            return (f"[0:v]fps={config.OUT_FPS},split=2[g][c];"
                    f"[g]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[bg];"
                    f"[c]crop={c['w']}:{c['h']}:{c['x']}:{c['y']},scale=520:346:flags=lanczos,pad=528:354:4:4:color=0x111310,setsar=1[cam];"
                    f"[bg][cam]overlay=(W-w)/2:1430,setsar=1[v0];")
        return (f"[0:v]fps={config.OUT_FPS},split=2[g][c];"
                f"[g]crop={g['w']}:{g['h']}:{g['x']}:{g['y']},scale={W}:{split_plan['top_h']}:flags=lanczos,setsar=1[top];"
                f"[c]crop={c['w']}:{c['h']}:{c['x']}:{c['y']},scale={W}:{split_plan['bot_h']}:flags=lanczos,setsar=1[bot];"
                f"[top][bot]vstack=inputs=2,setsar=1[v0];")
    if layout == "crop":
        return f"[0:v]fps={config.OUT_FPS},scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},setsar=1[v0];"
    return (f"[0:v]fps={config.OUT_FPS},split=2[a][b];"
            f"[a]scale=270:480:force_original_aspect_ratio=increase,crop=270:480,boxblur=6:2,"
            f"scale={W}:{H},eq=brightness=-0.14:saturation=1.15[bg];"
            f"[b]scale={W}:{fg_h}:force_original_aspect_ratio=increase,crop={W}:{fg_h}[fg];"
            f"[bg][fg]overlay=0:(H-h)/2,setsar=1[v0];")


_qsv_ok = None
_ass_ok = None
RENDER_SECONDS = 600      # one clip's video encode; far beyond normal, so a frozen encoder is stopped, not waited on


def _has_ass_filter() -> bool:
    """True when this machine's ffmpeg includes the libass subtitle filter."""
    global _ass_ok
    if _ass_ok is None:
        try:
            p = subprocess.run(["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True,
                               timeout=5, creationflags=NO_WINDOW)
            _ass_ok = bool(re.search(r"^\s*[T.S]+\s+ass\s", p.stdout, re.M))
        except Exception:  # noqa: BLE001
            _ass_ok = False
    return _ass_ok


def _bgr_rgba(hex_bgr: str) -> tuple[int, int, int, int]:
    return (int(hex_bgr[4:6], 16), int(hex_bgr[2:4], 16), int(hex_bgr[0:2], 16), 255)


def _overlay_assets(work: Path, words: list, start: float, end: float, clip: dict, hook_y: int, cap_y: int):
    """Fallback overlay PNG generator when ffmpeg lacks libass (e.g. Homebrew minimal builds on macOS)."""
    from PIL import Image, ImageDraw, ImageFont

    ov_dir = work / "ov"
    ov_dir.mkdir(exist_ok=True)
    in_args, steps = [], []
    in_idx = 1

    hook = _clean(clip.get("hook", ""))
    spec = _hook_spec(hook)
    if spec:
        lines, f, size, line_h, bw, bh = spec
        pw, ph = int(math.ceil(bw)) + 4, int(math.ceil(bh)) + 4
        im = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
        dr = ImageDraw.Draw(im)
        r = min(HOOK["radius"], int(bh // 2))
        dr.rounded_rectangle([1, 1, pw - 2, ph - 2], radius=r, fill=_bgr_rgba(HOOK["box"]))
        pil_f = ImageFont.truetype(f.path, max(12, int(round(f.em(size)))))
        ink = _bgr_rgba(HOOK["ink"])
        for i, line in enumerate(lines):
            ly = ph / 2 - line_h * (len(lines) - 1) / 2 + i * line_h
            dr.text((pw / 2, ly), line, font=pil_f, fill=ink, anchor="mm")
        im.save(ov_dir / "hook.png")
        in_args += ["-i", "ov/hook.png"]
        hy = max(0, min(H - ph, int(round(hook_y - ph / 2))))
        steps.append((in_idx, f"x=(W-w)/2:y={hy}:eof_action=repeat"))
        in_idx += 1

    st = STYLES.get(clip.get("style", "bounce"), STYLES["bounce"])
    f = _Font.get(st["font"])
    strong = {_key(h) for h in clip.get("highlight_words", []) if _key(h)}
    groups = _caption_groups(words, start, end, st)
    band_h = 220
    cy_local = band_h // 2
    base_rgba, active_rgba, strong_rgba = _bgr_rgba(st["base"]), _bgr_rgba(st["active"]), _bgr_rgba(st["strong"])
    plate_idx = 0

    for gi, g in enumerate(groups):
        g_start = max(0.0, g[0]["s"] - start)
        g_end = min(g[-1]["e"] + 0.25, groups[gi + 1][0]["s"] if gi + 1 < len(groups) else end) - start
        if g_end - g_start < 0.02:
            continue
        texts = [(re.sub(r"[.,;:]+$", "", x["w"]).upper() if st["upper"] else re.sub(r"[.,;:]+$", "", x["w"])) for x in g]
        if not all(roman.drawable(t) for t in texts):
            continue
        size, gap, total = _group_metrics(texts, st, f)
        pil_cap = ImageFont.truetype(f.path, max(14, int(round(f.em(size)))))
        stroke_w = max(2, int(round(st["outline"] * size / st["size"])))
        widths = [f.width(t, size) for t in texts]
        xs, cur_x = [], (W - total) / 2
        for wd in widths:
            xs.append(cur_x + wd / 2)
            cur_x += wd + gap

        for ai, (w, _) in enumerate(zip(g, texts)):
            a = max(g_start, w["s"] - start)
            b = (g[ai + 1]["s"] - start) if ai + 1 < len(g) else g_end
            if b - a < 0.02:
                continue
            im = Image.new("RGBA", (W, band_h), (0, 0, 0, 0))
            dr = ImageDraw.Draw(im)
            for wi, (w0, t0) in enumerate(zip(g, texts)):
                fill = active_rgba if wi == ai else (strong_rgba if _key(w0["w"]) in strong else base_rgba)
                dr.text((xs[wi], cy_local), t0, font=pil_cap, fill=fill, anchor="mm",
                        stroke_width=stroke_w, stroke_fill=(0, 0, 0, 255))
            rel = f"ov/c_{plate_idx:04d}.png"
            im.save(work / rel)
            in_args += ["-i", rel]
            steps.append((in_idx, f"x=0:y={max(0, min(H - band_h, int(round(cap_y - band_h / 2))))}:eof_action=repeat:enable='between(t,{a:.3f},{b:.3f})'"))
            in_idx += 1
            plate_idx += 1

    if not steps:
        return [], "", "[v0]"
    chain, prev = [], "[v0]"
    for idx, (inp_i, opts) in enumerate(steps):
        nxt = "[v]" if idx == len(steps) - 1 else f"[vo{idx}]"
        chain.append(f"{prev}[{inp_i}:v]overlay={opts}{nxt}")
        prev = nxt
    return in_args, ";".join(chain), prev


def render_clip(src: Path, out_dir: Path, name: str, clip: dict, words: list, src_w: int, src_h: int) -> dict:
    """Writes <name>.mp4, <name>.jpg and <name>.ass into out_dir. Returns file names."""
    out_dir.mkdir(parents=True, exist_ok=True)
    start, end = clip["start"], clip["end"]
    cam_style = clip.get("cam_style", "split")
    if cam_style not in face_layout.CAM_STYLES:
        cam_style = "split"
    layout, fg_h, hook_y, cap_y = geometry(src_w, src_h, clip.get("layout", "blur"), cam_style)
    split_plan = None
    if layout == "gamer":
        split_plan = face_layout.plan_gamer_split(src, start, end, src_w, src_h, clip.get("cam_pos", "auto"))
        if not split_plan:
            layout, fg_h, hook_y, cap_y = geometry(src_w, src_h, "crop")
    ass = build_ass(words, start, end, clip.get("hook", ""), clip.get("style", "bounce"),
                    clip.get("highlight_words", []), hook_y, cap_y)
    # The clip is made in a short work folder and moved into place afterwards. The caption step of ffmpeg cannot
    # open files whose full path is longer than 260 characters on Windows, which a data folder deep inside other
    # folders easily reaches. Inside the work folder every name is short and relative (a drive colon would also
    # break the filter syntax).
    work = Path(tempfile.mkdtemp(prefix="pk"))
    try:
        out = _render_in(work, src, out_dir, name, clip, words, hook_y, cap_y, layout, fg_h, ass, split_plan, cam_style)
        if split_plan:
            out["cam_corner"] = split_plan["corner"]
            out["cam_style"] = cam_style
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _render_in(work: Path, src: Path, out_dir: Path, name: str, clip: dict, words: list, hook_y: int, cap_y: int,
               layout, fg_h, ass: str, split_plan=None, cam_style="split") -> dict:
    global _qsv_ok
    start, end = clip["start"], clip["end"]
    (work / f"{name}.ass").write_text(ass, encoding="utf-8")
    (work / "f").mkdir()
    for f in config.FONTS_DIR.glob("*.ttf"):
        shutil.copyfile(f, work / "f" / f.name)
    base_fc = _filter(layout, fg_h, split_plan, cam_style)
    if _has_ass_filter():
        ov_inputs = []
        fc = base_fc + f"[v0]ass={name}.ass:fontsdir=f[v]"
        v_map = "[v]"
    else:
        ov_inputs, ov_fc, v_map = _overlay_assets(work, words, start, end, clip, hook_y, cap_y)
        fc = base_fc + ov_fc if ov_fc else base_fc.rstrip(";")
    base = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.2f}", "-t", f"{end - start:.2f}", "-i", str(src),
            *ov_inputs, "-filter_complex", fc, "-map", v_map, "-map", "0:a:0", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11",
            "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-movflags", "+faststart", "-pix_fmt"]
    encoders = []
    if _qsv_ok is not False:
        encoders.append(("qsv", ["nv12", "-c:v", "h264_qsv", "-global_quality", "23", "-preset", "fast"]))
    encoders.append(("x264", ["yuv420p", "-c:v", "libx264", "-crf", "21", "-preset", "veryfast"]))
    err = None
    for label, enc in encoders:
        p = subprocess.run(base + enc + [f"{name}.mp4"], cwd=work, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", creationflags=NO_WINDOW, timeout=RENDER_SECONDS)
        if p.returncode == 0:
            if label == "qsv":
                _qsv_ok = True
            break
        err = p.stderr[-1200:]
        if label == "qsv":
            _qsv_ok = False
    else:
        raise RuntimeError(f"Render failed: {err}")
    cover_t = min(1.2, (end - start) / 2)
    run(["ffmpeg", "-y", "-v", "error", "-ss", f"{cover_t:.2f}", "-i", f"{name}.mp4", "-frames:v", "1",
         "-q:v", "3", f"{name}.jpg"], cwd=work)
    for ext in ("mp4", "jpg", "ass"):
        shutil.move(str(work / f"{name}.{ext}"), str(out_dir / f"{name}.{ext}"))
    return {"file": f"{name}.mp4", "cover": f"{name}.jpg", "layout": layout}
