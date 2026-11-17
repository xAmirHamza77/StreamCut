"""Gamer facecam-at-bottom layout, cam/face detection, render output, and creator settings tests."""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ["STREAMCUT_DATA"] = tempfile.mkdtemp(prefix="pc-gamer-")
os.environ.update(DATABASE_URL="", STREAMCUT_ACCOUNTS="0", STREAMCUT_FETCH_MODEL="0", STREAMCUT_PUBLIC_URL="http://127.0.0.1:8765")
sys.path.insert(0, ".")

from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

from studio import auth, brain, config, creators, ctx, db, face_layout, jobs, live, media, render, server  # noqa: E402

db.init()
render.install_fonts()
fails = []


def check(name, got, want=True):
    ok = want(got) if callable(want) else got == want
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"   got {got!r}"))
    if not ok:
        fails.append(name)


# ---- 1. Pure detection & geometry tests (matching clipship stationary webcam classifier)
bl_samples = [
    {"width": 960, "height": 540, "faces": [{"x": 70, "y": 400, "w": 68, "h": 76, "conf": 0.92}], "humans": []}
    for _ in range(8)
]
plan = face_layout.detect_webcam_inset(bl_samples, 1920, 1080)
check("stationary bottom-left facecam detected & scaled from 960x540 to 1920x1080",
      bool(plan) and plan["corner"] == "bl" and plan["top_h"] == 1200 and plan["bot_h"] == 720)
check("gameplay crop preserves 1080:1200 aspect inside 1920x1080",
      bool(plan) and plan["game"] == {"x": 474, "y": 0, "w": 972, "h": 1080})
check("webcam crop fits source bounds with even 3:2 ratio",
      bool(plan) and plan["cam"]["x"] % 2 == 0 and plan["cam"]["w"] % 2 == 0 and plan["cam"]["h"] % 2 == 0
      and 1.45 <= plan["cam"]["w"] / plan["cam"]["h"] <= 1.55
      and plan["cam"]["x"] + plan["cam"]["w"] <= 1920 and plan["cam"]["y"] + plan["cam"]["h"] <= 1080)

# Full-room person enclosing a face must NOT trigger webcam inset split
talking_head = [
    {"width": 1920, "height": 1080,
     "faces": [{"x": 890, "y": 180, "w": 150, "h": 180, "conf": 0.95}],
     "humans": [{"x": 620, "y": 140, "w": 680, "h": 920, "conf": 0.9}]}
    for _ in range(6)
]
check("full-room talking head is ignored as a corner webcam", face_layout.detect_webcam_inset(talking_head, 1920, 1080), None)

# Upper-body fallback near right edge when tiny face is obscured
upper_only = [
    {"width": 1920, "height": 1080, "faces": [],
     "humans": [{"x": 1550, "y": 720, "w": 320, "h": 340, "conf": 0.85}]}
    for _ in range(6)
]
uplan = face_layout.detect_webcam_inset(upper_only, 1920, 1080)
check("upper-body fallback locates bottom-right corner cam", bool(uplan) and uplan["corner"] == "br")

# Vertical source returns None
check("vertical video never attempts landscape webcam split",
      face_layout.detect_webcam_inset(bl_samples, 1080, 1920), None)

# Corner presets cover all 8 perimeter locations including bottom-center (bc)
presets = {k: face_layout.corner_preset_rect(1920, 1080, k) for k in ("bl", "br", "tl", "tr", "bc", "tc", "ml", "mr")}
check("corner presets place bl/br/tl/tr/bc in distinct locations",
      presets["bl"]["cam"]["x"] < presets["bc"]["cam"]["x"] < presets["br"]["cam"]["x"]
      and presets["tl"]["cam"]["y"] < presets["ml"]["cam"]["y"] < presets["bl"]["cam"]["y"]
      and presets["tc"]["cam"]["y"] < presets["bc"]["cam"]["y"])

# Bottom-center facecam detected even when horizontally centered
bc_samples = [
    {"width": 1920, "height": 1080, "faces": [{"x": 850, "y": 820, "w": 220, "h": 210, "conf": 0.9}], "humans": []}
    for _ in range(6)
]
bc_plan = face_layout.detect_webcam_inset(bc_samples, 1920, 1080)
check("bottom-center stream facecam detected cleanly", bool(bc_plan) and bc_plan["corner"] == "bc")

# Geometry coordinates for gamer split vs bottom-center card
g_lay, g_h, hook_y, cap_y = render.geometry(1920, 1080, "gamer", "split")
c_lay, c_h, c_hy, c_cy = render.geometry(1920, 1080, "gamer", "center")
check("gamer geometry supports split bar and bottom center card",
      (g_lay, g_h, hook_y, cap_y) == ("gamer", 1200, 220, 1170)
      and (c_lay, c_h, c_hy, c_cy) == ("gamer", 1920, 240, 1290))

# ---- 2. macOS Vision helper smoke test (if running on macOS with swiftc)
if sys.platform == "darwin" and Path("/usr/bin/swiftc").is_file():
    exe = face_layout.vision_binary()
    check("Apple Vision binary compiles cleanly on macOS", bool(exe) and exe.is_file())
    if exe:
        img_p = Path(config.DATA_DIR) / "probe.jpg"
        Image.new("RGB", (640, 360), (20, 40, 60)).save(img_p)
        res = face_layout.run_vision([img_p])
        check("Apple Vision binary returns structured detections JSON",
              len(res) == 1 and res[0]["width"] == 640 and "faces" in res[0] and "humans" in res[0])

# ---- 3. Real ffmpeg render test: top green gameplay + bottom-left blue facecam -> stacked 1080x1920
tmp_work = Path(config.DATA_DIR) / "render-test"
tmp_work.mkdir(parents=True, exist_ok=True)
src_vid = tmp_work / "stream.mp4"
subprocess.run(
    [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", "color=c=#18B848:s=640x360:r=30:d=1.5",
        "-f", "lavfi", "-i", "color=c=#1E66FF:s=160x106:r=30:d=1.5",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=1.5",
        "-filter_complex", "[0:v][1:v]overlay=10:246[v];[2:a]volume=0.2[a]",
        "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest", str(src_vid),
    ],
    check=True,
)
words = [{"w": "Clutch", "s": 0.2, "e": 0.6}, {"w": "win", "s": 0.7, "e": 1.1}]
out_dir = tmp_work / "out"
res_bl = render.render_clip(src_vid, out_dir, "gamer_bl",
                            {"start": 0.0, "end": 1.2, "hook": "INSANE", "style": "bounce", "layout": "gamer", "cam_pos": "bl"},
                            words, 640, 360)
info = media.probe(out_dir / res_bl["file"])
check("rendered gamer bl clip is 1080x1920", (res_bl["layout"], res_bl.get("cam_corner"), info["width"], info["height"]) == ("gamer", "bl", 1080, 1920))

cover_img = Image.open(out_dir / res_bl["cover"]).convert("RGB")
top_px = cover_img.getpixel((540, 500))    # green gameplay area
bot_px = cover_img.getpixel((540, 1560))   # blue bottom-left cam area
check("rendered frame puts green gameplay on top and blue facecam at the bottom",
      top_px[1] > top_px[0] + 60 and top_px[1] > top_px[2] + 60
      and bot_px[2] > bot_px[0] + 80 and bot_px[2] > bot_px[1] + 40)

res_auto = render.render_clip(src_vid, out_dir, "gamer_auto_nofaces",
                              {"start": 0.0, "end": 1.2, "hook": "TEST", "style": "clean", "layout": "gamer", "cam_pos": "auto"},
                              words, 640, 360)
info_auto = media.probe(out_dir / res_auto["file"])
check("auto mode without detected faces falls back gracefully to 1080x1920 crop",
      (res_auto["layout"], info_auto["width"], info_auto["height"]) == ("crop", 1080, 1920))

# Auto mode WITH a detected face in extracted frames splits gameplay top + facecam bottom
orig_rv = face_layout.run_vision
face_layout.run_vision = lambda frames: [
    {"path": str(f), "width": 640, "height": 360,
     "faces": [{"x": 45, "y": 265, "w": 55, "h": 62, "conf": 0.94}], "humans": []}
    for f in frames
]
try:
    res_det = render.render_clip(src_vid, out_dir, "gamer_auto_detected",
                                 {"start": 0.0, "end": 1.2, "hook": "DETECTED", "style": "hype", "layout": "gamer", "cam_pos": "auto"},
                                 words, 640, 360)
finally:
    face_layout.run_vision = orig_rv
det_img = Image.open(out_dir / res_det["cover"]).convert("RGB")
det_bot = det_img.getpixel((540, 1560))
check("auto mode with detected bottom-left face renders 1080x1920 gamer split",
      res_det["layout"] == "gamer" and res_det.get("cam_corner") == "bl" and det_bot[2] > det_bot[0] + 80)

# Bottom center card mode overlays blue cam centered at bottom over full green gameplay
res_ctr = render.render_clip(src_vid, out_dir, "gamer_center",
                             {"start": 0.0, "end": 1.2, "hook": "CENTER", "style": "bounce", "layout": "gamer", "cam_pos": "bl", "cam_style": "center"},
                             words, 640, 360)
ctr_img = Image.open(out_dir / res_ctr["cover"]).convert("RGB")
ctr_mid = ctr_img.getpixel((540, 1600))     # centered facecam box
ctr_left = ctr_img.getpixel((100, 1600))    # surrounding green gameplay beside centered box
ctr_top = ctr_img.getpixel((540, 700))      # green gameplay above
check("bottom center card places blue facecam in bottom-center over green gameplay",
      res_ctr["layout"] == "gamer" and res_ctr.get("cam_style") == "center"
      and ctr_mid[2] > ctr_mid[0] + 80 and ctr_left[1] > ctr_left[0] + 60 and ctr_top[1] > ctr_top[0] + 60)

# ---- 4. API & creator persistence checks
BASE = "http://127.0.0.1:8765"
HDR = {"Origin": BASE}
client = TestClient(server.app, base_url=BASE)
st = client.get("/api/status").json()
check("/api/status advertises gamer layout, cam_positions (with bc), and cam_styles (with center)",
      "gamer" in (st.get("layouts") or {})
      and {"auto", "bl", "br", "tl", "tr", "bc", "tc", "ml", "mr"} <= set(st.get("cam_positions") or {})
      and set(st.get("cam_styles") or {}) == {"split", "center"})

created = client.post("/api/creators", json={"name": "ShroudTest", "layout": "gamer", "cam_pos": "bc", "cam_style": "center"}, headers=HDR).json()
check("creator workspace saves gamer layout, bottom-center cam_pos, and bottom center card cam_style",
      created["layout"] == "gamer" and created["cam_pos"] == "bc" and created["cam_style"] == "center")
updated = client.put(f"/api/creators/{created['id']}", json={"cam_pos": "invalid", "cam_style": "bogus"}, headers=HDR).json()
check("invalid cam_pos and cam_style normalize cleanly to auto and split",
      updated["cam_pos"] == "auto" and updated["cam_style"] == "split")
updated2 = client.put(f"/api/creators/{created['id']}", json={"layout": "gamer", "cam_pos": "tr", "cam_style": "center"}, headers=HDR).json()

with ctx.as_user(auth.local_user().id):
    job = jobs.create({"type": "file", "name": "vod.mp4"}, {}, enqueue=False, creator=created["id"])
check("jobs inherit gamer layout, cam_pos, and cam_style from creator workspace",
      job["options"]["layout"] == "gamer" and job["options"]["cam_pos"] == "tr" and job["options"]["cam_style"] == "center")

app_html = client.get("/app").text
check("app UI includes gamer layout preview, bottom center card, and webcam controls",
      "creatorPreview" in app_html and "Show facecam" in app_html and "pv-pip" in app_html and "lay-prev.gamer" in app_html)
check("app UI includes StreamCut theme variables, branding, persistence, toggle buttons, and Appearance settings",
      '[data-theme="dark"]' in app_html and "--on-brand" in app_html and "streamcut_theme" in app_html
      and "StreamCut" in app_html and "#D5FF40" in app_html and "/brand/favicon.svg?v=sc2" in app_html
      and "/brand/icon.png?v=sc2" in app_html and "/brand/logo-light.png?v=sc2" in app_html
      and 'data-act="theme"' in app_html and 'data-act="thememode"' in app_html)

# ---- 5. Live recording & manual clip options tests
streamer = client.post("/api/creators", json={"name": "LiveGamer", "channel": "https://www.twitch.tv/livegamer_test", "layout": "blur"}, headers=HDR).json()
orig_start, orig_submit_live, orig_desc = live.Session.start, jobs.submit_live, brain.describe
live.Session.start = lambda self: live.SESSIONS.__setitem__(self.job["id"], self)
submitted = []
jobs.submit_live = lambda fn: submitted.append(fn)
brain.describe = lambda text, title="", up="": {"hook": "H", "title": "T", "description": "D", "hashtags": []}
try:
    rj = client.post(f"/api/creators/{streamer['id']}/live",
                     json={"options": {"layout": "gamer", "cam_pos": "br", "cam_style": "center", "style": "hype", "min_len": 12, "max_len": 40}},
                     headers=HDR).json()
    sess = live.SESSIONS[rj["id"]]
    check("starting live recording applies selected Options over creator default",
          rj["options"]["layout"] == "gamer" and sess.settings["layout"] == "gamer"
          and sess.settings["cam_pos"] == "br" and sess.settings["cam_style"] == "center"
          and sess.settings["style"] == "hype"
          and sess.settings["min_len"] == 12 and sess.settings["max_len"] == 40)

    upd = client.post(f"/api/live/{rj['id']}/options",
                      json={"layout": "crop", "cam_pos": "tl", "cam_style": "split", "style": "clean"}, headers=HDR).json()
    check("updating live options updates active session immediately",
          upd["options"]["layout"] == "crop" and sess.settings["layout"] == "crop" and sess.settings["cam_style"] == "split" and sess.settings["style"] == "clean")

    sess.samples = 16000 * 60
    sess.t_done = 60.0
    clip_resp = client.post(f"/api/live/{rj['id']}/clip",
                            json={"seconds": 30, "layout": "gamer", "cam_pos": "bc", "cam_style": "center", "style": "bounce"},
                            headers=HDR)
    sess._manual_step()
    c_made = sess.job["clips"][-1]
    check("manual live clip applies selected layout/cam_pos/cam_style/style overrides",
          clip_resp.status_code == 200 and c_made["layout"] == "gamer"
          and c_made["cam_pos"] == "bc" and c_made["cam_style"] == "center" and c_made["style"] == "bounce")

    sess._add_clip({"start": 5.0, "end": 20.0, "score": 88, "reason": "AI spike", "hook": "WOW", "title": "AI Clip", "description": ""})
    c_auto = sess.job["clips"][-1]
    check("auto live clip uses active session settings",
          c_auto["layout"] == "crop" and c_auto["cam_pos"] == "tl" and c_auto["style"] == "clean")
    live.SESSIONS.pop(rj["id"], None)
finally:
    live.Session.start, jobs.submit_live, brain.describe = orig_start, orig_submit_live, orig_desc

# ---- 6. Cross-platform YuNet detector + multi-face NPC distractor + top-corner hug tests
npc_distractor_samples = [
    {"width": 1024, "height": 576,
     "faces": [
         {"x": 660 if i < 4 else 690, "y": 130, "w": 120, "h": 145, "conf": 0.96},  # higher-conf moving NPC
         {"x": 96, "y": 310, "w": 68, "h": 80, "conf": 0.84},                         # stationary mid-left streamer
     ],
     "humans": []}
    for i in range(6)
]
npc_plan = face_layout.detect_webcam_inset(npc_distractor_samples, 1024, 576)
check("stationary peripheral facecam beats higher-confidence NPC face",
      bool(npc_plan) and npc_plan["corner"] == "ml" and npc_plan["cam"]["x"] < 150)

tl_samples = [
    {"width": 1024, "height": 577,
     "faces": [{"x": 112.5, "y": 66.5, "w": 42.5, "h": 53.9, "conf": 0.92}],
     "humans": []}
    for _ in range(6)
]
tl_plan = face_layout.detect_webcam_inset(tl_samples, 1024, 577)
check("top-corner cam hugs top edge to exclude banners underneath",
      bool(tl_plan) and tl_plan["corner"] == "tl" and tl_plan["cam"]["y"] <= 24 and tl_plan["cam"]["y"] + tl_plan["cam"]["h"] <= 152)

probe_blank = Path(config.DATA_DIR) / "probe_cv.jpg"
Image.new("RGB", (640, 360), (15, 25, 35)).save(probe_blank)
cv_res = face_layout._run_opencv([probe_blank])
check("OpenCV YuNet cross-platform detector is bundled & runs cleanly",
      face_layout._YUNET_MODEL.is_file() and len(cv_res) == 1 and cv_res[0]["width"] == 640 and "faces" in cv_res[0])

orig_vb = face_layout.vision_binary
face_layout.vision_binary = lambda: None
try:
    linux_path = face_layout.run_vision([probe_blank])
    check("run_vision works off macOS via OpenCV fallback", len(linux_path) == 1 and linux_path[0]["width"] == 640)
finally:
    face_layout.vision_binary = orig_vb

shutil.rmtree(os.environ["STREAMCUT_DATA"], ignore_errors=True)
print("ALL PASSED" if not fails else f"{len(fails)} failed")
sys.exit(1 if fails else 0)
