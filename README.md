<p align="center">
  <img src="web/brand/logo-light.png" alt="StreamCut" height="64">
</p>

# StreamCut

Turn long videos and live streams into captioned 9:16 vertical clips, ready to publish. Self-hosted and open source — run it on your own computer or Linux server with your own AI API key.

Paste a YouTube, Twitch, or Kick link (or upload a video file) and **StreamCut** transcribes the speech locally, uses your chosen AI model to identify viral-worthy moments, crops gameplay + webcam into vertical 9:16 layouts, burns in animated word-by-word captions and hooks, and publishes or schedules to connected YouTube, Instagram, and TikTok accounts. It can also watch creators automatically and clip live streams while they are still broadcasting.

## Features

- **AI Highlight Discovery**: Locally transcribes speech and combines word timestamps with audio loudness peaks (and live chat spikes when recording) so an AI model picks self-contained clips with hooks, titles, descriptions, hashtags, and highlight words.
- **Live Stream Auto-Clipping**: Continuously records Twitch, Kick, or YouTube Live streams into a rolling buffer (`buffer/`), evaluates speech and reaction peaks every few minutes, supports instant one-click manual hand-clips, and auto-watches creators when they go live.
- **Gamer Facecam Layouts & Auto-Detection**:
  - **Split bar (`split`)**: Gameplay on top (`1080×1200`) stacked over the detected facecam (`1080×720`).
  - **Bottom center card (`center`)**: Full-height gameplay with a framed webcam PiP overlay at bottom center.
  - **Cross-platform neural detection**: Uses Apple Vision on macOS and bundled OpenCV YuNet (`face_detection_yunet_2023mar.onnx`) on Linux/Windows/containers to locate stationary boxed, rounded, mid-side, or background-removed cams while filtering out in-game characters, plus 8 manual perimeter presets (`tl`, `tc`, `tr`, `ml`, `mr`, `bl`, `bc`, `br`).
- **Word-by-Word Animated Captions**: Three built-in typography presets (**Bounce**, **Hype**, **Clean**), automated Romanized Hinglish transliteration for Hindi/Urdu speech, smart sentence-ending snap boundaries, and blurred/cropped/gamer 9:16 layouts.
- **Creator Workspaces**: Independent defaults, channels, upload schedules, activity logs, and layout/aspect previews for each creator you clip.
- **Direct Publishing & Scheduling**: Built-in OAuth flows for YouTube Shorts, Instagram Reels, and TikTok with queued posting intervals and daily per-account caps (see [DEPLOY.md](DEPLOY.md#connecting-youtube-instagram-and-tiktok)).
- **Bring Your Own AI Key (BYOK)**: Supports Gemini, OpenAI, Grok (xAI), Groq, OpenRouter, Claude (Anthropic), local Ollama, or any OpenAI-compatible custom base URL. Also works without any AI key using loudness peak fallback.
- **Obsidian & Lime Dark-First UI**: Clean single-file frontend (`web/index.html`) with Obsidian Dark, Crisp Daylight, and Match Device appearance modes (`Poppins` UI typography, zero external build toolchain).
- **Single-Owner or Multi-User Mode**: Runs straight into the dashboard on `127.0.0.1` by default (`STREAMCUT_ACCOUNTS=0`), or switches on full sign-up, salted scrypt auth, Google Sign-In, SMTP verification, and `/admin` monitoring (`STREAMCUT_ACCOUNTS=1`).

## Quick Start

Requires Python 3.10+ and `ffmpeg`/`ffprobe` (`brew install ffmpeg` on macOS, `sudo apt install ffmpeg` on Linux, `winget install Gyan.FFmpeg` on Windows).

```bash
git clone https://github.com/xAmirHamza77/StreamCut.git
cd StreamCut
pip install -r requirements.txt
python -m studio.server
```

On first run, StreamCut automatically downloads the local Parakeet speech-to-text model (`~480 MB` once) and starts at `http://127.0.0.1:8765`.

1. Open `http://127.0.0.1:8765` in your browser.
2. Go to **Settings → AI**, choose a provider (`gemini`, `openai`, `groq`, `openrouter`, `anthropic`, `xai`, `ollama`, or `custom`), enter your API key (saved locally inside your data folder), pick a model, and click **Save & Test**.
3. Paste a video/stream URL or add a creator workspace to start making clips.

## Configuration

Copy `.env.example` to `.env` if you prefer environment configuration (never commit `.env` or `secrets/`):

```ini
STREAMCUT_AI_PROVIDER=gemini
STREAMCUT_AI_MODELS=gemini-2.5-flash
STREAMCUT_AI_KEY=your-key-here
```

| Provider ID | Service | Key / Endpoint Source |
|---|---|---|
| `gemini` | Google Gemini | https://aistudio.google.com/apikey |
| `openai` | OpenAI | https://platform.openai.com/api-keys |
| `xai` | Grok (xAI) | https://console.x.ai |
| `groq` | Groq | https://console.groq.com/keys |
| `openrouter` | OpenRouter | https://openrouter.ai/keys |
| `anthropic` | Anthropic Claude | https://console.anthropic.com |
| `ollama` | Local Ollama (`http://127.0.0.1:11434/v1`) | https://ollama.com (no key needed) |
| `custom` | OpenAI-compatible endpoint | Also set `STREAMCUT_AI_BASE_URL` |

Key environment variables (full list in [`.env.example`](.env.example)):

| Setting | Purpose |
|---|---|
| `STREAMCUT_AI_PROVIDER`, `STREAMCUT_AI_MODELS`, `STREAMCUT_AI_KEY` | Default AI provider, comma-separated fallback models, and API key |
| `STREAMCUT_ACCOUNTS` | Set to `1` on public servers to require user accounts & sign-in |
| `STREAMCUT_HOST`, `STREAMCUT_PORT`, `STREAMCUT_PUBLIC_URL` | Bind host (`127.0.0.1`), port (`8765`), and external base URL |
| `STREAMCUT_DATA` | Storage directory for videos, clips, database, and models (default `./data`) |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Optional Google Sign-In and YouTube Shorts OAuth |
| `STREAMCUT_SMTP_*`, `STREAMCUT_MAIL_FROM` | Optional transactional email verification & password resets |

To grant admin privileges when `STREAMCUT_ACCOUNTS=1`:

```bash
python -m studio.admin grant you@example.com
```

## Architecture

| Path | Responsibility |
|---|---|
| `studio/server.py` | FastAPI routes, request origin/host guards, static branding & SPA serving |
| `studio/jobs.py`, `studio/live.py` | Background processing queue, rolling live buffer, auto-watch loop |
| `studio/transcribe.py`, `studio/asr_worker.py` | Local or remote GPU `sherpa-onnx` speech-to-text transcription |
| `studio/ai.py`, `studio/brain.py`, `studio/usage.py` | Unified AI client, prompt generation, JSON validation, token usage tracking |
| `studio/face_layout.py`, `studio/vision_detect.swift`, `studio/models/` | Cross-platform gamer webcam detector (Apple Vision + OpenCV YuNet) |
| `studio/render.py`, `studio/media.py` | FFmpeg clip rendering, 9:16 layouts, ASS word-by-word animated captions, `yt-dlp` downloads |
| `studio/auth.py`, `studio/accounts.py`, `studio/posting.py` | Salted scrypt auth, OAuth connectors (YouTube/Instagram/TikTok), post scheduler |
| `studio/db.py`, `studio/config.py`, `studio/storage.py` | SQLAlchemy core (SQLite/PostgreSQL), env settings, portable data folder management |
| `web/` | Plain HTML/CSS/JS frontend (no build step) and vector/PNG brand assets |
| `deploy/`, `Dockerfile`, `docker-compose.yml` | Production Docker Compose deployment with PostgreSQL, Caddy HTTPS, and GPU ASR worker |
| `tests/` | Automated self-contained test suite (`python tests/run.py`) |

See [AGENTS.md](AGENTS.md) for codebase conventions and [DEPLOY.md](DEPLOY.md) for server setup.

## Docker Deployment

```bash
cp .env.example .env
# Edit .env with your domain, PostgreSQL password, and AI settings
docker compose up -d --build
```

Includes `ffmpeg`, `deno` (required by `yt-dlp` for YouTube extraction), bundled caption fonts, OpenCV YuNet weights, PostgreSQL, and optional Caddy automatic HTTPS (`COMPOSE_PROFILES=own-https`).

## Running Tests

```bash
python tests/run.py
```

Starts a temporary test server on a spare port against an empty temporary data directory — no API keys or real media required.

## Security

- **Local-first defaults**: When `STREAMCUT_ACCOUNTS=0`, StreamCut binds to `127.0.0.1`, verifies host headers, and enforces same-origin request validation.
- **Credential safety**: Passwords use salted `scrypt` hashes, session tokens are stored SHA-256 hashed in `HTTPOnly` cookies, and secrets live solely in `.env` or `/secrets` (ignored by git).
- **Prompt safety**: Untrusted video titles, channel names, and transcripts are delimited and sanitized via `brain._safe` / `brain._text`.

## License

- Code: [MIT](LICENSE)
- Caption Fonts (`studio/fonts/` — Anton, Archivo Black, Poppins): SIL Open Font License (`OFL-*.txt`)
- Bundled Face Detection Model (`studio/models/face_detection_yunet_2023mar.onnx`): Apache License 2.0 ([`LICENSE-yunet.txt`](studio/models/LICENSE-yunet.txt))
