# AI Video Indexer

Search your home videos by what was said or what appears on screen. Index files on a local folder or network share, then run semantic search with timestamps and playback links.

**Current release:** [v0.2.1](https://github.com/LeonitusVIII/ai-video-indexer/releases/tag/v0.2.1) — see [CHANGELOG.md](CHANGELOG.md) for details.
## Requirements

| Requirement | Notes |
|-------------|--------|
| **Windows 10/11** | Tested on Windows |
| **Python 3.11 or 3.12** | [python.org/downloads](https://www.python.org/downloads/) — enable **Add to PATH** |
| **FFmpeg** | Installed automatically by `setup.bat` via winget when possible; otherwise [ffmpeg.org/download.html](https://ffmpeg.org/download.html) |
| **NVIDIA GPU** (recommended) | RTX series for faster Whisper + vision; CPU works but is much slower |
| **Disk space** | ~15 GB for Python packages; extra space for AI model downloads |
| **RAM** | 16 GB minimum; 32 GB recommended for vision step |

Optional: **VLC** for “Play here” from search results.

## Quick start

1. Unzip this folder anywhere (e.g. `C:\Apps\VideoIndexer`).
2. Double-click **`setup.bat`** — installs Streamlit, FFmpeg (when winget is available), and other dependencies. Wait for it to finish.
3. Double-click **`start.bat`** — opens the app at [http://localhost:8501](http://localhost:8501).

## First-time workflow

1. **Library** tab — add a folder path (Browse or paste a path like `D:\Videos` or `\\SERVER\Media\Home Videos`).
2. **Run Jobs** tab — click **Scan Library** to catalog videos in that folder.
3. Select pipeline steps (defaults are all on), then **Run Selected Pipeline**.
4. **Search** tab — try a natural-language query after indexing completes.

### Pipeline steps

| Step | What it does |
|------|----------------|
| Scan library | Manual only — catalog video files in SQLite (Run Jobs → **Scan Library**) |
| Normalize old videos | Remux `.mov`, `.vob`, `.avi`, etc. to `.mkv` |
| Transcribe | Whisper speech-to-text with timestamps |
| Analyze vision | Qwen2.5-VL frame descriptions |
| Build metadata | Merge transcript + vision for search |
| Index search DB | Embed segments into local Qdrant database |

Sidecar files (`.transcript.json`, `.vision.json`, etc.) are saved next to each video.

## Discord notifications (optional)

**Tools/System** → paste your own Discord webhook URL. Nothing is sent unless you configure it.

## Troubleshooting

- **setup.bat fails** — open `data\install_status.json` for the failed step; ensure Python 3.11+ and internet access.
- **ffmpeg not found** — install FFmpeg and restart the terminal / PC so PATH updates.
- **GPU not used** — confirm `nvidia-smi` works in a terminal; enable “Use GPU” on Run Jobs.
- **Search empty** — run **Index search DB** and confirm metadata sidecars exist for your videos.
- **Network share paths** — use full UNC paths (`\\server\share\folder`); the Windows account running the app needs read access.

## Folder layout

```
VideoIndexer/
  app.py              Main Streamlit UI
  setup.bat           First-time install
  start.bat           Launch the app
  config.json         Your folders and settings (created from config.example.json)
  scripts/            Processing pipeline scripts
  data/               SQLite DB, Qdrant index, job state
  jobs/               Job status JSON files
  logs/               Processing logs
```

## Creating a release zip (for developers)

From PowerShell in this folder:

```powershell
.\package_release.ps1
```

Output: `dist\VideoIndexer.zip` — share that file. It excludes your venv, logs, database, and personal config.

## Privacy

Everything runs locally on your machine. Videos are not uploaded anywhere unless you add a Discord webhook for job notifications.

## GitHub / development

Clone the repository, then run `setup.bat` and `start.bat` as above. Your local `config.json`, database, logs, and Qdrant index are gitignored and stay on your machine.

```powershell
git clone https://github.com/LeonitusVIII/ai-video-indexer.git
cd ai-video-indexer
.\setup.bat
.\start.bat
```

To build a shareable zip without personal data:

```powershell
.\package_release.ps1
```
