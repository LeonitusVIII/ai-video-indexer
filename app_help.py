"""Help tab content for AI Video Indexer."""
import sys
from pathlib import Path

import streamlit as st

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from job_utils import VISION_MODEL_OPTIONS

WHISPER_DOCS_URL = "https://github.com/openai/whisper#available-models-and-languages"
FASTER_WHISPER_URL = "https://github.com/SYSTRAN/faster-whisper"

VISION_MODEL_LINKS = {
    "qwen2-vl-2b": "https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct",
    "qwen2.5-vl-3b": "https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2.5-vl-7b": "https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct",
}


def render_whisper_model_reference(*, compact=False):
    st.markdown(
        """
| Model | Speed | Accuracy | VRAM (approx.) | Best for |
|-------|-------|----------|----------------|----------|
| **tiny** | Fastest | Lowest | ~1 GB | Quick tests, clear speech only |
| **base** | Very fast | Low | ~1 GB | Draft passes, short clips |
| **small** | Fast | Moderate | ~2 GB | Balance on limited GPU |
| **medium** | Moderate | Good | ~5 GB | Better names/terms without largest model |
| **large-v2** | Slow | Very good | ~10 GB | High accuracy, prior Whisper generation |
| **large-v3** | Slowest | Best | ~10 GB | **Recommended** for home video on RTX-class GPUs |

This app runs Whisper through **faster-whisper** (CTranslate2). Models download on first use.
        """
    )
    if not compact:
        st.markdown(
            f"""
**Tips**
- Home video with background noise benefits from **large-v3** when you have a GPU.
- Use **small** or **medium** on CPU-only setups or very large libraries where speed matters more.
- Larger models do not fix silent or music-only sections — they mainly improve word accuracy.

**External references:** [OpenAI Whisper models]({WHISPER_DOCS_URL}) · [faster-whisper]({FASTER_WHISPER_URL})
            """
        )


def render_vision_model_reference(*, compact=False):
    rows = []
    for key, option in VISION_MODEL_OPTIONS.items():
        label = option["label"].split(" (")[0]
        model_id = option["model_id"]
        link = VISION_MODEL_LINKS.get(key, "")
        if key == "qwen2-vl-2b":
            speed, quality, vram, best = "Fastest", "Good", "~4 GB", "Fast scans, limited VRAM"
        elif key == "qwen2.5-vl-3b":
            speed, quality, vram, best = "Balanced", "Very good", "~6 GB", "**Default** — best balance"
        else:
            speed, quality, vram, best = "Slowest", "Best detail", "~16 GB", "Maximum scene detail"
        name_cell = f"[{label}]({link})" if link else label
        rows.append(f"| {name_cell} | {speed} | {quality} | {vram} | {best} |")

    st.markdown(
        """
| Model | Speed | Detail | VRAM (approx.) | Best for |
|-------|-------|--------|----------------|----------|
"""
        + "\n".join(rows)
        + """

Models run **locally** on your GPU (or CPU if GPU is off). First run downloads weights from Hugging Face.
        """
    )
    if not compact:
        st.markdown(
            """
**Tips**
- Vision time scales with **frame interval** — halving the interval can roughly double vision runtime.
- **2B** is the lightest option; **7B** needs substantially more VRAM and time.
- Vision describes sampled still frames, not full motion — lower intervals catch more action.

**External references:** """
            + " · ".join(
                f"[{VISION_MODEL_OPTIONS[k]['label'].split(' (')[0]}]({VISION_MODEL_LINKS[k]})"
                for k in VISION_MODEL_OPTIONS
                if k in VISION_MODEL_LINKS
            )
        )


def whisper_model_info_popover():
    with st.popover("Model info"):
        render_whisper_model_reference(compact=True)
        st.caption("Full guide: open the **Help** tab → **Whisper models**.")
        st.link_button("OpenAI Whisper model list", WHISPER_DOCS_URL, use_container_width=True)


def vision_model_info_popover():
    with st.popover("Model info"):
        render_vision_model_reference(compact=True)
        st.caption("Full guide: open the **Help** tab → **Vision models**.")
        st.link_button(
            "Qwen2.5-VL 3B on Hugging Face",
            VISION_MODEL_LINKS["qwen2.5-vl-3b"],
            use_container_width=True,
        )


def render_help_tab():
    st.header("Help")
    st.write(
        "AI Video Indexer catalogs home videos on your PC or network share, "
        "transcribes speech, describes scenes, and lets you search by what was "
        "said or shown — with timestamps you can jump to in your video player."
    )

    st.subheader("Quick start")
    st.markdown(
        """
1. **Library** — Add a folder path (`D:\\Videos` or `\\\\SERVER\\Share\\Videos`).
2. **Run Jobs** — click **Scan Library** to catalog videos in that folder.
3. **Run Jobs** — Choose pipeline steps, then **Run Selected Pipeline**.
4. **Search** — Try a natural-language query after indexing finishes.

Only one background job runs at a time (scan, pipeline, or install).
        """
    )

    st.subheader("Tabs")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
**Dashboard**  
Live job progress, system stats, and library summary.

**Library**  
Add/remove folders, browse scanned files, and see per-video pipeline status (✓ · ↻ ✗).

**Run Jobs**  
Scan, processing settings, pipeline steps, and quick actions (*Index missing only*, *Refresh stale*, etc.).
            """
        )
    with col2:
        st.markdown(
            """
**Search**  
Semantic + keyword search with optional filters (folder, date, size, extension).

**Tools/System**  
Install dependencies, system check, Discord notifications, reset search index.

**Logs**  
View processing logs for troubleshooting.
            """
        )

    st.subheader("Pipeline steps")
    st.markdown(
        """
| Step | Purpose |
|------|---------|
| **Scan library** | Finds video files and updates the SQLite catalog. Removes catalog entries for files deleted from disk under the scanned folder. |
| **Normalize** | Remuxes legacy formats (`.mov`, `.vob`, `.avi`, …) to `.mkv` without re-encoding. |
| **Transcribe** | Whisper speech-to-text with timestamps (sidecar `.transcript.json` next to each video). |
| **Vision** | Sampled frame descriptions via a local Qwen vision model (`.vision.json`). |
| **Metadata** | Merges transcript + vision into searchable chunks (`.metadata.json`). |
| **Index search DB** | Embeds chunks into local Qdrant for the Search tab. |
        """
    )

    st.subheader("Skip modes")
    st.markdown(
        """
- **Missing outputs only** — Safest for re-runs; skips videos that already completed a step.
- **Stale or outdated sidecars** — Re-process when the video file is newer than its sidecars, or metadata is newer than the search index.
- **Incomplete only** — Anything not fully through all steps.
- **All videos** — Every selected video, still respecting per-step **Force** checkboxes.

Use **Force** on a step to re-run it even when output already exists.
        """
    )

    st.subheader("Whisper models")
    render_whisper_model_reference(compact=False)

    st.subheader("Vision models")
    render_vision_model_reference(compact=False)

    st.subheader("Sidecar files")
    st.markdown(
        """
Processing writes JSON/text files **next to each video** on your share (same folder as the file):

- `.transcript.json`, `.transcript.txt`, `.whisper.srt`
- `.vision.json`
- `.metadata.json`

The app database lives under `data/video_indexer.db`; search vectors live under `data/qdrant/`.
        """
    )

    st.subheader("Common workflows")
    st.markdown(
        """
**New folder**  
Add folder → Scan Library → Run Selected Pipeline (all steps on, skip mode *Missing only*).

**New videos in an existing folder**  
Scan Library → **Index missing only** quick action (or full pipeline with missing-only).

**Changed transcript/vision settings**  
**Refresh stale metadata/index** with skip mode *Stale* (or Force on specific steps).

**Search returns nothing**  
Confirm **Index search DB** ran, check Dashboard *Indexed* count, or **Reset search index** under Tools then re-index.

**Normalize moved files**  
After normalize, re-run **Index search DB** for affected videos so search paths stay correct.
        """
    )

    st.subheader("Requirements")
    st.markdown(
        """
- Windows 10/11, Python 3.11+, FFmpeg, ~15 GB for AI packages.
- NVIDIA GPU recommended (Whisper + vision); CPU works but is much slower.
- Network shares: use full UNC paths; the Windows account running the app needs read/write access.
        """
    )

    st.subheader("Privacy")
    st.info(
        "Everything runs locally. Videos are not uploaded anywhere unless you configure "
        "an optional Discord webhook for job notifications under Tools/System."
    )
