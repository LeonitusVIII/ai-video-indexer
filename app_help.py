"""Help tab content for AI Video Indexer."""
import streamlit as st


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
2. **Run Jobs** — **Scan Library** (or run the pipeline with *Scan library before pipeline* enabled).
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

**Tools / Status**  
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
| **Normalize** | Remuxes legacy formats (`.mov`, `.vob`, `.avi`, …) to `.mkv` without re-encoding. Use **dry run** first to preview. |
| **Transcribe** | Whisper speech-to-text with timestamps (sidecar `.transcript.json` next to each video). |
| **Vision** | Sampled frame descriptions via Qwen2.5-VL (`.vision.json`). |
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
        "an optional Discord webhook for job notifications under Tools / Status."
    )
