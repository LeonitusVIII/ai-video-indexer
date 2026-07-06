import datetime
import html
import os
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from pipeline_utils import STEP_LABELS, step_status


def status_icon(done):
    return "✓" if done else "·"


def step_status_icon(status):
    return {
        "complete": "✓",
        "missing": "·",
        "stale": "↻",
        "failed": "✗",
    }.get(status, "·")


def videos_status_dataframe(videos):
    import pandas as pd

    rows = []
    for video in videos:
        row = {
            "File": video.get("filename", ""),
            "Modified": (video.get("modified_time") or "")[:10],
            "Size MB": round((video.get("size_bytes") or 0) / (1024 * 1024), 1),
        }
        for step_key, label in STEP_LABELS.items():
            row[label] = step_status_icon(step_status(video, step_key))
        row["Path"] = video.get("path", "")
        rows.append(row)

    return pd.DataFrame(rows)


def parse_iso_datetime(value):
    if not value:
        return None
    try:
        return datetime.datetime.fromisoformat(value)
    except ValueError:
        return None


def format_duration(seconds):
    if seconds is None or seconds < 0:
        return "—"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def compute_job_eta(job):
    if job.get("status") != "running":
        return None

    started = parse_iso_datetime(job.get("started_at"))
    processed = int(job.get("processed", 0))
    total = int(job.get("total", 0))

    if not started or total <= 0:
        return None

    elapsed = (datetime.datetime.now() - started).total_seconds()
    if elapsed <= 0:
        return None

    if processed > 0:
        remaining_units = max(total - processed, 0)
        rate = processed / elapsed
        if rate <= 0:
            return None
        return format_duration(remaining_units / rate)

    percent = int(job.get("percent", 0))
    if percent > 0:
        total_estimated = elapsed * 100 / percent
        return format_duration(max(total_estimated - elapsed, 0))

    return None


def browse_for_folder():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as e:
        return None, f"Folder browser unavailable: {e}"

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        folder = filedialog.askdirectory(title="Select video folder")
    finally:
        root.destroy()

    if not folder:
        return None, None

    return folder, None


def format_modified_time(value):
    """Format ISO modified time for display (date + time when available)."""
    if not value:
        return ""
    text = str(value).strip()
    try:
        dt = datetime.datetime.fromisoformat(text)
        return dt.strftime("%Y-%m-%d %H:%M")
    except ValueError:
        return text[:16] if len(text) > 10 else text


def search_result_meta_line(result):
    """One-line meta: segment range, optional modified time, sources."""
    parts = [f'{result["start_label"]} – {result["end_label"]}']
    modified = format_modified_time(result.get("modified_time"))
    if modified:
        parts.append(f"Modified {modified}")
    sources = result.get("sources") or []
    if sources:
        parts.append(", ".join(sources))
    return " · ".join(parts)


def search_result_preview_text(result, max_len=220):
    """Plain preview text for search cards (CSS clamps display to two lines)."""
    text = (result.get("text") or "").strip()
    if text.lower().startswith("visual:"):
        text = text[7:].strip()
    if " Visual: " in text:
        transcript, vision = text.split(" Visual: ", 1)
        parts = []
        if transcript.strip():
            parts.append(transcript.strip())
        if vision.strip():
            parts.append(vision.strip())
        text = " · ".join(parts)
    text = " ".join(text.split())
    if len(text) > max_len:
        return text[: max_len - 1].rstrip() + "…"
    return text or "No preview text available."


def search_result_summary(result, max_len=90):
    return search_result_preview_text(result, max_len=max_len)


def search_score_tier(score):
    """Map hybrid match score to label and WCAG-friendly badge colors."""
    score = float(score)
    if score >= 0.75:
        return "Strong", "#065f46", "#d1fae5"
    if score >= 0.55:
        return "Good", "#3f6212", "#ecfccb"
    if score >= 0.40:
        return "Fair", "#78350f", "#fde68a"
    return "Weak", "#7c2d12", "#fed7aa"


def format_search_score_badge(score):
    label, color, background = search_score_tier(score)
    pct = round(float(score) * 100)
    return (
        f'<span class="search-score-badge" title="{label} match ({pct}%)" '
        f'style="color:{color}; background:{background}; border-color:{color};">'
        f"{pct}% · {label}</span>"
    )


def format_search_result_summary_html(text):
    safe = html.escape(text or "")
    return (
        f'<div class="search-result-block">'
        f'<p class="search-result-summary">{safe}</p>'
        f"</div>"
    )


def inject_search_results_styles():
    return """
<style>
/* WCAG 2.1 AA: body text >= 4.5:1 on light surfaces; large/bold >= 3:1 */
div[data-testid="stVerticalBlockBorderWrapper"] .search-result-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #0f172a !important;
    line-height: 1.4;
    margin: 0 0 0.2rem 0;
}
div[data-testid="stVerticalBlockBorderWrapper"] .search-result-meta {
    font-size: 0.875rem;
    color: #334155 !important;
    line-height: 1.45;
    margin: 0 0 0.35rem 0;
}
div[data-testid="stVerticalBlockBorderWrapper"] .search-result-summary {
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
    line-height: 1.55;
    color: #1e293b !important;
    font-size: 0.95rem;
    margin: 0.35rem 0 0.15rem 0;
}
.search-score-badge {
    display: inline-block;
    padding: 0.22rem 0.6rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    white-space: nowrap;
    border: 1px solid rgba(15, 23, 42, 0.08);
}
div[data-testid="stVerticalBlockBorderWrapper"] .search-result-block,
div[data-testid="stVerticalBlockBorderWrapper"] .search-result-block * {
    color: inherit;
}
div[data-testid="stVerticalBlockBorderWrapper"] .search-result-block a {
    color: #1e3a8a !important;
    text-decoration: underline;
}
</style>
"""


def format_search_result_text(text):
    text = (text or "").strip()
    if not text:
        return "_No text available._"

    if " Visual: " in text:
        transcript, vision = text.split(" Visual: ", 1)
        parts = []
        if transcript.strip():
            parts.append(f"**Transcript**\n\n{transcript.strip()}")
        if vision.strip():
            parts.append(f"**Visual**\n\n{vision.strip()}")
        return "\n\n".join(parts)

    if text.lower().startswith("visual:"):
        return f"**Visual**\n\n{text[7:].strip()}"

    return f"**Transcript**\n\n{text}"


def open_video_at_timestamp(video_path, start_seconds):
    path = Path(video_path)
    if not path.exists():
        return False, f"Video not found: {path}"

    start_seconds = max(0, int(float(start_seconds)))

    vlc_candidates = [
        shutil.which("vlc"),
        r"C:\Program Files\VideoLAN\VLC\vlc.exe",
        r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
    ]
    for vlc in vlc_candidates:
        if vlc and Path(vlc).exists():
            subprocess.Popen(
                [vlc, str(path), f"--start-time={start_seconds}"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True, f"Opened in VLC at {start_seconds}s"

    ffplay = shutil.which("ffplay")
    if ffplay:
        subprocess.Popen(
            [ffplay, "-ss", str(start_seconds), "-autoexit", str(path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True, f"Opened in ffplay at {start_seconds}s"

    os.startfile(str(path))
    return True, "Opened video (could not jump to timestamp — install VLC for timed playback)"
