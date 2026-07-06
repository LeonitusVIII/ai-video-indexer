"""Pipeline skip logic, per-step status, and failure tracking."""
import datetime
import json
import sqlite3
from pathlib import Path

from job_utils import (
    metadata_json_path,
    transcript_json_path,
    vision_json_path,
)

APP_DIR = Path(__file__).resolve().parent.parent
FAILURES_FILE = APP_DIR / "data" / "pipeline_failures.json"
_failures_cache = None
_failures_mtime = None

STEP_KEYS = ("normalize", "transcribe", "vision", "metadata", "index")

STEP_LABELS = {
    "normalize": "Normalize",
    "transcribe": "Transcript",
    "vision": "Vision",
    "metadata": "Metadata",
    "index": "Indexed",
}

STEP_DB_FLAG = {
    "transcribe": "has_transcript",
    "vision": "has_vision",
    "metadata": "has_metadata",
    "index": "indexed_in_qdrant",
}

OLD_EXTENSIONS = {
    ".vob", ".mpg", ".mpeg", ".m2ts", ".mts", ".avi", ".wmv", ".mod", ".tod", ".mov",
}

SKIP_MODES = ("all", "missing_only", "stale_only", "incomplete_only")


def add_pipeline_control_args(parser):
    parser.add_argument(
        "--skip-mode",
        choices=SKIP_MODES,
        default="all",
        help="Which videos to process for this step.",
    )
    parser.add_argument(
        "--videos-file",
        default="",
        help="Optional JSON file with a list of video paths to restrict processing.",
    )
    parser.add_argument(
        "--step-overwrite",
        default="",
        help="Optional JSON dict of step_key->true/false overriding global overwrite.",
    )


def skip_mode_from_args(args):
    return getattr(args, "skip_mode", "all") or "all"


def step_overwrite_from_args(args, step_key, global_overwrite):
    raw = getattr(args, "step_overwrite", "") or ""
    if raw:
        try:
            overrides = json.loads(raw)
            if step_key in overrides:
                return overrides[step_key] in (True, "true", 1, "1")
        except Exception:
            pass
    return global_overwrite


def load_video_allowlist(args):
    path = getattr(args, "videos_file", "") or ""
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    return {str(p) for p in json.loads(file_path.read_text(encoding="utf-8"))}


def load_failures():
    global _failures_cache, _failures_mtime
    if not FAILURES_FILE.exists():
        _failures_cache = {}
        _failures_mtime = None
        return _failures_cache
    mtime = FAILURES_FILE.stat().st_mtime
    if _failures_cache is not None and mtime == _failures_mtime:
        return _failures_cache
    try:
        _failures_cache = json.loads(
            FAILURES_FILE.read_text(encoding="utf-8")
        ).get("videos", {})
    except Exception:
        _failures_cache = {}
    _failures_mtime = mtime
    return _failures_cache


def save_failures(failures):
    global _failures_cache, _failures_mtime
    FAILURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    FAILURES_FILE.write_text(
        json.dumps({"videos": failures}, indent=2),
        encoding="utf-8",
    )
    _failures_cache = failures
    _failures_mtime = FAILURES_FILE.stat().st_mtime if FAILURES_FILE.exists() else None


def record_step_failure(video_path, step_key, message=""):
    failures = load_failures()
    entry = failures.setdefault(str(video_path), {})
    entry[step_key] = message or "failed"
    save_failures(failures)


def clear_step_failure(video_path, step_key):
    failures = load_failures()
    path = str(video_path)
    entry = failures.get(path, {})
    if step_key in entry:
        del entry[step_key]
        if entry:
            failures[path] = entry
        else:
            failures.pop(path, None)
        save_failures(failures)


def clear_all_failures():
    if FAILURES_FILE.exists():
        FAILURES_FILE.unlink()


def get_video_row(db, video_path):
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("SELECT * FROM videos WHERE path = ?", (str(video_path),))
    row = cur.fetchone()
    con.close()
    return dict(row) if row else {"path": str(video_path)}


def normalize_needed(video_path):
    path = Path(video_path)
    if path.suffix.lower() not in OLD_EXTENSIONS:
        return False
    if path.name.upper().startswith("VTS_") and path.suffix.lower() == ".vob":
        return False
    return not path.with_suffix(".mkv").exists()


def vision_output_complete(sidecar_path):
    path = Path(sidecar_path)
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("status", "complete") == "complete"


def step_is_complete(video_row, step_key):
    path = video_row.get("path", "")
    if step_key == "normalize":
        return not normalize_needed(path)

    if step_key == "vision":
        sidecar = _step_sidecar(path, step_key)
        if not vision_output_complete(sidecar):
            return False
        flag_col = STEP_DB_FLAG.get(step_key)
        return bool(flag_col and video_row.get(flag_col))

    flag_col = STEP_DB_FLAG.get(step_key)
    sidecar = _step_sidecar(path, step_key)
    if flag_col and video_row.get(flag_col) and sidecar and sidecar.exists():
        return True
    return False


def _step_sidecar(video_path, step_key):
    path = Path(video_path)
    if step_key == "transcribe":
        return transcript_json_path(path)
    if step_key == "vision":
        return vision_json_path(path)
    if step_key in {"metadata", "index"}:
        return metadata_json_path(path)
    return None


def step_is_stale(video_row, step_key):
    path = Path(video_row.get("path", ""))
    if not path.exists():
        return False

    video_mtime = path.stat().st_mtime

    if step_key == "transcribe":
        sidecar = transcript_json_path(path)
        return sidecar.exists() and sidecar.stat().st_mtime < video_mtime

    if step_key == "vision":
        sidecar = vision_json_path(path)
        return sidecar.exists() and sidecar.stat().st_mtime < video_mtime

    if step_key == "metadata":
        meta = metadata_json_path(path)
        if not meta.exists():
            return False
        meta_mtime = meta.stat().st_mtime
        for dep in (transcript_json_path(path), vision_json_path(path)):
            if dep.exists() and dep.stat().st_mtime > meta_mtime:
                return True
        return meta.stat().st_mtime < video_mtime

    if step_key == "index":
        meta = metadata_json_path(path)
        if not meta.exists():
            return False
        if not video_row.get("indexed_in_qdrant"):
            return True
        qdrant_meta = APP_DIR / "data" / "qdrant" / "meta.json"
        if qdrant_meta.exists() and meta.stat().st_mtime > qdrant_meta.stat().st_mtime:
            return True
        return False

    return False


def step_status(video_row, step_key):
    path = str(video_row.get("path", ""))
    if step_key in load_failures().get(path, {}):
        return "failed"
    if step_is_complete(video_row, step_key):
        if step_is_stale(video_row, step_key):
            return "stale"
        return "complete"
    return "missing"


def should_process_video(db, video_path, step_key, skip_mode, overwrite):
    if overwrite:
        return True

    row = get_video_row(db, video_path)

    if skip_mode == "all":
        if step_key == "normalize":
            return normalize_needed(video_path)
        return True

    status = step_status(row, step_key)

    if skip_mode == "missing_only":
        return status in {"missing", "failed"}

    if skip_mode == "stale_only":
        return status in {"stale", "missing", "failed"}

    if skip_mode == "incomplete_only":
        return status != "complete"

    return True


def filter_videos_for_step(videos, db, step_key, skip_mode, overwrite, allowlist=None):
    selected = []
    for video in videos:
        path = str(video)
        if allowlist is not None and path not in allowlist:
            continue
        if should_process_video(db, path, step_key, skip_mode, overwrite):
            selected.append(video)
    return selected


def video_has_incomplete_steps(video_row):
    return any(step_status(video_row, k) != "complete" for k in STEP_KEYS)


def filter_videos_by_status(videos, status_filter):
    if status_filter == "All":
        return videos
    if status_filter == "Incomplete":
        return [v for v in videos if video_has_incomplete_steps(v)]
    if status_filter == "Failed":
        failures = load_failures()
        return [v for v in videos if str(v.get("path", "")) in failures]
    if status_filter == "Stale":
        return [v for v in videos if any(step_status(v, k) == "stale" for k in STEP_KEYS)]
    if status_filter.startswith("Missing "):
        label = status_filter.replace("Missing ", "").strip().lower()
        alias = {
            "transcript": "transcribe",
            "indexed": "index",
            "index": "index",
            "normalize": "normalize",
            "vision": "vision",
            "metadata": "metadata",
        }
        step_key = alias.get(label, label)
        return [
            v for v in videos
            if step_status(v, step_key) in {"missing", "failed"}
        ]
    return videos
