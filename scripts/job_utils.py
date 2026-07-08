import datetime
import json
import sqlite3
import subprocess
import sys
from pathlib import Path

VIDEO_EXTENSIONS = {
    ".mp4", ".mkv", ".mov", ".avi", ".wmv", ".mpg", ".mpeg",
    ".vob", ".mts", ".m2ts", ".3gp", ".webm", ".flv", ".m4v"
}

APP_DIR = Path(__file__).resolve().parent.parent
QDRANT_DIR = APP_DIR / "data" / "qdrant"
COLLECTION_NAME = "video_segments"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

VISION_MODEL_OPTIONS = {
    "qwen2.5-vl-3b": {
        "label": "Qwen2.5-VL 3B (recommended, ~6 GB VRAM)",
        "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
        "family": "qwen2_5_vl",
    },
    "qwen2.5-vl-7b": {
        "label": "Qwen2.5-VL 7B (higher quality, ~16 GB VRAM)",
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "family": "qwen2_5_vl",
    },
    "qwen2-vl-2b": {
        "label": "Qwen2-VL 2B (fastest, lower VRAM)",
        "model_id": "Qwen/Qwen2-VL-2B-Instruct",
        "family": "qwen2_vl",
    },
}

DEFAULT_VISION_MODEL_KEY = "qwen2.5-vl-3b"
VISION_MODEL = VISION_MODEL_OPTIONS[DEFAULT_VISION_MODEL_KEY]["model_id"]


def read_status(status_file):
    try:
        return json.loads(Path(status_file).read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_status(status_file, data):
    if not status_file:
        return
    Path(status_file).write_text(json.dumps(data, indent=2), encoding="utf-8")


def fail_job_status(status_file, exc, *, prefix=""):
    """Mark a job status file failed — used when a script crashes unexpectedly."""
    status = read_status(status_file) if status_file else {}
    message = str(exc).strip() or exc.__class__.__name__
    if prefix:
        message = f"{prefix}: {message}"
    status.update({
        "status": "failed",
        "current": message[:500],
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    write_status(status_file, status)


def status_file_from_argv(argv=None):
    argv = argv or sys.argv
    for index, arg in enumerate(argv):
        if arg == "--status-file" and index + 1 < len(argv):
            return argv[index + 1]
    return None


def run_script_main(main_fn, status_file=None):
    """Top-level guard so unexpected exceptions update the job JSON before exit."""
    try:
        main_fn()
    except SystemExit:
        raise
    except Exception as exc:
        if status_file:
            fail_job_status(status_file, exc)
        print(f"FAILED: {exc}", flush=True)
        raise SystemExit(1) from exc


def set_item_progress(status, status_file, item_label, item_processed, item_total):
    """Track within-file progress (e.g. vision frames on the current video)."""
    item_total = int(item_total or 0)
    item_processed = int(item_processed or 0)
    if item_total <= 0:
        for key in ("item_label", "item_processed", "item_total", "item_percent"):
            status.pop(key, None)
    else:
        status.update({
            "item_label": str(item_label or "Current file"),
            "item_processed": min(item_processed, item_total),
            "item_total": item_total,
            "item_percent": int((min(item_processed, item_total) / item_total) * 100),
        })
    write_status(status_file, status)


def clear_item_progress(status, status_file):
    set_item_progress(status, status_file, "", 0, 0)


def should_stop(status_file):
    return read_status(status_file).get("stop_requested", False)


LEGACY_REVIEW_DIR = "_OLD_FILES_REVIEW"


def find_videos(folder):
    return sorted(
        p for p in Path(folder).rglob("*")
        if (
            p.is_file()
            and p.suffix.lower() in VIDEO_EXTENSIONS
            and LEGACY_REVIEW_DIR not in p.parts
            and ".audio_enhanced" not in p.stem.lower()
        )
    )


def sidecar_path(video, suffix):
    video = Path(video)
    return video.with_suffix(video.suffix + suffix)


def transcript_json_path(video):
    return sidecar_path(video, ".transcript.json")


def vision_json_path(video):
    return sidecar_path(video, ".vision.json")


def metadata_json_path(video):
    return sidecar_path(video, ".metadata.json")


def update_video_flag(db, video_path, column):
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(
        f"UPDATE videos SET {column} = 1 WHERE path = ?",
        (str(video_path),)
    )
    con.commit()
    con.close()


def get_video_duration(video_path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "ffprobe failed")
    return float(result.stdout.strip())


def compute_frame_times(duration, interval_seconds, min_frames):
    if duration <= 0:
        return [0.0]

    interval_times = []
    t = 0.0
    while t < duration:
        interval_times.append(round(t, 3))
        t += interval_seconds

    if len(interval_times) >= min_frames:
        return interval_times

    if min_frames == 1:
        return [round(duration / 2, 3)]

    return [
        round((i + 1) * duration / (min_frames + 1), 3)
        for i in range(min_frames)
    ]


def thumbnail_path(video):
    return sidecar_path(video, ".thumbnail.jpg")


def compute_file_fingerprint(video_path, size_bytes=None):
    import hashlib

    path = Path(video_path)
    if size_bytes is None:
        size_bytes = path.stat().st_size
    digest = hashlib.md5()
    digest.update(str(size_bytes).encode("utf-8"))
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if size_bytes > 2 * 1024 * 1024:
            handle.seek(-1024 * 1024, 2)
            digest.update(handle.read(1024 * 1024))
    return digest.hexdigest()


def ensure_video_thumbnail(video_path, duration_seconds=None):
    path = Path(video_path)
    output = thumbnail_path(path)
    if output.exists() and output.stat().st_mtime >= path.stat().st_mtime:
        return output
    if duration_seconds is None:
        duration_seconds = get_video_duration(path)
    timestamp = min(max(float(duration_seconds) * 0.1, 1.0), max(float(duration_seconds) - 1, 1.0))
    extract_frame_image(path, timestamp, output)
    return output


def extract_frame_image(video_path, timestamp_seconds, output_path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss", str(timestamp_seconds),
            "-i", str(video_path),
            "-frames:v", "1",
            "-q:v", "2",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def add_processing_args(parser):
    parser.add_argument("--use-gpu", choices=["true", "false"], default="true")
    parser.add_argument("--overwrite", choices=["true", "false"], default="false")
    parser.add_argument("--vision-interval", type=int, default=30)
    parser.add_argument("--min-frames", type=int, default=3)
    parser.add_argument("--transcription-model", default="large-v3")
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL_KEY)


def vision_model_key_from_args(args):
    key = getattr(args, "vision_model", DEFAULT_VISION_MODEL_KEY) or DEFAULT_VISION_MODEL_KEY
    if key in VISION_MODEL_OPTIONS:
        return key
    for option_key, option in VISION_MODEL_OPTIONS.items():
        if option["model_id"] == key:
            return option_key
    return DEFAULT_VISION_MODEL_KEY


def vision_model_config_from_args(args):
    key = vision_model_key_from_args(args)
    return VISION_MODEL_OPTIONS[key]


def transcription_model_from_args(args):
    return getattr(args, "transcription_model", "large-v3") or "large-v3"


def use_gpu_from_args(args):
    return getattr(args, "use_gpu", "true") == "true"


def overwrite_from_args(args):
    return getattr(args, "overwrite", "false") == "true"


def format_timestamp(seconds):
    seconds = max(0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
