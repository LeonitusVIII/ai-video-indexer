"""Standalone HEVC/H.265 transcode job (not part of the main pipeline)."""
import argparse
import csv
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_utils import find_videos, read_status, should_stop, write_status
from pipeline_utils import clear_step_failure, record_step_failure

APP_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = APP_DIR / "logs"
LEGACY_REVIEW_DIR = "_OLD_FILES_REVIEW"

SIDECAR_SUFFIXES = [
    ".transcript.json",
    ".transcript.txt",
    ".whisper.srt",
    ".vision.json",
    ".metadata.json",
    ".thumbnail.jpg",
]

QUALITY_PRESETS = {
    "archival": {"crf": 20, "preset": "slow", "label": "Archival (larger, best quality)"},
    "balanced": {"crf": 24, "preset": "medium", "label": "Balanced (recommended)"},
    "compact": {"crf": 28, "preset": "medium", "label": "Compact (smaller files)"},
    "smallest": {"crf": 32, "preset": "slow", "label": "Smallest (most compression)"},
}

HEVC_CODECS = {"hevc", "h265", "hev1", "hvc1"}


def add_transcode_args(parser):
    parser.add_argument("--quality-preset", default="balanced", choices=tuple(QUALITY_PRESETS))
    parser.add_argument("--crf", type=int, default=0, help="0 = use preset CRF")
    parser.add_argument("--rate-control", default="crf", choices=["crf", "bitrate", "filesize"])
    parser.add_argument("--target-bitrate-kbps", type=int, default=0)
    parser.add_argument("--target-filesize-mb", type=int, default=0)
    parser.add_argument("--encoder", default="auto", choices=["auto", "nvenc", "x265"])
    parser.add_argument("--x265-preset", default="medium")
    parser.add_argument("--audio-mode", default="copy", choices=["copy", "aac192", "aac128"])
    parser.add_argument("--output-mode", default="suffix", choices=["suffix", "replace"])
    parser.add_argument("--output-extension", default=".mp4")
    parser.add_argument("--skip-hevc-source", choices=["true", "false"], default="true")
    parser.add_argument("--max-height", type=int, default=0, help="0 keeps source resolution")
    parser.add_argument("--overwrite", choices=["true", "false"], default="false")
    parser.add_argument("--min-free-gb", type=float, default=5.0)
    parser.add_argument("--videos-file", default="")


def run_cmd(cmd):
    print("RUN:", " ".join(str(x) for x in cmd), flush=True)
    return subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def ffprobe_value(path, entries):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", entries,
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip().splitlines()[0] if result.stdout.strip() else ""


def video_codec_name(path):
    return ffprobe_value(path, "stream=codec_name").lower()


def video_height(path):
    value = ffprobe_value(path, "stream=height")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def encoder_available(encoder_name):
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return encoder_name in (result.stdout or "")


def resolve_encoder(requested):
    if requested == "x265":
        return "libx265", "software"
    if requested == "nvenc":
        if encoder_available("hevc_nvenc"):
            return "hevc_nvenc", "nvenc"
        raise RuntimeError("NVENC HEVC encoder not available in FFmpeg.")
    if encoder_available("hevc_nvenc"):
        return "hevc_nvenc", "nvenc"
    return "libx265", "software"


def video_duration_seconds(path):
    value = ffprobe_value(path, "format=duration")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def audio_bitrate_kbps(path, audio_mode):
    if audio_mode == "aac192":
        return 192
    if audio_mode == "aac128":
        return 128
    value = ffprobe_value(path, "stream=bit_rate")
    try:
        kbps = int(round(float(value) / 1000))
        return max(64, kbps)
    except (TypeError, ValueError):
        return 128


def rate_control_mode(args):
    mode = getattr(args, "rate_control", "crf") or "crf"
    if mode == "bitrate" and int(getattr(args, "target_bitrate_kbps", 0) or 0) > 0:
        return "bitrate"
    if mode == "filesize" and int(getattr(args, "target_filesize_mb", 0) or 0) > 0:
        return "filesize"
    return "crf"


def target_video_bitrate_kbps(src, args):
    mode = rate_control_mode(args)
    if mode == "bitrate":
        return max(500, int(args.target_bitrate_kbps))
    if mode == "filesize":
        duration = video_duration_seconds(src)
        if duration <= 0:
            raise RuntimeError(f"Could not read duration for bitrate sizing: {src}")
        total_kbps = int((int(args.target_filesize_mb) * 1024 * 1024 * 8) / duration / 1000)
        audio_kbps = audio_bitrate_kbps(src, args.audio_mode)
        return max(500, total_kbps - audio_kbps)
    return 0


def effective_crf(args):
    if args.crf and int(args.crf) > 0:
        return max(18, min(36, int(args.crf)))
    preset = QUALITY_PRESETS.get(args.quality_preset, QUALITY_PRESETS["balanced"])
    return int(preset["crf"])


def effective_preset(args, encoder_kind):
    if encoder_kind == "nvenc":
        return "p4"
    if args.x265_preset:
        return args.x265_preset
    preset = QUALITY_PRESETS.get(args.quality_preset, QUALITY_PRESETS["balanced"])
    return preset["preset"]


def output_path_for(src, args):
    ext = args.output_extension if args.output_extension.startswith(".") else f".{args.output_extension}"
    if args.output_mode == "replace":
        return src.with_suffix(ext)
    stem = src.stem
    if stem.endswith(".hevc"):
        return src.with_name(stem + ext)
    return src.with_name(f"{stem}.hevc{ext}")


def build_ffmpeg_cmd(src, dst, args):
    encoder, encoder_kind = resolve_encoder(args.encoder)
    preset = effective_preset(args, encoder_kind)
    src_height = video_height(src)
    scale_filter = None
    if args.max_height and src_height and src_height > int(args.max_height):
        scale_filter = f"scale=-2:{int(args.max_height)}"

    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src)]

    if scale_filter:
        cmd.extend(["-vf", scale_filter])

    mode = rate_control_mode(args)
    if mode == "crf":
        crf = effective_crf(args)
        if encoder_kind == "nvenc":
            cmd.extend([
                "-c:v", "hevc_nvenc",
                "-preset", preset,
                "-rc", "vbr",
                "-cq", str(crf),
                "-b:v", "0",
            ])
        else:
            cmd.extend([
                "-c:v", "libx265",
                "-crf", str(crf),
                "-preset", preset,
                "-tag:v", "hvc1",
            ])
    else:
        video_kbps = target_video_bitrate_kbps(src, args)
        max_kbps = int(video_kbps * 1.25)
        buf_kbps = int(video_kbps * 2)
        if encoder_kind == "nvenc":
            cmd.extend([
                "-c:v", "hevc_nvenc",
                "-preset", preset,
                "-rc", "vbr",
                "-b:v", f"{video_kbps}k",
                "-maxrate", f"{max_kbps}k",
                "-bufsize", f"{buf_kbps}k",
            ])
        else:
            cmd.extend([
                "-c:v", "libx265",
                "-b:v", f"{video_kbps}k",
                "-maxrate", f"{max_kbps}k",
                "-bufsize", f"{buf_kbps}k",
                "-preset", preset,
                "-tag:v", "hvc1",
            ])

    if args.audio_mode == "copy":
        cmd.extend(["-c:a", "copy"])
    elif args.audio_mode == "aac192":
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    else:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])

    cmd.extend(["-movflags", "+faststart", str(dst)])
    return cmd


def copy_sidecars(src, dst):
    copied = []
    for suffix in SIDECAR_SUFFIXES:
        old_sidecar = Path(str(src) + suffix)
        new_sidecar = Path(str(dst) + suffix)
        if old_sidecar.exists():
            print(f"COPY SIDECAR: {old_sidecar} -> {new_sidecar}", flush=True)
            shutil.copy2(old_sidecar, new_sidecar)
            copied.append(str(new_sidecar))
    return copied


def move_original_to_review(src, folder):
    review_dir = folder / LEGACY_REVIEW_DIR
    review_dir.mkdir(parents=True, exist_ok=True)
    target = review_dir / src.name
    if target.exists():
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        target = review_dir / f"{src.stem}_{stamp}{src.suffix}"
    print(f"MOVE ORIGINAL: {src} -> {target}", flush=True)
    shutil.move(str(src), str(target))
    return target


def migrate_catalog_record(db, old_path, new_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from normalize_videos import migrate_video_record

    migrate_video_record(db, old_path, new_path)


def load_video_allowlist(args):
    path = getattr(args, "videos_file", "") or ""
    if not path:
        return None
    file_path = Path(path)
    if not file_path.exists():
        return None
    return {str(p) for p in json.loads(file_path.read_text(encoding="utf-8"))}


def should_transcode(src, dst, args):
    if args.skip_hevc_source == "true" and video_codec_name(src) in HEVC_CODECS:
        return False, "already_hevc"
    if dst.exists() and args.overwrite != "true":
        return False, "output_exists"
    return True, "ok"


def transcode_video(src, dst, args):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        cmd = build_ffmpeg_cmd(src, dst, args)
    except Exception as exc:
        return False, str(exc)
    result = run_cmd(cmd)
    if result.returncode != 0:
        if dst.exists():
            dst.unlink(missing_ok=True)
        return False, (result.stderr or result.stdout or "ffmpeg failed").strip()
    if not dst.exists() or dst.stat().st_size <= 0:
        return False, "Output file missing or empty"
    return True, ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--status-file", required=True)
    add_transcode_args(parser)
    args = parser.parse_args()

    folder = Path(args.folder)
    status_file = Path(args.status_file)
    allowlist = load_video_allowlist(args)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    converted_log = LOG_DIR / f"transcode_hevc_converted_{timestamp}.csv"
    failed_log = LOG_DIR / f"transcode_hevc_failed_{timestamp}.csv"

    status = read_status(status_file)
    status.update({
        "status": "running",
        "current": "Finding videos for HEVC transcode...",
        "percent": 0,
        "processed": 0,
        "total": 0,
    })
    write_status(status_file, status)

    print(f"HEVC transcode started for: {folder}", flush=True)
    mode = rate_control_mode(args)
    settings_line = (
        f"Settings: rate={mode}, encoder={args.encoder}, audio={args.audio_mode}, "
        f"output={args.output_mode}"
    )
    if mode == "crf":
        settings_line += f", crf={effective_crf(args)}, preset={args.quality_preset}"
    elif mode == "bitrate":
        settings_line += f", target_bitrate={args.target_bitrate_kbps} kbps"
    else:
        settings_line += f", target_filesize={args.target_filesize_mb} MB"
    print(settings_line, flush=True)

    videos = find_videos(folder)
    if allowlist is not None:
        videos = [video for video in videos if str(video) in allowlist]

    sys.path.insert(0, str(APP_DIR))
    from disk_space import transcode_space_check, format_bytes, get_disk_space

    space = transcode_space_check(
        folder,
        videos,
        output_mode=args.output_mode,
        min_free_gb=float(args.min_free_gb),
    )
    print(f"Disk check: {space['message']}", flush=True)
    if space["disk"].get("available") and not space["ok"]:
        status.update({
            "status": "failed",
            "current": space["message"],
            "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        write_status(status_file, status)
        print(f"FAILED: {space['message']}", flush=True)
        sys.exit(1)

    total = len(videos)
    status.update({"total": total, "current": f"Found {total} video files."})
    write_status(status_file, status)
    print(f"Found {total} video files.", flush=True)

    converted_rows = []
    failed_rows = []
    skipped = 0

    for i, src in enumerate(videos, start=1):
        if should_stop(status_file):
            status.update({
                "status": "stopped",
                "current": "Stopped by user.",
                "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
            })
            write_status(status_file, status)
            print("Stopped by user.", flush=True)
            sys.exit(0)

        dst = output_path_for(src, args)
        ok, reason = should_transcode(src, dst, args)
        status.update({
            "current": f"Transcoding: {src.name}",
            "processed": i - 1,
            "total": total,
            "percent": int(((i - 1) / total) * 100) if total else 100,
        })
        write_status(status_file, status)

        if not ok:
            skipped += 1
            print(f"Skipping ({reason}): {src}", flush=True)
        else:
            print(f"\nTranscoding: {src} -> {dst}", flush=True)
            per_file = transcode_space_check(
                folder,
                [src],
                output_mode=args.output_mode,
                min_free_gb=float(args.min_free_gb),
            )
            if per_file["disk"].get("available") and not per_file["ok"]:
                failed_rows.append({
                    "source": str(src),
                    "output": str(dst),
                    "error": per_file["message"],
                })
                record_step_failure(src, "transcode", per_file["message"])
                print(f"FAILED (disk space): {src}", flush=True)
                print(per_file["message"], flush=True)
                continue

            before_size = src.stat().st_size
            success, error = transcode_video(src, dst, args)
            if success:
                after_size = dst.stat().st_size
                ratio = (after_size / before_size) if before_size else 0
                converted_rows.append({
                    "source": str(src),
                    "output": str(dst),
                    "before_bytes": before_size,
                    "after_bytes": after_size,
                    "size_ratio": f"{ratio:.2f}",
                    "encoder": resolve_encoder(args.encoder)[0],
                    "rate_control": rate_control_mode(args),
                    "crf": effective_crf(args) if rate_control_mode(args) == "crf" else "",
                    "video_kbps": (
                        target_video_bitrate_kbps(src, args)
                        if rate_control_mode(args) != "crf"
                        else ""
                    ),
                })
                clear_step_failure(src, "transcode")
                if args.output_mode == "replace":
                    copy_sidecars(src, dst)
                    migrate_catalog_record(args.db, src, dst)
                    move_original_to_review(src, folder)
                    for suffix in SIDECAR_SUFFIXES:
                        old_sidecar = Path(str(src) + suffix)
                        if old_sidecar.exists():
                            old_sidecar.unlink()
                print(
                    f"Completed: {dst.name} "
                    f"({before_size / (1024 * 1024):.1f} MB -> {after_size / (1024 * 1024):.1f} MB)",
                    flush=True,
                )
            else:
                failed_rows.append({"source": str(src), "output": str(dst), "error": error})
                record_step_failure(src, "transcode", error)
                print(f"FAILED: {src}", flush=True)
                print(error, flush=True)

        percent = int((i / total) * 100) if total else 100
        status.update({
            "processed": i,
            "total": total,
            "percent": percent,
            "current": str(src),
        })
        write_status(status_file, status)

    if converted_rows:
        with converted_log.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(converted_rows[0].keys()))
            writer.writeheader()
            writer.writerows(converted_rows)
    if failed_rows:
        with failed_log.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(failed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(failed_rows)

    status.update({
        "status": "complete",
        "percent": 100,
        "current": (
            f"HEVC transcode complete. Converted: {len(converted_rows)}, "
            f"skipped: {skipped}, failed: {len(failed_rows)}."
        ),
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    write_status(status_file, status)
    print(status["current"], flush=True)


if __name__ == "__main__":
    from job_utils import run_script_main, status_file_from_argv

    run_script_main(main, status_file_from_argv())
