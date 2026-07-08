import argparse
import csv
import datetime
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_utils import read_status, should_stop, write_status
from pipeline_utils import (
    add_pipeline_control_args,
    load_video_allowlist,
    record_step_failure,
    clear_step_failure,
    should_process_video,
    skip_mode_from_args,
    step_overwrite_from_args,
)

APP_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = APP_DIR / "logs"

OLD_EXTENSIONS = {
    ".vob", ".mpg", ".mpeg", ".m2ts", ".mts", ".avi", ".wmv", ".mod", ".tod",
    ".mov",
}

SIDECAR_SUFFIXES = [
    ".transcript.json",
    ".transcript.txt",
    ".whisper.srt",
    ".vision.json",
    ".metadata.json",
]

LEGACY_REVIEW_DIR = "_OLD_FILES_REVIEW"


def add_normalize_args(parser):
    parser.add_argument("--overwrite", choices=["true", "false"], default="false")


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


def ffprobe_ok(path):
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_type",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.returncode == 0


def remux_to_mkv(src, dst):
    cmd = [
        "ffmpeg",
        "-y",
        "-ignore_chapters", "1",
        "-fflags", "+genpts",
        "-i", str(src),
        "-map", "0",
        "-c", "copy",
        str(dst),
    ]

    result = run_cmd(cmd)
    if result.returncode != 0:
        return False, result.stderr
    return True, ""


def copy_sidecars(src, dst):
    copied = []

    for suffix in SIDECAR_SUFFIXES:
        old_sidecar = Path(str(src) + suffix)
        new_sidecar = Path(str(dst) + suffix)

        if old_sidecar.exists():
            print(f"COPY SIDECAR: {old_sidecar} -> {new_sidecar}", flush=True)
            copied.append(str(new_sidecar))
            shutil.copy2(old_sidecar, new_sidecar)

    return copied


def delete_original_and_old_sidecars(src):
    print(f"DELETE ORIGINAL: {src}", flush=True)
    if src.exists():
        src.unlink()

    for suffix in SIDECAR_SUFFIXES:
        old_sidecar = Path(str(src) + suffix)
        if old_sidecar.exists():
            print(f"DELETE OLD SIDECAR: {old_sidecar}", flush=True)
            old_sidecar.unlink()


def should_skip(src):
    if LEGACY_REVIEW_DIR in src.parts:
        return True

    if src.suffix.lower() not in OLD_EXTENSIONS:
        return True

    name_upper = src.name.upper()
    if name_upper.startswith("VTS_") and src.suffix.lower() == ".vob":
        return True

    return False


def migrate_video_record(db, old_path, new_path):
    old_path = str(old_path)
    new_path = str(new_path)

    if not Path(new_path).exists():
        return

    stat = Path(new_path).stat()
    modified = datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds")

    con = sqlite3.connect(db)
    cur = con.cursor()

    cur.execute(
        """
        SELECT has_transcript, has_vision, has_metadata, indexed_in_qdrant
        FROM videos WHERE path = ?
        """,
        (old_path,),
    )
    old_row = cur.fetchone()

    cur.execute("SELECT 1 FROM videos WHERE path = ?", (new_path,))
    has_new = cur.fetchone() is not None

    if old_row and has_new:
        cur.execute(
            """
            UPDATE videos SET
                filename = ?,
                extension = ?,
                size_bytes = ?,
                modified_time = ?,
                has_transcript = MAX(has_transcript, ?),
                has_vision = MAX(has_vision, ?),
                has_metadata = MAX(has_metadata, ?),
                indexed_in_qdrant = MAX(indexed_in_qdrant, ?)
            WHERE path = ?
            """,
            (
                Path(new_path).name,
                Path(new_path).suffix.lower(),
                stat.st_size,
                modified,
                old_row[0],
                old_row[1],
                old_row[2],
                old_row[3],
                new_path,
            ),
        )
        cur.execute("DELETE FROM videos WHERE path = ?", (old_path,))
    elif old_row:
        cur.execute(
            """
            UPDATE videos SET
                path = ?,
                filename = ?,
                extension = ?,
                size_bytes = ?,
                modified_time = ?
            WHERE path = ?
            """,
            (
                new_path,
                Path(new_path).name,
                Path(new_path).suffix.lower(),
                stat.st_size,
                modified,
                old_path,
            ),
        )
    elif not has_new:
        cur.execute(
            """
            INSERT INTO videos (
                path, filename, folder, extension, size_bytes, modified_time, scanned_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_path,
                Path(new_path).name,
                str(Path(new_path).parent),
                Path(new_path).suffix.lower(),
                stat.st_size,
                modified,
                datetime.datetime.now().isoformat(timespec="seconds"),
            ),
        )

    con.commit()
    con.close()

    try:
        sys.path.insert(0, str(APP_DIR))
        from search_engine import rename_qdrant_video_path

        rename_qdrant_video_path(old_path, new_path)
    except Exception as exc:
        print(f"Note: could not update search index path: {exc}", flush=True)


def reconcile_stale_records(db, folder):
    folder_path = Path(folder)
    legacy_review = folder_path / LEGACY_REVIEW_DIR
    prefix = str(folder_path).rstrip("\\/") + "%"

    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("SELECT path FROM videos WHERE path LIKE ?", (prefix,))
    paths = [row[0] for row in cur.fetchall()]
    con.close()

    stale_delete_paths = []
    for path_str in paths:
        path = Path(path_str)

        if legacy_review in path.parents:
            original = folder_path / path.relative_to(legacy_review)
            mkv = original.with_suffix(".mkv")
            if mkv.exists():
                migrate_video_record(db, path, mkv)
            else:
                stale_delete_paths.append(path_str)
            continue

        if path.suffix.lower() in OLD_EXTENSIONS:
            mkv = path.with_suffix(".mkv")
            if mkv.exists():
                migrate_video_record(db, path, mkv)

    if stale_delete_paths:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from catalog_db import CatalogWriter, delete_video_paths

        with CatalogWriter(db) as writer:
            delete_video_paths(writer, stale_delete_paths)


def find_candidates(folder):
    return sorted(
        p for p in Path(folder).rglob("*")
        if p.is_file() and not should_skip(p)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--status-file", required=True)
    add_normalize_args(parser)
    add_pipeline_control_args(parser)
    args = parser.parse_args()

    folder = Path(args.folder)
    status_file = Path(args.status_file)
    global_overwrite = args.overwrite == "true"
    overwrite = step_overwrite_from_args(args, "normalize", global_overwrite)
    skip_mode = skip_mode_from_args(args)
    allowlist = load_video_allowlist(args)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    convert_log = LOG_DIR / f"normalize_converted_{timestamp}.csv"
    failed_log = LOG_DIR / f"normalize_failed_{timestamp}.csv"

    status = read_status(status_file)
    status.update({
        "status": "running",
        "current": "Finding old-format videos...",
        "percent": 0,
        "processed": 0,
        "total": 0,
    })
    write_status(status_file, status)

    candidates = find_candidates(folder)
    files = [
        src for src in candidates
        if (allowlist is None or str(src) in allowlist)
        and should_process_video(args.db, src, "normalize", skip_mode, overwrite)
    ]
    total = len(files)

    print(f"Normalize job started for: {folder}", flush=True)
    print(f"Found {total} old-format files.", flush=True)
    if total == 0:
        sample_exts = sorted({
            p.suffix.lower()
            for p in folder.rglob("*")
            if p.is_file() and LEGACY_REVIEW_DIR not in p.parts
        })
        if sample_exts:
            print(
                f"No matching files. Folder contains extensions: {', '.join(sample_exts)}",
                flush=True,
            )
        print(
            "Supported normalize extensions: "
            + ", ".join(sorted(OLD_EXTENSIONS)),
            flush=True,
        )

    status.update({
        "total": total,
        "current": f"Found {total} old-format files.",
    })
    write_status(status_file, status)

    reconcile_stale_records(args.db, folder)
    print("Reconciled stale library records.", flush=True)

    converted_rows = []
    failed_rows = []

    for i, src in enumerate(files, start=1):
        if should_stop(status_file):
            status.update({
                "status": "stopped",
                "current": "Stopped by user.",
                "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
            })
            write_status(status_file, status)
            print("Stopped by user.", flush=True)
            sys.exit(0)

        dst = src.with_suffix(".mkv")

        status.update({
            "current": str(src),
            "processed": i - 1,
            "total": total,
            "percent": int(((i - 1) / total) * 100) if total else 100,
        })
        write_status(status_file, status)

        if dst.exists() and not overwrite:
            print(f"SKIP: MKV already exists: {dst}", flush=True)
            migrate_video_record(args.db, src, dst)
            if src.exists():
                delete_original_and_old_sidecars(src)
            converted_rows.append({
                "source_original": str(src),
                "new_mkv": str(dst),
                "copied_sidecars": "already converted",
            })
        else:
            print(f"\nCONVERTING: {src}", flush=True)
            print(f"TO:         {dst}", flush=True)

            ok, error = remux_to_mkv(src, dst)

            if not ok:
                print(f"FAILED CONVERT: {src}", flush=True)
                record_step_failure(src, "normalize", error or "convert failed")
                failed_rows.append({
                    "source": str(src),
                    "target": str(dst),
                    "error": error[-2000:] if error else "",
                })
            elif not ffprobe_ok(dst):
                print(f"FAILED VERIFY: {dst}", flush=True)
                record_step_failure(src, "normalize", "ffprobe verification failed")
                failed_rows.append({
                    "source": str(src),
                    "target": str(dst),
                    "error": "ffprobe verification failed",
                })
            else:
                copied_sidecars = copy_sidecars(src, dst)
                delete_original_and_old_sidecars(src)
                clear_step_failure(src, "normalize")

                try:
                    migrate_video_record(args.db, src, dst)
                except sqlite3.IntegrityError:
                    print(f"DB reconcile retry for: {src}", flush=True)
                    reconcile_stale_records(args.db, folder)
                    migrate_video_record(args.db, src, dst)

                converted_rows.append({
                    "source_original": str(src),
                    "new_mkv": str(dst),
                    "copied_sidecars": "; ".join(copied_sidecars),
                })

        percent = int((i / total) * 100) if total else 100
        status.update({
            "processed": i,
            "total": total,
            "percent": percent,
            "current": str(src),
        })
        write_status(status_file, status)

    if converted_rows or failed_rows:
        print("\nWriting logs...", flush=True)

        with open(convert_log, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["source_original", "new_mkv", "copied_sidecars"],
            )
            writer.writeheader()
            writer.writerows(converted_rows)

        with open(failed_log, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["source", "target", "error"],
            )
            writer.writeheader()
            writer.writerows(failed_rows)

        print(f"Converted log: {convert_log}", flush=True)
        print(f"Failed log:    {failed_log}", flush=True)

    summary = (
        f"Normalize complete. Converted: {len(converted_rows)}, failed: {len(failed_rows)}."
    )

    status.update({
        "status": "complete",
        "percent": 100,
        "current": summary,
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    write_status(status_file, status)

    print(f"\nDone. Converted: {len(converted_rows)}", flush=True)
    print(f"Failed:    {len(failed_rows)}", flush=True)


if __name__ == "__main__":
    from job_utils import run_script_main, status_file_from_argv

    run_script_main(main, status_file_from_argv())
