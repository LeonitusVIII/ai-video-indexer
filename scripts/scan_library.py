import argparse
import datetime
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_utils import find_videos, read_status, should_stop, write_status

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from app_db import prune_missing_videos, sync_sidecar_flags


def upsert_video(db, video_path):
    stat = video_path.stat()

    con = sqlite3.connect(db)
    cur = con.cursor()

    cur.execute("""
        INSERT INTO videos (
            path,
            filename,
            folder,
            extension,
            size_bytes,
            modified_time,
            scanned_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            filename=excluded.filename,
            folder=excluded.folder,
            extension=excluded.extension,
            size_bytes=excluded.size_bytes,
            modified_time=excluded.modified_time,
            scanned_at=excluded.scanned_at
    """, (
        str(video_path),
        video_path.name,
        str(video_path.parent),
        video_path.suffix.lower(),
        stat.st_size,
        datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        datetime.datetime.now().isoformat(timespec="seconds")
    ))

    con.commit()
    con.close()
    sync_sidecar_flags(db, video_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--status-file", required=True)
    args = parser.parse_args()

    folder = Path(args.folder)
    status_file = Path(args.status_file)

    status = read_status(status_file)
    status.update({
        "status": "running",
        "current": "Finding video files...",
        "percent": 0,
        "processed": 0,
        "total": 0
    })
    write_status(status_file, status)

    print(f"Scanning folder: {folder}", flush=True)

    videos = find_videos(folder)
    total = len(videos)

    status.update({
        "current": f"Found {total} video files.",
        "total": total
    })
    write_status(status_file, status)

    print(f"Found {total} video files.", flush=True)

    scanned_paths = {str(video) for video in videos}
    prefix = str(folder).rstrip("\\/") + "%"
    con = sqlite3.connect(args.db)
    cur = con.cursor()
    cur.execute("SELECT path FROM videos WHERE path LIKE ?", (prefix,))
    for (existing_path,) in cur.fetchall():
        if existing_path not in scanned_paths:
            cur.execute("DELETE FROM videos WHERE path = ?", (existing_path,))
            print(f"Removed missing from catalog: {existing_path}", flush=True)
    con.commit()
    con.close()

    removed_ghosts = prune_missing_videos(args.db, folder_prefix=str(folder))
    for path in removed_ghosts:
        if path not in scanned_paths:
            print(f"Pruned ghost entry (file missing): {path}", flush=True)

    for i, video in enumerate(videos, start=1):
        if should_stop(status_file):
            status.update({
                "status": "stopped",
                "current": "Stopped by user.",
                "finished_at": datetime.datetime.now().isoformat(timespec="seconds")
            })
            write_status(status_file, status)
            print("Stopped by user.", flush=True)
            sys.exit(0)

        upsert_video(args.db, video)

        percent = int((i / total) * 100) if total else 100

        status.update({
            "processed": i,
            "total": total,
            "percent": percent,
            "current": str(video)
        })
        write_status(status_file, status)

        print(f"[{i}/{total}] {video}", flush=True)

    status.update({
        "status": "complete",
        "percent": 100,
        "current": f"Scan complete. Pruned {len(removed_ghosts)} ghost entries.",
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds")
    })
    write_status(status_file, status)

    print(f"Scan complete. Pruned {len(removed_ghosts)} ghost entries.", flush=True)


if __name__ == "__main__":
    main()
