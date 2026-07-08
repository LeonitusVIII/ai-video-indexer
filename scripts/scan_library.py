import argparse
import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from catalog_db import CatalogWriter, upsert_scanned_video
from job_utils import find_videos, read_status, should_stop, write_status

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR.parent))

from app_db import prune_missing_videos


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
        "total": 0,
    })
    write_status(status_file, status)

    print(f"Scanning folder: {folder}", flush=True)

    videos = find_videos(folder)
    total = len(videos)

    status.update({
        "current": f"Found {total} video files.",
        "total": total,
    })
    write_status(status_file, status)
    print(f"Found {total} video files.", flush=True)

    scanned_paths = {str(video) for video in videos}
    prefix = str(folder).rstrip("\\/") + "%"

    with CatalogWriter(args.db) as writer:
        cur = writer.con.cursor()
        cur.execute("SELECT path FROM videos WHERE path LIKE ?", (prefix,))
        stale_paths = []
        for (existing_path,) in cur.fetchall():
            if existing_path not in scanned_paths:
                stale_paths.append(existing_path)
        if stale_paths:
            from catalog_db import delete_video_paths
            delete_video_paths(writer, stale_paths)
            for existing_path in stale_paths:
                print(f"Removed missing from catalog: {existing_path}", flush=True)

    removed_ghosts = prune_missing_videos(args.db, folder_prefix=str(folder))
    for path in removed_ghosts:
        if path not in scanned_paths:
            print(f"Pruned ghost entry (file missing): {path}", flush=True)

    with CatalogWriter(args.db) as writer:
        for i, video in enumerate(videos, start=1):
            if should_stop(status_file):
                status.update({
                    "status": "stopped",
                    "current": "Stopped by user.",
                    "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
                })
                write_status(status_file, status)
                print("Stopped by user.", flush=True)
                sys.exit(0)

            upsert_scanned_video(writer, video)

            percent = int((i / total) * 100) if total else 100
            status.update({
                "processed": i,
                "total": total,
                "percent": percent,
                "current": str(video),
            })
            write_status(status_file, status)
            print(f"[{i}/{total}] {video}", flush=True)

    status.update({
        "status": "complete",
        "percent": 100,
        "current": f"Scan complete. Pruned {len(removed_ghosts)} ghost entries.",
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    write_status(status_file, status)
    print(f"Scan complete. Pruned {len(removed_ghosts)} ghost entries.", flush=True)


if __name__ == "__main__":
    from job_utils import run_script_main, status_file_from_argv

    run_script_main(main, status_file_from_argv())
