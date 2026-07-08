import argparse
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_utils import (
    add_processing_args,
    find_videos,
    get_video_duration,
    metadata_json_path,
    overwrite_from_args,
    read_status,
    should_stop,
    transcript_json_path,
    vision_json_path,
    write_status,
)
from person_tags import extract_people_tags
from catalog_db import CatalogWriter, update_video_metadata
from pipeline_utils import (
    add_pipeline_control_args,
    clear_step_failure,
    filter_videos_for_step,
    get_video_row,
    load_video_rows_map,
    load_video_allowlist,
    record_step_failure,
    skip_mode_from_args,
    step_overwrite_from_args,
    step_status,
)


def load_json(path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_search_chunks(transcript_data, vision_data, interval_seconds):
    chunks = []
    segments = (transcript_data or {}).get("segments", [])
    frames = (vision_data or {}).get("frames", [])

    if segments:
        for segment in segments:
            start = float(segment.get("start", 0))
            end = float(segment.get("end", start))
            text = (segment.get("text") or "").strip()

            nearby_vision = [
                f.get("description", "").strip()
                for f in frames
                if start <= float(f.get("time", 0)) <= end
            ]
            nearby_vision = [v for v in nearby_vision if v]

            combined_parts = []
            if text:
                combined_parts.append(text)
            if nearby_vision:
                combined_parts.append("Visual: " + " ".join(nearby_vision))

            if not combined_parts:
                continue

            chunks.append({
                "start": start,
                "end": end,
                "text": " ".join(combined_parts),
                "sources": [
                    s for s, present in [
                        ("transcript", bool(text)),
                        ("vision", bool(nearby_vision)),
                    ] if present
                ],
            })
        return chunks

    if frames:
        for frame in frames:
            start = float(frame.get("time", 0))
            description = (frame.get("description") or "").strip()
            if not description:
                continue
            chunks.append({
                "start": start,
                "end": start + interval_seconds,
                "text": f"Visual: {description}",
                "sources": ["vision"],
            })

    return chunks


def build_metadata_for_video(video, interval_seconds, *, enable_person_tags=True):
    transcript_data = load_json(transcript_json_path(video))
    vision_data = load_json(vision_json_path(video))

    try:
        duration_seconds = get_video_duration(video)
    except Exception:
        duration_seconds = (vision_data or {}).get("duration_seconds", 0)

    people_tags = []
    if enable_person_tags:
        people_tags = extract_people_tags(vision_data, transcript_data)

    search_chunks = build_search_chunks(
        transcript_data, vision_data, interval_seconds
    )
    if people_tags:
        search_chunks.insert(0, {
            "start": 0.0,
            "end": max(float(duration_seconds or 0), 1.0),
            "text": "People: " + ", ".join(people_tags),
            "sources": ["person_tags"],
        })

    stat = video.stat()
    return {
        "video": str(video),
        "filename": video.name,
        "folder": str(video.parent),
        "size_bytes": stat.st_size,
        "modified_time": datetime.datetime.fromtimestamp(
            stat.st_mtime
        ).isoformat(timespec="seconds"),
        "duration_seconds": round(duration_seconds, 3),
        "has_transcript": transcript_data is not None,
        "has_vision": vision_data is not None,
        "transcript_language": (transcript_data or {}).get("language"),
        "people_tags": people_tags,
        "transcript": transcript_data,
        "vision": vision_data,
        "search_chunks": search_chunks,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--status-file", required=True)
    add_processing_args(parser)
    add_pipeline_control_args(parser)
    args = parser.parse_args()

    folder = Path(args.folder)
    status_file = Path(args.status_file)
    global_overwrite = overwrite_from_args(args)
    overwrite = step_overwrite_from_args(args, "metadata", global_overwrite)
    skip_mode = skip_mode_from_args(args)
    allowlist = load_video_allowlist(args)
    interval_seconds = max(1, int(args.vision_interval))

    status = read_status(status_file)
    status.update({
        "status": "running",
        "current": "Finding videos for metadata build...",
        "percent": 0,
        "processed": 0,
        "total": 0,
    })
    write_status(status_file, status)

    print(f"Metadata build started for: {folder}", flush=True)

    videos = filter_videos_for_step(
        find_videos(folder),
        args.db,
        "metadata",
        skip_mode,
        overwrite,
        allowlist,
    )
    total = len(videos)
    row_cache = load_video_rows_map(args.db, videos)

    status.update({
        "total": total,
        "current": f"Found {total} video files.",
    })
    write_status(status_file, status)
    print(f"Found {total} video files.", flush=True)

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

            output_file = metadata_json_path(video)
            row = get_video_row(args.db, video, row_cache=row_cache)

            if output_file.exists() and not overwrite and step_status(row, "metadata") == "complete":
                print(f"Skipping existing metadata: {video}", flush=True)
                update_video_metadata(writer, video)
            else:
                status.update({
                    "current": f"Building metadata: {video}",
                    "processed": i - 1,
                    "total": total,
                    "percent": int(((i - 1) / total) * 100) if total else 100,
                })
                write_status(status_file, status)
                print(f"\nBuilding metadata: {video}", flush=True)

                try:
                    metadata = build_metadata_for_video(video, interval_seconds)
                    output_file.write_text(
                        json.dumps(metadata, indent=2), encoding="utf-8"
                    )
                    update_video_metadata(writer, video, metadata.get("people_tags"))
                    clear_step_failure(video, "metadata")
                    print(
                        f"Created {len(metadata['search_chunks'])} search chunks",
                        flush=True,
                    )
                except Exception as e:
                    record_step_failure(video, "metadata", str(e))
                    print(f"FAILED: {video}", flush=True)
                    print(str(e), flush=True)

            percent = int((i / total) * 100) if total else 100
            status.update({
                "processed": i,
                "total": total,
                "percent": percent,
                "current": str(video),
            })
            write_status(status_file, status)

    status.update({
        "status": "complete",
        "percent": 100,
        "current": "Metadata build complete.",
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    write_status(status_file, status)
    print("Metadata build complete.", flush=True)


if __name__ == "__main__":
    from job_utils import run_script_main, status_file_from_argv

    run_script_main(main, status_file_from_argv())
