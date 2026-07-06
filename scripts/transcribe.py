import argparse
from pathlib import Path
import sqlite3
import json
import datetime
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_utils import (
    add_processing_args,
    find_videos,
    overwrite_from_args,
    read_status,
    should_stop,
    transcription_model_from_args,
    use_gpu_from_args,
    write_status,
)
from pipeline_utils import (
    add_pipeline_control_args,
    clear_step_failure,
    filter_videos_for_step,
    get_video_row,
    load_video_allowlist,
    record_step_failure,
    skip_mode_from_args,
    step_overwrite_from_args,
    step_status,
)


def update_video_transcript_status(db, video_path):
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(
        "UPDATE videos SET has_transcript = 1 WHERE path = ?",
        (str(video_path),)
    )
    con.commit()
    con.close()


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
    use_gpu = use_gpu_from_args(args)
    global_overwrite = overwrite_from_args(args)
    overwrite = step_overwrite_from_args(args, "transcribe", global_overwrite)
    skip_mode = skip_mode_from_args(args)
    allowlist = load_video_allowlist(args)

    status = read_status(status_file)
    status.update({
        "status": "running",
        "current": "Finding videos for transcription...",
        "percent": 0,
        "processed": 0,
        "total": 0
    })
    write_status(status_file, status)

    print(f"Transcription job started for: {folder}", flush=True)

    videos = filter_videos_for_step(
        find_videos(folder),
        args.db,
        "transcribe",
        skip_mode,
        overwrite,
        allowlist,
    )
    total = len(videos)

    status.update({
        "total": total,
        "current": f"Found {total} video files."
    })
    write_status(status_file, status)

    print(f"Found {total} video files.", flush=True)

    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        status.update({
            "status": "failed",
            "current": f"Could not import faster_whisper: {e}",
            "finished_at": datetime.datetime.now().isoformat(timespec="seconds")
        })
        write_status(status_file, status)
        print(f"FAILED: Could not import faster_whisper: {e}", flush=True)
        sys.exit(1)

    print(f"Loading Whisper model: {transcription_model_from_args(args)}", flush=True)

    device = "cuda" if use_gpu else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    whisper_model = transcription_model_from_args(args)

    try:
        model = WhisperModel(
            whisper_model,
            device=device,
            compute_type=compute_type
        )
    except Exception as e:
        status.update({
            "status": "failed",
            "current": f"Could not load Whisper model on {device}: {e}",
            "finished_at": datetime.datetime.now().isoformat(timespec="seconds")
        })
        write_status(status_file, status)
        print(f"FAILED: Could not load Whisper model on {device}: {e}", flush=True)
        sys.exit(1)

    for i, video in enumerate(videos, start=1):
        current_status = read_status(status_file)

        if current_status.get("stop_requested"):
            status.update({
                "status": "stopped",
                "current": "Stopped by user.",
                "finished_at": datetime.datetime.now().isoformat(timespec="seconds")
            })
            write_status(status_file, status)
            print("Stopped by user.", flush=True)
            sys.exit(0)

        txt_file = video.with_suffix(video.suffix + ".transcript.txt")
        srt_file = video.with_suffix(video.suffix + ".whisper.srt")
        json_file = video.with_suffix(video.suffix + ".transcript.json")
        row = get_video_row(args.db, video)

        if (
            txt_file.exists()
            and srt_file.exists()
            and json_file.exists()
            and not overwrite
            and step_status(row, "transcribe") == "complete"
        ):
            print(f"Skipping existing transcript: {video}", flush=True)
            update_video_transcript_status(args.db, video)
        else:
            status.update({
                "current": f"Transcribing: {video}",
                "processed": i - 1,
                "total": total,
                "percent": int(((i - 1) / total) * 100) if total else 100
            })
            write_status(status_file, status)

            print(f"Transcribing: {video}", flush=True)

            try:
                segments, info = model.transcribe(
                    str(video),
                    beam_size=5,
                    vad_filter=True,
                    word_timestamps=True
                )

                transcript_segments = []
                txt_lines = []
                srt_blocks = []

                for idx, segment in enumerate(segments, start=1):
                    text = segment.text.strip()

                    transcript_segments.append({
                        "id": idx,
                        "start": segment.start,
                        "end": segment.end,
                        "text": text
                    })

                    txt_lines.append(
                        f"[{segment.start:.2f} - {segment.end:.2f}] {text}"
                    )

                    def srt_time(seconds):
                        hours = int(seconds // 3600)
                        minutes = int((seconds % 3600) // 60)
                        secs = int(seconds % 60)
                        millis = int((seconds - int(seconds)) * 1000)
                        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

                    srt_blocks.append(
                        f"{idx}\n{srt_time(segment.start)} --> {srt_time(segment.end)}\n{text}\n"
                    )

                txt_file.write_text("\n".join(txt_lines), encoding="utf-8")
                srt_file.write_text("\n".join(srt_blocks), encoding="utf-8")

                json_file.write_text(
                    json.dumps({
                        "video": str(video),
                        "language": info.language,
                        "language_probability": info.language_probability,
                        "segments": transcript_segments
                    }, indent=2),
                    encoding="utf-8"
                )

                update_video_transcript_status(args.db, video)
                clear_step_failure(video, "transcribe")

                print(f"Completed: {video}", flush=True)

            except Exception as e:
                record_step_failure(video, "transcribe", str(e))
                print(f"FAILED: {video}", flush=True)
                print(str(e), flush=True)

        percent = int((i / total) * 100) if total else 100
        status.update({
            "processed": i,
            "total": total,
            "percent": percent,
            "current": str(video)
        })
        write_status(status_file, status)

    status.update({
        "status": "complete",
        "percent": 100,
        "current": "Transcription complete.",
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds")
    })
    write_status(status_file, status)

    print("Transcription complete.", flush=True)


if __name__ == "__main__":
    main()