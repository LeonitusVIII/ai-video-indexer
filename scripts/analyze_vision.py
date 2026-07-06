import argparse
import datetime
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_utils import (
    add_processing_args,
    compute_frame_times,
    extract_frame_image,
    find_videos,
    get_video_duration,
    overwrite_from_args,
    read_status,
    should_stop,
    update_video_flag,
    use_gpu_from_args,
    vision_json_path,
    vision_model_config_from_args,
    clear_item_progress,
    set_item_progress,
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
    vision_output_complete,
)


VISION_PROMPT = (
    "Describe this video frame in detail for search indexing. "
    "Include people, activities, locations, objects, clothing, and mood."
)


def load_vision_model(model_config, use_gpu):
    try:
        import torch
        from transformers import AutoProcessor
        from qwen_vl_utils import process_vision_info
    except Exception as e:
        raise RuntimeError(
            "Missing vision dependencies. Run Install / Update AI Dependencies "
            f"from the Tools/System tab. ({e})"
        ) from e

    family = model_config["family"]
    model_id = model_config["model_id"]
    if family == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration

        model_cls = Qwen2_5_VLForConditionalGeneration
    elif family == "qwen2_vl":
        from transformers import Qwen2VLForConditionalGeneration

        model_cls = Qwen2VLForConditionalGeneration
    else:
        raise RuntimeError(f"Unsupported vision model family: {family}")

    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Loading vision model: {model_id} on {device}", flush=True)

    model = model_cls.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor, process_vision_info, device, model_id


def describe_frame(model, processor, process_vision_info, image_path):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": VISION_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0].strip()


def load_resume_vision(output_file, *, model_id=None, interval_seconds=None, min_frames=None):
    if not output_file.exists():
        return None
    try:
        data = json.loads(output_file.read_text(encoding="utf-8"))
    except Exception:
        return None
    if data.get("status") == "complete":
        return data
    if not data.get("frames"):
        return None
    if model_id is not None:
        if (
            data.get("model") != model_id
            or int(data.get("frame_interval_seconds") or 0) != interval_seconds
            or int(data.get("min_frames_per_video") or 0) != min_frames
        ):
            return None
    return data


def build_vision_payload(
    video,
    model_id,
    interval_seconds,
    min_frames,
    frames,
    *,
    status="in_progress",
):
    return {
        "video": str(video),
        "model": model_id,
        "duration_seconds": round(get_video_duration(video), 3)
        if status == "complete"
        else round(
            max((frame.get("time", 0) for frame in frames), default=0),
            3,
        ),
        "frame_interval_seconds": interval_seconds,
        "min_frames_per_video": min_frames,
        "frames": frames,
        "status": status,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def write_vision_output(output_file, payload):
    output_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze_video(
    video,
    model,
    processor,
    process_vision_info,
    interval_seconds,
    min_frames,
    model_id,
    output_file,
    resume_data=None,
    status_file=None,
    status=None,
):
    duration = get_video_duration(video)
    frame_times = compute_frame_times(duration, interval_seconds, min_frames)
    existing_frames = []
    if resume_data:
        existing_frames = list(resume_data.get("frames") or [])
    done_times = {frame["time"] for frame in existing_frames}
    frames = list(existing_frames)
    frame_total = len(frame_times)
    video_name = Path(video).name
    pending_times = [t for t in frame_times if t not in done_times]

    if not pending_times and existing_frames:
        return build_vision_payload(
            video,
            model_id,
            interval_seconds,
            min_frames,
            frames,
            status="complete",
        )

    print(
        f"  Vision frames: {len(pending_times)} remaining of {frame_total}",
        flush=True,
    )

    with tempfile.TemporaryDirectory(prefix="video_vision_") as tmp:
        tmp_dir = Path(tmp)

        for frame_idx, frame_time in enumerate(pending_times, start=1):
            if status_file is not None and status is not None:
                completed = frame_total - len(pending_times) + frame_idx - 1
                set_item_progress(
                    status,
                    status_file,
                    video_name,
                    completed,
                    frame_total,
                )

            image_path = tmp_dir / f"frame_{frame_time:.3f}.jpg"
            extract_frame_image(video, frame_time, image_path)
            description = describe_frame(
                model, processor, process_vision_info, image_path
            )
            frames.append({
                "time": frame_time,
                "description": description,
            })
            print(f"  [{frame_time:.1f}s] {description[:120]}", flush=True)

            partial = build_vision_payload(
                video,
                model_id,
                interval_seconds,
                min_frames,
                frames,
                status="in_progress",
            )
            partial["duration_seconds"] = round(duration, 3)
            write_vision_output(output_file, partial)

            if status_file is not None and status is not None:
                completed = frame_total - len(pending_times) + frame_idx
                set_item_progress(
                    status,
                    status_file,
                    video_name,
                    completed,
                    frame_total,
                )

    if status_file is not None and status is not None:
        clear_item_progress(status, status_file)

    return build_vision_payload(
        video,
        model_id,
        interval_seconds,
        min_frames,
        frames,
        status="complete",
    )


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
    model_config = vision_model_config_from_args(args)
    model_id = model_config["model_id"]
    global_overwrite = overwrite_from_args(args)
    overwrite = step_overwrite_from_args(args, "vision", global_overwrite)
    skip_mode = skip_mode_from_args(args)
    allowlist = load_video_allowlist(args)
    interval_seconds = max(1, int(args.vision_interval))
    min_frames = max(1, int(args.min_frames))

    status = read_status(status_file)
    status.update({
        "status": "running",
        "current": "Finding videos for vision analysis...",
        "percent": 0,
        "processed": 0,
        "total": 0,
        "vision_interval": interval_seconds,
        "min_frames": min_frames,
        "vision_model": model_id,
    })
    write_status(status_file, status)

    print(
        f"Vision analysis started for: {folder} "
        f"(model={model_id}, interval={interval_seconds}s, min_frames={min_frames})",
        flush=True,
    )

    videos = filter_videos_for_step(
        find_videos(folder),
        args.db,
        "vision",
        skip_mode,
        overwrite,
        allowlist,
    )
    total = len(videos)

    status.update({
        "total": total,
        "current": f"Found {total} video files.",
    })
    write_status(status_file, status)
    print(f"Found {total} video files.", flush=True)

    if total == 0:
        status.update({
            "status": "complete",
            "percent": 100,
            "current": "No videos found.",
            "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        write_status(status_file, status)
        return

    try:
        model, processor, process_vision_info, device, model_id = load_vision_model(
            model_config,
            use_gpu,
        )
    except Exception as e:
        status.update({
            "status": "failed",
            "current": str(e),
            "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        write_status(status_file, status)
        print(f"FAILED: {e}", flush=True)
        sys.exit(1)

    print(f"Vision model ready on {device}", flush=True)

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

        output_file = vision_json_path(video)
        row = get_video_row(args.db, video)

        if (
            output_file.exists()
            and not overwrite
            and vision_output_complete(output_file)
            and step_status(row, "vision") == "complete"
        ):
            print(f"Skipping existing vision file: {video}", flush=True)
            update_video_flag(args.db, video, "has_vision")
            clear_item_progress(status, status_file)
        else:
            resume_data = None
            if output_file.exists() and not overwrite:
                resume_data = load_resume_vision(
                    output_file,
                    model_id=model_id,
                    interval_seconds=interval_seconds,
                    min_frames=min_frames,
                )
                if resume_data:
                    print(f"Resuming partial vision output: {video}", flush=True)
                elif output_file.exists():
                    try:
                        partial = json.loads(output_file.read_text(encoding="utf-8"))
                    except Exception:
                        partial = {}
                    if partial.get("frames") and partial.get("status") != "complete":
                        print(
                            f"Partial vision output uses different settings; restarting: {video}",
                            flush=True,
                        )

            status.update({
                "current": f"Analyzing vision: {video}",
                "processed": i - 1,
                "total": total,
                "percent": int(((i - 1) / total) * 100) if total else 100,
            })
            clear_item_progress(status, status_file)
            write_status(status_file, status)
            print(f"\nAnalyzing: {video}", flush=True)

            try:
                result = analyze_video(
                    video,
                    model,
                    processor,
                    process_vision_info,
                    interval_seconds,
                    min_frames,
                    model_id,
                    output_file,
                    resume_data=resume_data,
                    status_file=status_file,
                    status=status,
                )
                write_vision_output(output_file, result)
                update_video_flag(args.db, video, "has_vision")
                clear_step_failure(video, "vision")
                print(f"Completed: {video}", flush=True)
            except Exception as e:
                record_step_failure(video, "vision", str(e))
                print(f"FAILED: {video}", flush=True)
                print(str(e), flush=True)

        percent = int((i / total) * 100) if total else 100
        clear_item_progress(status, status_file)
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
        "current": "Vision analysis complete.",
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    write_status(status_file, status)
    print("Vision analysis complete.", flush=True)


if __name__ == "__main__":
    main()
