import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_utils import add_processing_args, read_status, should_stop, write_status
from pipeline_utils import clear_all_failures, load_failures

APP_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = APP_DIR / "data"
CONFIG_FILE = APP_DIR / "config.json"
RESUME_FILE = DATA_DIR / "pipeline_resume.json"
PYTHON = APP_DIR / "venv" / "Scripts" / "python.exe"
SCRIPTS_DIR = Path(__file__).resolve().parent

SCAN_STEP = ("scan", "scan_library.py", "Rescan library")

PIPELINE_STEPS = [
    ("normalize", "normalize_videos.py", "Normalize old videos"),
    ("transcribe", "transcribe.py", "Transcribe videos"),
    ("vision", "analyze_vision.py", "Analyze vision"),
    ("metadata", "build_metadata.py", "Build metadata"),
    ("index", "index_qdrant.py", "Index search DB"),
]


def add_pipeline_args(parser):
    add_processing_args(parser)
    parser.add_argument("--folders-file", default="")
    parser.add_argument("--resume", choices=["true", "false"], default="false")
    parser.add_argument(
        "--scan-after",
        choices=["true", "false"],
        default="true",
        help="Run library scan after all processing steps for each folder.",
    )
    parser.add_argument(
        "--skip-mode",
        choices=["all", "missing_only", "stale_only", "incomplete_only"],
        default="all",
    )
    parser.add_argument("--videos-file", default="")
    parser.add_argument("--step-overwrite", default="")
    for step_key, _, _ in PIPELINE_STEPS:
        parser.add_argument(
            f"--step-{step_key.replace('_', '-')}",
            choices=["true", "false"],
            default="true",
        )


def step_enabled(args, step_key):
    return getattr(args, f"step_{step_key}", "true") == "true"


def enabled_steps(args):
    steps = [
        (key, script, label)
        for key, script, label in PIPELINE_STEPS
        if step_enabled(args, key)
    ]
    if getattr(args, "scan_after", "true") == "true":
        steps.append(SCAN_STEP)
    return steps


def load_folders(args):
    if args.folders_file:
        path = Path(args.folders_file)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    return [args.folder]


def load_resume_state():
    if not RESUME_FILE.exists():
        return None
    try:
        return json.loads(RESUME_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_resume_state(state):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESUME_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def clear_resume_state():
    if RESUME_FILE.exists():
        RESUME_FILE.unlink()


def resume_start_point(args, folders, steps):
    if args.resume != "true":
        return 0, 0

    saved = load_resume_state()
    if not saved:
        return 0, 0

    saved_folders = saved.get("folders", [])
    saved_steps = saved.get("step_keys", [])
    current_step_keys = [key for key, _, _ in steps]

    if saved_folders != folders or saved_steps != current_step_keys:
        return 0, 0

    if saved.get("status") not in {"failed", "stopped"}:
        return 0, 0

    return int(saved.get("folder_index", 0)), int(saved.get("step_index", 0))


def build_step_cmd(args, folder, script_name, step_status_file):
    cmd = [
        str(PYTHON),
        str(SCRIPTS_DIR / script_name),
        "--folder", folder,
        "--db", args.db,
        "--status-file", str(step_status_file),
    ]

    if script_name == "scan_library.py":
        return cmd

    if script_name == "normalize_videos.py":
        cmd.extend([
            "--overwrite", args.overwrite,
        ])
    else:
        cmd.extend([
            "--use-gpu", args.use_gpu,
            "--overwrite", args.overwrite,
            "--vision-interval", str(args.vision_interval),
            "--min-frames", str(args.min_frames),
            "--transcription-model", args.transcription_model,
            "--vision-model", args.vision_model,
        ])

    cmd.extend(["--skip-mode", args.skip_mode])
    if args.step_overwrite:
        cmd.extend(["--step-overwrite", args.step_overwrite])
    if args.videos_file:
        cmd.extend(["--videos-file", args.videos_file])

    return cmd


def total_work_units(num_folders, num_steps):
    return max(num_folders * num_steps, 1)


def update_pipeline_status(
    status_file,
    status,
    completed_units,
    total_units,
    folder_label,
    step_label,
    child_status,
    step_key=None,
):
    child = child_status or {}
    child_percent = int(child.get("percent", 0))
    step_fraction = (child_percent / 100) if total_units else 0
    overall = int(((completed_units + step_fraction) / total_units) * 100)

    status.update({
        "percent": min(overall, 99),
        "processed": completed_units,
        "total": total_units,
        "current": f"{folder_label} — {step_label} — {child.get('current', '')}",
        "pipeline_step": step_label,
        "pipeline_step_key": step_key or status.get("pipeline_step_key", ""),
        "pipeline_folder": folder_label,
        "step_percent": child_percent,
        "step_processed": int(child.get("processed", 0) or 0),
        "step_total": int(child.get("total", 0) or 0),
        "step_current": child.get("current", ""),
        "item_percent": int(child.get("item_percent", 0) or 0),
        "item_processed": int(child.get("item_processed", 0) or 0),
        "item_total": int(child.get("item_total", 0) or 0),
        "item_label": child.get("item_label", ""),
    })
    write_status(status_file, status)


def propagate_stop(pipeline_status_file, step_status_file):
    if should_stop(pipeline_status_file):
        step_status = read_status(step_status_file)
        step_status["stop_requested"] = True
        write_status(step_status_file, step_status)
        return True
    return False


def run_step(
    args,
    folder,
    step_key,
    script_name,
    step_label,
    status_file,
    log_file,
    completed_units,
    total_units,
    folder_label,
):
    steps_dir = status_file.parent / "steps"
    steps_dir.mkdir(exist_ok=True)
    step_status_file = steps_dir / f"{status_file.stem}_{step_key}.json"

    cmd = build_step_cmd(args, folder, script_name, step_status_file)

    with open(log_file, "a", encoding="utf-8") as log:
        log.write(f"\n{'=' * 60}\n")
        log.write(f"FOLDER: {folder}\n")
        log.write(f"STEP: {step_label}\n")
        log.write(f"{'=' * 60}\n\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(APP_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

        pipeline_status = read_status(status_file)
        pipeline_status["pid"] = proc.pid
        write_status(status_file, pipeline_status)

        while proc.poll() is None:
            if propagate_stop(status_file, step_status_file):
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
                return "stopped"

            child_status = read_status(step_status_file)
            pipeline_status = read_status(status_file)
            update_pipeline_status(
                status_file,
                pipeline_status,
                completed_units,
                total_units,
                folder_label,
                step_label,
                child_status,
                step_key=step_key,
            )
            time.sleep(1)

    child_status = read_status(step_status_file)

    if should_stop(status_file) or child_status.get("status") == "stopped":
        return "stopped"

    if proc.returncode != 0 or child_status.get("status") == "failed":
        return "failed"

    return "complete"


def load_notification_settings():
    if not CONFIG_FILE.exists():
        return {}
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return config.get("notifications") or {}


def count_pipeline_failures():
    failures = load_failures()
    return sum(len(steps) for steps in failures.values())


def send_discord(event, folder_label, details, log_file):
    settings = load_notification_settings()
    webhook = (settings.get("discord_webhook_url") or "").strip()
    if not webhook:
        return

    flag_name = {
        "complete": "notify_on_complete",
        "complete_with_failures": "notify_on_complete",
        "failed": "notify_on_failed",
        "stopped": "notify_on_stopped",
    }.get(event)
    if flag_name and not settings.get(flag_name, True):
        return

    sys.path.insert(0, str(APP_DIR))
    from notifications import notify_pipeline_event

    ok, err = notify_pipeline_event(
        webhook,
        event,
        folder_label,
        details,
        log_file=str(log_file),
    )
    if not ok and err:
        print(f"Discord notification failed: {err}", flush=True)


def finalize_status(status_file, status, event, completed_units, total_units, message, resume_state=None):
    status.update({
        "status": event,
        "percent": 100 if event == "complete" else int((completed_units / total_units) * 100),
        "processed": completed_units,
        "total": total_units,
        "current": message,
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    write_status(status_file, status)

    if resume_state is not None:
        resume_state["status"] = event
        resume_state["folder_index"] = resume_state.get("folder_index", 0)
        resume_state["step_index"] = resume_state.get("step_index", 0)
        if event == "complete":
            clear_resume_state()
        else:
            save_resume_state(resume_state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True)
    parser.add_argument("--db", required=True)
    parser.add_argument("--status-file", required=True)
    add_pipeline_args(parser)
    args = parser.parse_args()

    status_file = Path(args.status_file)
    initial_status = read_status(status_file)
    log_file = Path(initial_status.get("log_file", ""))

    folders = load_folders(args)
    steps = enabled_steps(args)
    step_keys = [key for key, _, _ in steps]
    total_units = total_work_units(len(folders), len(steps))

    if args.resume != "true":
        clear_all_failures()

    start_folder_index, start_step_index = resume_start_point(args, folders, steps)

    status = initial_status
    status.update({
        "status": "running",
        "percent": 0,
        "processed": 0,
        "total": total_units,
        "current": "Starting pipeline...",
        "pipeline_steps": [label for _, _, label in steps],
        "pipeline_folders": folders,
        "started_at": status.get("started_at") or datetime.datetime.now().isoformat(timespec="seconds"),
        "finished_at": "",
        "resume_mode": args.resume == "true",
        "skip_mode": args.skip_mode,
    })
    write_status(status_file, status)

    resume_state = {
        "folders": folders,
        "step_keys": step_keys,
        "folder_index": start_folder_index,
        "step_index": start_step_index,
        "status": "running",
        "job_id": status.get("job_id", ""),
    }
    save_resume_state(resume_state)

    folder_label = folders[0] if len(folders) == 1 else f"{len(folders)} folders"

    print(f"Pipeline started for {len(folders)} folder(s)", flush=True)
    if args.resume == "true" and (start_folder_index or start_step_index):
        print(
            f"Resuming from folder {start_folder_index + 1}/{len(folders)}, "
            f"step {start_step_index + 1}/{len(steps)}",
            flush=True,
        )

    if not steps:
        finalize_status(status_file, status, "failed", 0, total_units, "No pipeline steps selected.")
        send_discord("failed", folder_label, "No pipeline steps selected.", log_file)
        sys.exit(1)

    completed_units = (start_folder_index * len(steps)) + start_step_index

    for folder_index in range(start_folder_index, len(folders)):
        folder = folders[folder_index]
        folder_name = Path(folder.rstrip("\\/")).name or folder

        step_start = start_step_index if folder_index == start_folder_index else 0

        for step_index in range(step_start, len(steps)):
            step_key, script_name, step_label = steps[step_index]

            if should_stop(status_file):
                resume_state.update({"folder_index": folder_index, "step_index": step_index, "status": "stopped"})
                finalize_status(
                    status_file, read_status(status_file), "stopped",
                    completed_units, total_units,
                    f"Stopped before {folder_name} — {step_label}",
                    resume_state,
                )
                send_discord("stopped", folder_label, status["current"], log_file)
                sys.exit(0)

            status = read_status(status_file)
            update_pipeline_status(
                status_file, status, completed_units, total_units,
                folder_name, step_label, {"percent": 0, "current": "Starting step..."},
                step_key=step_key,
            )

            print(f"\n>>> [{folder_name}] {step_label}", flush=True)
            result = run_step(
                args,
                folder,
                step_key,
                script_name,
                step_label,
                status_file,
                log_file,
                completed_units,
                total_units,
                folder_name,
            )

            step_status_file = status_file.parent / "steps" / f"{status_file.stem}_{step_key}.json"
            child_status = read_status(step_status_file)
            status = read_status(status_file)
            update_pipeline_status(
                status_file, status, completed_units, total_units,
                folder_name, step_label, child_status,
                step_key=step_key,
            )

            if result == "stopped":
                resume_state.update({"folder_index": folder_index, "step_index": step_index, "status": "stopped"})
                finalize_status(
                    status_file, read_status(status_file), "stopped",
                    completed_units, total_units,
                    f"Stopped during {folder_name} — {step_label}",
                    resume_state,
                )
                send_discord("stopped", folder_label, status["current"], log_file)
                sys.exit(0)

            if result == "failed":
                resume_state.update({"folder_index": folder_index, "step_index": step_index, "status": "failed"})
                finalize_status(
                    status_file, read_status(status_file), "failed",
                    completed_units, total_units,
                    f"Failed during {folder_name} — {step_label}",
                    resume_state,
                )
                send_discord("failed", folder_label, status["current"], log_file)
                sys.exit(1)

            completed_units += 1
            if step_index + 1 >= len(steps):
                resume_state.update({
                    "folder_index": folder_index + 1,
                    "step_index": 0,
                    "status": "running",
                })
            else:
                resume_state.update({
                    "folder_index": folder_index,
                    "step_index": step_index + 1,
                    "status": "running",
                })
            save_resume_state(resume_state)

        start_step_index = 0

    failure_count = count_pipeline_failures()
    if failure_count:
        message = (
            f"Pipeline finished with {failure_count} per-video failure(s) "
            f"({len(folders)} folder(s), {len(steps)} step(s) each). "
            "See Tools/System or pipeline_failures.json for details."
        )
        event = "complete_with_failures"
    else:
        message = f"Pipeline complete ({len(folders)} folder(s), {len(steps)} step(s) each)."
        event = "complete"

    finalize_status(
        status_file,
        read_status(status_file),
        "complete",
        completed_units,
        total_units,
        message,
        resume_state,
    )
    clear_resume_state()
    send_discord(event, folder_label, message, log_file)
    print(f"\n{message}", flush=True)


if __name__ == "__main__":
    main()
