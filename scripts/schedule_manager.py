"""Overnight / scheduled pipeline runs with stop-time and resume support."""
import datetime
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import psutil

SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = SCRIPT_DIR.parent
CONFIG_FILE = APP_DIR / "config.json"
DATA_DIR = APP_DIR / "data"
LOG_DIR = APP_DIR / "logs"
JOBS_DIR = APP_DIR / "jobs"
DB_FILE = APP_DIR / "data" / "video_indexer.db"
RESUME_FILE = DATA_DIR / "pipeline_resume.json"
PIPELINE_FOLDERS_FILE = DATA_DIR / "pipeline_folders.json"
SCHEDULE_STATE_FILE = DATA_DIR / "schedule_state.json"
SCHEDULE_LOG_FILE = LOG_DIR / "schedule.log"
PYTHON = APP_DIR / "venv" / "Scripts" / "python.exe"

PIPELINE_STEP_SUFFIXES = (
    "_scan",
    "_normalize",
    "_transcribe",
    "_vision",
    "_metadata",
    "_index",
)

WHISPER_MODELS = (
    "tiny", "base", "small", "medium", "large-v2", "large-v3",
)

WEEKDAY_LABELS = (
    ("mon", "Monday"),
    ("tue", "Tuesday"),
    ("wed", "Wednesday"),
    ("thu", "Thursday"),
    ("fri", "Friday"),
    ("sat", "Saturday"),
    ("sun", "Sunday"),
)

WEEKDAY_KEYS = [key for key, _ in WEEKDAY_LABELS]

DEFAULT_SCHEDULE = {
    "enabled": False,
    "days_of_week": ["mon", "tue", "wed", "thu", "fri"],
    "start_time": "22:00",
    "stop_time": "06:00",
    "scope": "all_folders",
    "auto_resume": True,
}

WINDOWS_TASK_NAME = "AI Video Indexer Overnight"


def merge_schedule_defaults(config):
    schedule = config.setdefault("schedule", {})
    for key, value in DEFAULT_SCHEDULE.items():
        if key not in schedule:
            schedule[key] = value.copy() if isinstance(value, list) else value
    days = schedule.get("days_of_week") or []
    schedule["days_of_week"] = [d for d in days if d in WEEKDAY_KEYS]
    return schedule


def load_config():
    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    else:
        config = {}
    merge_schedule_defaults(config)
    return config


def parse_hhmm(value):
    text = str(value or "00:00").strip()
    parts = text.split(":")
    try:
        hour = int(parts[0]) if parts else 0
        minute = int(parts[1]) if len(parts) > 1 else 0
    except (TypeError, ValueError):
        return 22, 0
    return hour % 24, minute % 60


def minutes_since_midnight(dt):
    return (dt.hour * 60) + dt.minute


def is_overnight_window(start_time, stop_time):
    start_h, start_m = parse_hhmm(start_time)
    stop_h, stop_m = parse_hhmm(stop_time)
    start_min = (start_h * 60) + start_m
    stop_min = (stop_h * 60) + stop_m
    return stop_min <= start_min


def is_in_run_window(now, start_time, stop_time):
    now_min = minutes_since_midnight(now)
    start_h, start_m = parse_hhmm(start_time)
    stop_h, stop_m = parse_hhmm(stop_time)
    start_min = (start_h * 60) + start_m
    stop_min = (stop_h * 60) + stop_m
    if is_overnight_window(start_time, stop_time):
        return now_min >= start_min or now_min < stop_min
    return start_min <= now_min < stop_min


def weekday_key(dt):
    return WEEKDAY_KEYS[dt.weekday()]


def is_active_schedule_day(now, days_of_week, start_time, stop_time):
    days = set(days_of_week or [])
    if not days:
        return False
    if is_overnight_window(start_time, stop_time):
        now_min = minutes_since_midnight(now)
        stop_h, stop_m = parse_hhmm(stop_time)
        stop_min = (stop_h * 60) + stop_m
        if now_min < stop_min:
            yesterday = now - datetime.timedelta(days=1)
            return weekday_key(yesterday) in days or weekday_key(now) in days
    return weekday_key(now) in days


def safe_folder_name(folder):
    name = Path(folder.rstrip("\\/")).name
    if not name:
        name = folder.replace("\\", "_").replace("/", "_").replace(":", "")
    keep = []
    for ch in name:
        keep.append(ch if ch.isalnum() or ch in (" ", "-", "_") else "_")
    return "".join(keep).strip().replace(" ", "_")[:60]


def is_pid_running(pid):
    if not pid:
        return False
    try:
        proc = psutil.Process(int(pid))
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, ValueError, TypeError):
        return False


def read_job(job_file):
    try:
        return json.loads(Path(job_file).read_text(encoding="utf-8"))
    except Exception:
        return None


def sync_resume_to_failed(job):
    if not RESUME_FILE.exists() or not job:
        return
    try:
        saved = json.loads(RESUME_FILE.read_text(encoding="utf-8"))
    except Exception:
        return
    if saved.get("job_id") != job.get("job_id"):
        return
    if saved.get("status") == "running":
        saved["status"] = "failed"
        RESUME_FILE.write_text(json.dumps(saved, indent=2), encoding="utf-8")


def cleanup_stale_running_jobs():
    changed = False
    for job_file in JOBS_DIR.glob("*.json"):
        if any(job_file.stem.endswith(suffix) for suffix in PIPELINE_STEP_SUFFIXES):
            continue
        job = read_job(job_file)
        if not job or job.get("status") != "running":
            continue
        if is_pid_running(job.get("pid")):
            continue
        job["status"] = "failed"
        job["current"] = job.get("current") or "Job interrupted (process no longer running)."
        job["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        job_file.write_text(json.dumps(job, indent=2), encoding="utf-8")
        sync_resume_to_failed(job)
        changed = True
    return changed


def latest_job_files(running_only=False):
    jobs = [
        p for p in JOBS_DIR.glob("*.json")
        if not any(p.stem.endswith(suffix) for suffix in PIPELINE_STEP_SUFFIXES)
    ]
    if running_only:
        jobs = [
            p for p in jobs
            if (read_job(p) or {}).get("status") == "running"
        ]

    def sort_key(path):
        job = read_job(path) or {}
        status_rank = 1 if job.get("status") == "running" else 0
        started = job.get("started_at") or ""
        return (status_rank, started, path.name)

    return sorted(jobs, key=sort_key, reverse=True)


def running_jobs():
    cleanup_stale_running_jobs()
    return latest_job_files(running_only=True)


def running_pipeline_jobs():
    return [
        path for path in running_jobs()
        if (read_job(path) or {}).get("script") == "run_pipeline.py"
    ]


def can_resume_pipeline():
    if not RESUME_FILE.exists():
        return False
    try:
        saved = json.loads(RESUME_FILE.read_text(encoding="utf-8"))
        return saved.get("status") in {"failed", "stopped"}
    except Exception:
        return False


def schedule_target_folders(config):
    schedule = config.get("schedule", {})
    folders = [f for f in config.get("folders", []) if f]
    scope = schedule.get("scope", "all_folders")
    selected = config.get("selected_folder", "")
    if scope == "selected_folder" and selected:
        return [selected]
    return folders or ([selected] if selected else [])


def processing_args(config):
    from job_utils import DEFAULT_VISION_MODEL_KEY, VISION_MODEL_OPTIONS

    p = config.get("processing", {})
    model = p.get("transcription_model", "large-v3")
    if model not in WHISPER_MODELS:
        model = "large-v3"
    vision_model = p.get("vision_model", DEFAULT_VISION_MODEL_KEY)
    if vision_model not in VISION_MODEL_OPTIONS:
        vision_model = DEFAULT_VISION_MODEL_KEY
    return [
        "--use-gpu", "true" if p.get("use_gpu", True) else "false",
        "--overwrite", "true" if p.get("overwrite_existing", False) else "false",
        "--vision-interval", str(int(p.get("vision_frame_interval_seconds", 30))),
        "--min-frames", str(int(p.get("min_frames_per_video", 3))),
        "--transcription-model", model,
        "--vision-model", vision_model,
    ]


def pipeline_args(config, folders, resume=False):
    steps = config.get("pipeline", {})
    processing = config.get("processing", {})
    args = processing_args(config)
    step_flags = [
        ("normalize", "normalize"),
        ("transcribe", "transcribe"),
        ("vision", "vision"),
        ("metadata", "metadata"),
        ("index", "index"),
    ]
    for cli_name, config_key in step_flags:
        enabled = steps.get(config_key, True)
        args.extend([f"--step-{cli_name}", "true" if enabled else "false"])

    PIPELINE_FOLDERS_FILE.write_text(json.dumps(folders), encoding="utf-8")
    args.extend(["--folders-file", str(PIPELINE_FOLDERS_FILE)])
    args.extend(["--resume", "true" if resume else "false"])
    args.extend(["--skip-mode", processing.get("skip_mode", "missing_only")])
    scan_after = processing.get("scan_after_pipeline", True)
    args.extend(["--scan-after", "true" if scan_after else "false"])

    step_overwrite = processing.get("step_overwrite", {})
    if step_overwrite:
        args.extend(["--step-overwrite", json.dumps(step_overwrite)])
    return args


def request_stop_job(job_file):
    job = read_job(job_file)
    if not job:
        return False
    job["stop_requested"] = True
    job_file.write_text(json.dumps(job, indent=2), encoding="utf-8")
    pid = job.get("pid")
    if pid:
        try:
            os.kill(pid, signal.CTRL_BREAK_EVENT)
        except Exception:
            try:
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    capture_output=True,
                    text=True,
                )
            except Exception:
                return False
    return True


def launch_pipeline_job(config, resume=False):
    folders = schedule_target_folders(config)
    if not folders:
        return False, "No folders configured for scheduled runs."

    if running_jobs():
        return False, "A job is already running."

    anchor_folder = folders[0]
    if not resume and RESUME_FILE.exists():
        RESUME_FILE.unlink(missing_ok=True)

    script_path = SCRIPT_DIR / "run_pipeline.py"
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_label = safe_folder_name(anchor_folder)
    job_id = f"run_pipeline_{folder_label}_{timestamp}"
    log_file = LOG_DIR / f"{job_id}.log"
    status_file = JOBS_DIR / f"{job_id}.json"

    cmd = [
        str(PYTHON),
        str(script_path),
        "--folder",
        anchor_folder,
        "--db",
        str(DB_FILE),
        "--status-file",
        str(status_file),
    ]
    cmd.extend(pipeline_args(config, folders, resume=resume))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)

    with open(log_file, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(APP_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

    status = {
        "job_id": job_id,
        "script": "run_pipeline.py",
        "folder": anchor_folder,
        "folder_label": folder_label,
        "pid": proc.pid,
        "status": "running",
        "percent": 0,
        "current": "Scheduled pipeline starting...",
        "processed": 0,
        "total": 0,
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "finished_at": "",
        "log_file": str(log_file),
        "stop_requested": False,
        "scheduled": True,
    }
    status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
    return True, job_id


def load_schedule_state():
    if not SCHEDULE_STATE_FILE.exists():
        return {}
    try:
        return json.loads(SCHEDULE_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_schedule_state(state):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SCHEDULE_STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def append_schedule_log(message):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now().isoformat(timespec="seconds")
    with SCHEDULE_LOG_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"[{stamp}] {message}\n")


def schedule_window_key(now, start_time, stop_time):
    """One run window per overnight span (evening start date)."""
    if is_overnight_window(start_time, stop_time):
        stop_h, stop_m = parse_hhmm(stop_time)
        stop_min = (stop_h * 60) + stop_m
        if minutes_since_midnight(now) < stop_min:
            anchor = now - datetime.timedelta(days=1)
        else:
            anchor = now
    else:
        anchor = now
    return anchor.date().isoformat()


def should_start_pipeline(config, now=None):
    now = now or datetime.datetime.now()
    schedule = config.get("schedule", {})
    if not schedule.get("enabled"):
        return False, "disabled"
    if not is_in_run_window(now, schedule["start_time"], schedule["stop_time"]):
        return False, "outside_window"
    if not is_active_schedule_day(
        now,
        schedule.get("days_of_week", []),
        schedule["start_time"],
        schedule["stop_time"],
    ):
        return False, "inactive_day"
    if running_jobs():
        return False, "job_running"

    state = load_schedule_state()
    window_key = schedule_window_key(now, schedule["start_time"], schedule["stop_time"])
    if state.get("started_window") == window_key:
        return False, "already_started_window"

    folders = schedule_target_folders(config)
    if not folders:
        return False, "no_folders"
    return True, "ok"


def running_scheduled_pipeline_jobs():
    return [
        path for path in running_pipeline_jobs()
        if (read_job(path) or {}).get("scheduled")
    ]


def should_stop_pipeline(config, now=None):
    now = now or datetime.datetime.now()
    schedule = config.get("schedule", {})
    if not schedule.get("enabled"):
        return False, "disabled"
    pipeline_jobs = running_scheduled_pipeline_jobs()
    if not pipeline_jobs:
        return False, "no_scheduled_pipeline"
    if is_in_run_window(now, schedule["start_time"], schedule["stop_time"]):
        if is_active_schedule_day(
            now,
            schedule.get("days_of_week", []),
            schedule["start_time"],
            schedule["stop_time"],
        ):
            return False, "still_in_window"
    return True, "stop_time"


def tick_schedule(config=None, *, force=False):
    """Check schedule rules and start/stop pipeline jobs. Returns action summary."""
    config = config or load_config()
    merge_schedule_defaults(config)
    now = datetime.datetime.now()
    result = {
        "checked_at": now.isoformat(timespec="seconds"),
        "action": "none",
        "detail": "",
    }

    stop_ok, stop_reason = should_stop_pipeline(config, now)
    if stop_ok:
        stopped = []
        for job_file in running_scheduled_pipeline_jobs():
            if request_stop_job(job_file):
                stopped.append(job_file.stem)
        if stopped:
            state = load_schedule_state()
            state["last_stop_at"] = now.isoformat(timespec="seconds")
            state["last_action"] = "stopped"
            save_schedule_state(state)
            msg = f"Stopped scheduled pipeline job(s): {', '.join(stopped)} ({stop_reason})"
            append_schedule_log(msg)
            result.update({"action": "stopped", "detail": msg})
            return result

    start_ok, start_reason = should_start_pipeline(config, now)
    if not start_ok and not force:
        result["detail"] = start_reason
        return result

    if force and running_jobs():
        result["detail"] = "job_running"
        return result

    if not start_ok and force:
        folders = schedule_target_folders(config)
        if not folders:
            result["detail"] = "no_folders"
            return result
        start_reason = "forced"

    resume = bool(config.get("schedule", {}).get("auto_resume", True) and can_resume_pipeline())
    launched, detail = launch_pipeline_job(config, resume=resume)
    if not launched:
        result["detail"] = detail
        append_schedule_log(f"Start skipped: {detail}")
        return result

    schedule = config.get("schedule", {})
    window_key = schedule_window_key(now, schedule["start_time"], schedule["stop_time"])
    state = load_schedule_state()
    state.update({
        "started_window": window_key,
        "last_start_at": now.isoformat(timespec="seconds"),
        "last_action": "started",
        "last_job_id": detail,
        "last_resume": resume,
    })
    save_schedule_state(state)
    mode = "resume" if resume else "fresh"
    msg = f"Started scheduled pipeline ({mode}): {detail}"
    append_schedule_log(msg)
    result.update({"action": "started", "detail": msg, "resume": resume})
    return result


def schedule_needs_background_poll(config):
    return bool(config.get("schedule", {}).get("enabled"))


def schedule_status(config=None):
    config = config or load_config()
    merge_schedule_defaults(config)
    now = datetime.datetime.now()
    schedule = config.get("schedule", {})
    state = load_schedule_state()
    in_window = is_in_run_window(now, schedule["start_time"], schedule["stop_time"])
    active_day = is_active_schedule_day(
        now,
        schedule.get("days_of_week", []),
        schedule["start_time"],
        schedule["stop_time"],
    )
    pipeline_jobs = running_pipeline_jobs()
    return {
        "enabled": schedule.get("enabled", False),
        "in_window": in_window,
        "active_day": active_day,
        "can_resume": can_resume_pipeline(),
        "running_pipeline": bool(pipeline_jobs),
        "running_scheduled_pipeline": bool(running_scheduled_pipeline_jobs()),
        "running_job_ids": [p.stem for p in pipeline_jobs],
        "target_folders": schedule_target_folders(config),
        "state": state,
        "would_start": should_start_pipeline(config, now)[0],
        "would_stop": should_stop_pipeline(config, now)[0],
    }


def windows_task_command():
    runner = SCRIPT_DIR / "scheduled_runner.py"
    return (
        f'cmd /c cd /d "{APP_DIR}" && '
        f'"{PYTHON}" "{runner}"'
    )


def register_windows_scheduled_task():
    if not PYTHON.exists():
        return False, f"Python not found: {PYTHON}"
    runner = SCRIPT_DIR / "scheduled_runner.py"
    if not runner.exists():
        return False, f"Missing runner script: {runner}"

    tr = windows_task_command()
    cmd = [
        "schtasks",
        "/Create",
        "/TN",
        WINDOWS_TASK_NAME,
        "/TR",
        tr,
        "/SC",
        "MINUTE",
        "/MO",
        "5",
        "/F",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as exc:
        return False, str(exc)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, err or "schtasks failed"
    return True, (
        f"Registered Windows task **{WINDOWS_TASK_NAME}** (runs every 5 minutes). "
        "Enable the schedule in Tools/System and keep your PC awake overnight."
    )


def remove_windows_scheduled_task():
    cmd = ["schtasks", "/Delete", "/TN", WINDOWS_TASK_NAME, "/F"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
    except Exception as exc:
        return False, str(exc)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return False, err or "schtasks delete failed"
    return True, f"Removed Windows task **{WINDOWS_TASK_NAME}**."
