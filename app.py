import streamlit as st
from pathlib import Path
import html
import json
import subprocess
import datetime
import sqlite3
import os
import signal
import sys
import pandas as pd
import psutil
from search_engine import (
    search_videos,
    get_search_index_stats,
    reset_search_index,
    release_qdrant_client,
    delete_qdrant_points_for_folder,
    prune_orphan_qdrant_points,
    format_search_log,
    SEARCH_LOG_FILE,
)
from app_db import (
    init_db,
    get_videos,
    get_library_stats,
    prune_missing_videos,
    remove_folder_from_library,
    find_duplicate_groups,
)
from app_help import render_help_tab, whisper_model_info_popover, vision_model_info_popover
from app_helpers import (
    videos_status_dataframe,
    open_video_at_timestamp,
    browse_for_folder,
    search_result_preview_text,
    format_search_result_text,
    format_search_score_badge,
    format_search_result_summary_html,
    inject_search_results_styles,
    search_result_meta_line,
    format_modified_time,
    find_vision_resume_mismatches,
    find_resume_step_mismatch,
    format_step_key_list,
    get_search_result_thumbnail,
    search_results_to_csv,
    search_results_to_markdown,
)
from app_jobs import render_job_panel, render_job_status_banner, tail_log_file
from app_wizard import render_setup_wizard, environment_ready
from notifications import send_discord_webhook
from pipeline_estimate import render_pipeline_estimator_ui

SCRIPT_DIR_IMPORT = Path(__file__).parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR_IMPORT))
from pipeline_utils import filter_videos_by_status, video_has_incomplete_steps, STEP_LABELS
from job_utils import DEFAULT_VISION_MODEL_KEY, VISION_MODEL_OPTIONS
from model_deps import missing_vision_packages

APP_DIR = Path(__file__).parent
CONFIG_FILE = APP_DIR / "config.json"
LOG_DIR = APP_DIR / "logs"
SCRIPT_DIR = APP_DIR / "scripts"
DATA_DIR = APP_DIR / "data"
JOBS_DIR = APP_DIR / "jobs"
DB_FILE = DATA_DIR / "video_indexer.db"
RESUME_FILE = DATA_DIR / "pipeline_resume.json"
PIPELINE_FOLDERS_FILE = DATA_DIR / "pipeline_folders.json"
PIPELINE_VIDEO_FILTER = DATA_DIR / "pipeline_video_filter.json"
SKIP_MODE_OPTIONS = {
    "all": "All videos (respect overwrite settings)",
    "missing_only": "Missing outputs only (recommended)",
    "stale_only": "Stale or outdated sidecars only",
    "incomplete_only": "Anything not fully complete",
}
WHISPER_MODELS = (
    "tiny",
    "base",
    "small",
    "medium",
    "large-v2",
    "large-v3",
)
VISION_MODEL_KEYS = tuple(VISION_MODEL_OPTIONS.keys())
SEARCH_EXAMPLE_QUERIES = (
    "birthday party",
    "kids swimming",
    "Christmas morning",
    "at the beach",
    "grandpa talking",
)
for p in [LOG_DIR, SCRIPT_DIR, DATA_DIR, JOBS_DIR]:
    p.mkdir(exist_ok=True)

DEFAULT_CONFIG = {
    "folders": [],
    "selected_folder": "",
    "processing": {
        "transcription_model": "large-v3",
        "vision_model": DEFAULT_VISION_MODEL_KEY,
        "use_gpu": True,
        "vision_frame_interval_seconds": 30,
        "min_frames_per_video": 3,
        "overwrite_existing": False,
        "skip_mode": "missing_only",
        "scan_after_pipeline": True,
        "step_overwrite": {
            "normalize": False,
            "transcribe": False,
            "vision": False,
            "metadata": False,
            "index": False,
        },
    },
    "pipeline": {
        "normalize": True,
        "transcribe": True,
        "vision": True,
        "metadata": True,
        "index": True
    },
    "notifications": {
        "discord_webhook_url": "",
        "notify_on_complete": True,
        "notify_on_failed": True,
        "notify_on_stopped": False
    }
}


def load_config():
    if CONFIG_FILE.exists():
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    else:
        config = DEFAULT_CONFIG.copy()

    if "pipeline" in config:
        config["pipeline"].pop("enhance_audio", None)

    processing = config.setdefault("processing", {})
    processing.pop("scan_before_pipeline", None)
    processing.pop("normalize_dry_run", None)
    processing.setdefault("scan_after_pipeline", True)
    processing.setdefault("skip_mode", "missing_only")
    processing.setdefault("transcription_model", "large-v3")
    processing.setdefault("vision_model", DEFAULT_VISION_MODEL_KEY)
    processing.setdefault("step_overwrite", DEFAULT_CONFIG["processing"]["step_overwrite"].copy())

    return config


def save_config(config):
    CONFIG_FILE.write_text(json.dumps(config, indent=2), encoding="utf-8")


def is_pid_running(pid):
    if not pid:
        return False
    try:
        proc = psutil.Process(int(pid))
        return proc.is_running() and proc.status() != psutil.STATUS_ZOMBIE
    except (psutil.Error, ValueError, TypeError):
        return False


def sync_resume_to_failed(job):
    """Keep resume coordinates after a crash but mark the run as resumable."""
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
    """Mark orphaned job files as failed when their process is no longer running."""
    changed = False
    for job_file in JOBS_DIR.glob("*.json"):
        job = read_job(job_file)
        if not job or job.get("status") != "running":
            continue
        if is_pid_running(job.get("pid")):
            continue
        job["status"] = "failed"
        job["current"] = "Job interrupted (process no longer running)."
        job["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        job_file.write_text(json.dumps(job, indent=2), encoding="utf-8")
        sync_resume_to_failed(job)
        changed = True
    return changed


def running_job_summary():
    cleanup_stale_running_jobs()
    jobs = active_job_files()
    if not jobs:
        return None
    job = read_job(jobs[0]) or {}
    return job.get("script") or job.get("job_id") or "job"


def block_if_job_running(action_label="start another job"):
    summary = running_job_summary()
    if summary:
        st.error(
            f"A job is already running ({summary}). "
            f"Stop it on the **Dashboard** before you {action_label}."
        )
        return True
    return False


PIPELINE_STEP_SUFFIXES = (
    "_scan",
    "_normalize",
    "_transcribe",
    "_vision",
    "_metadata",
    "_index",
)


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


def active_job_files():
    """Jobs currently running — used for live dashboard/banner UI."""
    return latest_job_files(running_only=True)


def cleanup_stale_resume_state():
    if not RESUME_FILE.exists():
        return

    try:
        saved = json.loads(RESUME_FILE.read_text(encoding="utf-8"))
    except Exception:
        RESUME_FILE.unlink(missing_ok=True)
        return

    status = saved.get("status")
    if status == "complete":
        RESUME_FILE.unlink(missing_ok=True)
        return

    job_id = saved.get("job_id", "")
    job_file = JOBS_DIR / f"{job_id}.json"
    job = read_job(job_file) if job_file.exists() else None

    if status == "running":
        if job and job.get("status") == "running":
            return
        saved["status"] = "failed"
        RESUME_FILE.write_text(json.dumps(saved, indent=2), encoding="utf-8")


def can_resume_pipeline():
    cleanup_stale_resume_state()
    if not RESUME_FILE.exists():
        return False
    try:
        saved = json.loads(RESUME_FILE.read_text(encoding="utf-8"))
        return saved.get("status") in {"failed", "stopped"}
    except Exception:
        return False


def read_job(job_file):
    try:
        return json.loads(job_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_folder_name(folder):
    name = Path(folder.rstrip("\\/")).name
    if not name:
        name = folder.replace("\\", "_").replace("/", "_").replace(":", "")
    keep = []
    for ch in name:
        keep.append(ch if ch.isalnum() or ch in (" ", "-", "_") else "_")
    return "".join(keep).strip().replace(" ", "_")[:60]


def pipeline_args(config, folders, resume=False, video_filter_file=None):
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

    if video_filter_file:
        args.extend(["--videos-file", str(video_filter_file)])

    return args


def write_pipeline_video_filter(paths):
    PIPELINE_VIDEO_FILTER.write_text(
        json.dumps([str(p) for p in paths]),
        encoding="utf-8",
    )
    return PIPELINE_VIDEO_FILTER


def clear_pipeline_video_filter():
    if PIPELINE_VIDEO_FILTER.exists():
        PIPELINE_VIDEO_FILTER.unlink()


def run_pipeline_job(
    config,
    target_folders,
    anchor_folder,
    folder_label=None,
    resume=False,
    video_filter_file=None,
):
    if not resume:
        if RESUME_FILE.exists():
            RESUME_FILE.unlink()
    run_script(
        "run_pipeline.py",
        anchor_folder,
        pipeline_args(
            config,
            target_folders,
            resume=resume,
            video_filter_file=video_filter_file,
        ),
        folder_label=folder_label,
    )


def processing_args(config):
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


def run_model_deps_install(stack):
    folder = config.get("selected_folder") or str(APP_DIR)
    return run_script(
        "install_model_deps.py",
        folder,
        extra_args=["--stack", stack],
    )


def start_pipeline_on_videos(config, folder, video_paths, *, resume=False):
    """Run the configured pipeline on an explicit list of video paths."""
    if not video_paths:
        return False
    filter_file = write_pipeline_video_filter([str(p) for p in video_paths])
    run_pipeline_job(
        config,
        [folder],
        folder,
        None,
        resume=resume,
        video_filter_file=filter_file,
    )
    return True


def run_script(script_name, folder, extra_args=None, folder_label=None):
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        st.error(f"Missing script: {script_path}")
        return False

    if block_if_job_running("start a new job"):
        return False

    if script_name in ("run_pipeline.py", "index_qdrant.py"):
        release_qdrant_client()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_label = folder_label or safe_folder_name(folder)
    job_id = f"{script_path.stem}_{folder_label}_{timestamp}"

    log_file = LOG_DIR / f"{job_id}.log"
    status_file = JOBS_DIR / f"{job_id}.json"

    cmd = [
        str(APP_DIR / "venv" / "Scripts" / "python.exe"),
        str(script_path),
        "--folder",
        folder,
        "--db",
        str(DB_FILE),
        "--status-file",
        str(status_file)
    ]

    if extra_args:
        cmd.extend(extra_args)

    with open(log_file, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(APP_DIR),
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )

    status = {
        "job_id": job_id,
        "script": script_name,
        "folder": folder,
        "folder_label": folder_label,
        "pid": proc.pid,
        "status": "running",
        "percent": 0,
        "current": "",
        "processed": 0,
        "total": 0,
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "finished_at": "",
        "log_file": str(log_file),
        "stop_requested": False,
    }

    status_file.write_text(json.dumps(status, indent=2), encoding="utf-8")
    st.success(f"Started: {script_name}")
    return True


def stop_job(job_file):
    job = read_job(job_file)
    if not job:
        return

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
                    text=True
                )
            except Exception as e:
                st.error(f"Could not stop process: {e}")


@st.fragment(run_every=datetime.timedelta(seconds=1))
def render_dashboard_live():
    job_files = active_job_files()
    render_job_panel(
        job_files,
        read_job,
        stop_job,
        jobs_dir=JOBS_DIR,
        key_prefix="dash_",
        empty_message="No job is running right now.",
    )

    st.divider()

    st.subheader("System Live Status")

    cpu = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()

    col1, col2, col3 = st.columns(3)

    col1.metric("CPU Usage", f"{cpu:.0f}%")
    col2.metric("RAM Used", f"{ram.used / (1024**3):.1f} GB")
    col3.metric("RAM Available", f"{ram.available / (1024**3):.1f} GB")

    try:
        gpu_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )

        if gpu_result.returncode == 0 and gpu_result.stdout.strip():
            gpu_parts = [p.strip() for p in gpu_result.stdout.strip().split(",")]

            if len(gpu_parts) >= 5:
                gpu_name, gpu_util, gpu_mem_used, gpu_mem_total, gpu_temp = gpu_parts[:5]

                gpu_left, gpu_right = st.columns(2)
                with gpu_left:
                    st.markdown(f"**GPU:** {gpu_name}")
                    st.markdown(f"**GPU usage:** {gpu_util}%")
                with gpu_right:
                    st.markdown(f"**GPU memory:** {gpu_mem_used} / {gpu_mem_total} MB")
                    st.markdown(f"**GPU temp:** {gpu_temp}°C")
        else:
            st.info("No NVIDIA GPU status available.")
    except Exception as e:
        st.info(f"GPU status unavailable: {e}")

    st.divider()

    st.subheader("Library Summary")

    stats = get_library_stats(str(DB_FILE))

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Videos", stats["total"])
    col2.metric("Transcribed", stats["transcribed"])
    col3.metric("Vision", stats["vision"])
    col4.metric("Indexed", stats["indexed"])


def render_logs_live():
    """Log viewer — no @st.fragment here; fragments + selectbox inside tabs break tab layout."""
    logs = sorted(LOG_DIR.glob("*.log"), reverse=True)

    if not logs:
        st.info("No logs yet.")
        return

    ctrl_col1, ctrl_col2 = st.columns([3, 1])
    with ctrl_col1:
        selected_log = st.selectbox(
            "Select log",
            logs,
            format_func=lambda p: p.name,
            key="logs_tab_select",
        )
    with ctrl_col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False, key="logs_auto_refresh")

    if auto_refresh:
        try:
            from streamlit_autorefresh import st_autorefresh

            st_autorefresh(interval=2000, limit=None, key="logs_autorefresh_tick")
        except ImportError:
            st.caption("Install streamlit-autorefresh for auto-refresh.")

    if selected_log.exists():
        try:
            content = tail_log_file(selected_log, max_lines=400)
        except Exception as e:
            st.error(f"Could not read log: {e}")
            return

        st.caption(f"Showing last 400 lines of **{selected_log.name}**")
        st.code(content, language=None)


def _render_job_banner(key_prefix):
    render_job_status_banner(
        active_job_files(), read_job, stop_job, key_prefix=key_prefix
    )


def make_job_banner_fragment(key_prefix):
    @st.fragment(run_every=datetime.timedelta(seconds=1))
    def banner():
        _render_job_banner(key_prefix)
    return banner


job_banner_library = make_job_banner_fragment("bn_lib_")
job_banner_jobs = make_job_banner_fragment("bn_jobs_")
job_banner_search = make_job_banner_fragment("bn_search_")
job_banner_tools = make_job_banner_fragment("bn_tools_")
job_banner_logs = make_job_banner_fragment("bn_logs_")
job_banner_help = make_job_banner_fragment("bn_help_")

INSTALL_JOB_SCRIPTS = {"install_dependencies.py", "install_model_deps.py"}


def dependency_job_files(running_only=False):
    jobs = [
        p for p in latest_job_files(running_only=running_only)
        if (read_job(p) or {}).get("script") in INSTALL_JOB_SCRIPTS
        or p.name.startswith("install_dependencies_")
        or p.name.startswith("install_model_deps_")
    ]
    return jobs


@st.fragment(run_every=datetime.timedelta(seconds=1))
def render_dependency_install_status():
    """Live install progress for Tools/System (mirrors Dashboard job panel)."""
    running_installs = dependency_job_files(running_only=True)
    if running_installs:
        render_job_panel(
            running_installs,
            read_job,
            stop_job,
            jobs_dir=JOBS_DIR,
            key_prefix="tools_dep_",
            empty_message="",
        )
        return

    latest = dependency_job_files(running_only=False)
    if not latest:
        st.info("No dependency install has run yet.")
        return

    dep_job_file = latest[0]
    dep_job = read_job(dep_job_file)
    if not dep_job:
        st.info("No dependency install has run yet.")
        return

    if dep_job.get("status") == "running":
        render_job_panel(
            [dep_job_file],
            read_job,
            stop_job,
            jobs_dir=JOBS_DIR,
            key_prefix="tools_dep_",
            empty_message="",
        )
        return

    st.caption("Latest dependency install")
    render_job_panel(
        [dep_job_file],
        read_job,
        stop_job,
        jobs_dir=JOBS_DIR,
        key_prefix="tools_dep_last_",
        empty_message="",
    )


init_db(str(DB_FILE))
cleanup_stale_resume_state()
config = load_config()
env_ready = environment_ready(APP_DIR)

st.set_page_config(page_title="AI Video Indexer", layout="wide")
st.title("AI Video Indexer")

tab_dashboard, tab_library, tab_jobs, tab_search, tab_tools, tab_logs, tab_help = st.tabs(
    ["Dashboard", "Library", "Run Jobs", "Search", "Tools/System", "Logs", "Help"]
)

with tab_dashboard:
    st.header("Dashboard")
    config = render_setup_wizard(config, APP_DIR, save_config)
    render_dashboard_live()
    
with tab_library:
    job_banner_library()
    st.header("Video Library")

    if "library_folder_input" not in st.session_state:
        st.session_state.library_folder_input = ""

    folder_col1, folder_col2 = st.columns([4, 1])
    with folder_col2:
        st.write("")
        st.write("")
        browse_clicked = st.button(
            "Browse...", key="browse_folder_btn", use_container_width=True
        )

    if browse_clicked:
        selected_path, browse_error = browse_for_folder()
        if browse_error:
            st.warning(browse_error)
        elif selected_path:
            st.session_state.library_folder_input = selected_path
            st.rerun()

    with folder_col1:
        with st.form("library_add_folder_form", clear_on_submit=False):
            new_folder = st.text_input(
                "Add video folder path",
                value=st.session_state.get("library_folder_input", ""),
                placeholder=r"D:\Videos\Home Movies  or  \\SERVER\Share\Videos",
            )
            add_folder = st.form_submit_button("Add Folder")

    if add_folder:
        folder = new_folder.strip()
        if folder:
            if folder not in config["folders"]:
                config["folders"].append(folder)
            config["selected_folder"] = folder
            st.session_state.library_folder_input = ""
            save_config(config)
            st.success(f"Added folder: {folder}")
            st.rerun()

    if config["folders"]:
        selected = st.selectbox(
            "Selected folder",
            config["folders"],
            index=config["folders"].index(config["selected_folder"])
            if config["selected_folder"] in config["folders"]
            else 0
        )
        config["selected_folder"] = selected
        save_config(config)

        st.code(config["selected_folder"])

        remove_confirm = st.checkbox(
            "I understand this will remove this folder from the app library database",
            key="remove_folder_confirm"
        )

        if st.button("Remove Selected Folder from Library"):
            if not remove_confirm:
                st.warning("Check the confirmation box first.")
            else:
                folder_to_remove = config["selected_folder"]

                remove_folder_from_library(
                    str(DB_FILE),
                    folder_to_remove,
                    qdrant_cleanup_fn=delete_qdrant_points_for_folder,
                )

                config["folders"] = [
                    f for f in config["folders"]
                    if f != folder_to_remove
                ]

                config["selected_folder"] = config["folders"][0] if config["folders"] else ""

                save_config(config)

                st.success(f"Removed folder from library: {folder_to_remove}")
                st.rerun()

        videos = get_videos(folder=selected)

        st.subheader("Scanned Files")
        st.write(f"Videos in database for this folder: **{len(videos)}**")

        if videos:
            incomplete_count = sum(1 for v in videos if video_has_incomplete_steps(v))
            st.caption(
                f"Status legend: ✓ complete · missing · ↻ stale sidecar · ✗ failed | "
                f"**{incomplete_count}** incomplete"
            )

            status_filter = st.selectbox(
                "Show",
                ["All", "Incomplete", "Failed", "Stale", "Missing Transcript", "Missing Index"],
                key="library_status_filter",
            )
            filtered_videos = filter_videos_by_status(videos, status_filter)
            status_df = videos_status_dataframe(filtered_videos)
            display_df = status_df.drop(columns=["Path"])
            if display_df["Thumbnail"].isna().all():
                display_df = display_df.drop(columns=["Thumbnail"])

            st.dataframe(
                display_df,
                width="stretch",
                hide_index=True,
                column_config={
                    "Thumbnail": st.column_config.ImageColumn("Preview", width="small"),
                },
            )

            duplicate_groups = find_duplicate_groups(str(DB_FILE), folder=selected)
            if duplicate_groups:
                with st.expander(
                    f"Possible duplicates ({len(duplicate_groups)} group(s))",
                    expanded=False,
                ):
                    st.caption(
                        "Videos matched by content fingerprint or identical size + duration. "
                        "Rescan folders to refresh fingerprints."
                    )
                    for group in duplicate_groups:
                        st.markdown(
                            f"**{group['match_type']}** — "
                            + ", ".join(v["filename"] for v in group["videos"])
                        )
                        for video in group["videos"]:
                            st.code(video["path"])

            file_options = {v["filename"]: v["path"] for v in filtered_videos}
            if file_options:
                st.subheader("Process selected file(s)")
                selected_names = st.multiselect(
                    "Choose one or more videos",
                    options=sorted(file_options.keys()),
                    help="Run the pipeline steps selected on Run Jobs for only these files.",
                )
                single_col1, single_col2 = st.columns([1, 3])
                with single_col1:
                    if st.button(
                        "Run pipeline on selected",
                        key="library_run_selected_files",
                        disabled=not selected_names or not env_ready,
                    ):
                        paths = [file_options[name] for name in selected_names]
                        start_pipeline_on_videos(config, selected, paths)
                        st.success(
                            f"Started pipeline for {len(paths)} file(s). "
                            "Track progress on the Dashboard."
                        )

            retry_col1, retry_col2 = st.columns([1, 3])
            with retry_col1:
                if st.button("Retry incomplete in folder", key="retry_incomplete_library"):
                    incomplete_paths = [
                        v["path"] for v in videos if video_has_incomplete_steps(v)
                    ]
                    if not incomplete_paths:
                        st.info("No incomplete videos in this folder.")
                    else:
                        write_pipeline_video_filter(incomplete_paths)
                        saved_mode = config["processing"].get("skip_mode")
                        config["processing"]["skip_mode"] = "incomplete_only"
                        save_config(config)
                        run_pipeline_job(
                            config,
                            [selected],
                            selected,
                            None,
                            video_filter_file=PIPELINE_VIDEO_FILTER,
                        )
                        config["processing"]["skip_mode"] = saved_mode
                        save_config(config)
                        st.success(
                            f"Started pipeline for {len(incomplete_paths)} incomplete video(s). "
                            "Track progress on the Dashboard."
                        )
        else:
            st.info("No scanned files yet. Run Scan Library from the Run Jobs tab.")
    else:
        st.warning("No folders added yet.")

with tab_jobs:
    job_banner_jobs()
    st.header("Run Processing Jobs")

    if not env_ready:
        st.warning(
            "Processing jobs are disabled until setup completes. "
            "Run **setup.bat**, then use **Tools/System → Refresh System Check** to verify FFmpeg and the virtual environment."
        )

    if not config["folders"]:
        st.warning("Add a folder first on the Library tab.")
    else:
        selected_folder = st.selectbox(
            "Folder to process",
            config["folders"],
            index=config["folders"].index(config["selected_folder"])
            if config["selected_folder"] in config["folders"]
            else 0,
            key="job_folder"
        )

        st.subheader("Processing Settings")

        with st.container(border=True):
            st.markdown("**General**")
            gen_col1, gen_col2 = st.columns(2)
            with gen_col1:
                config["processing"]["use_gpu"] = st.checkbox(
                    "Use GPU where possible",
                    value=config["processing"].get("use_gpu", True),
                    help="Uses your GPU for Whisper transcription and local vision models when available.",
                )
            with gen_col2:
                config["processing"]["overwrite_existing"] = st.checkbox(
                    "Overwrite existing output files",
                    value=config["processing"].get("overwrite_existing", False),
                    help="When off, jobs skip videos that already have matching outputs. Safer for resuming long runs.",
                )

            st.caption("Skip mode controls which videos each pipeline step processes.")
            config["processing"]["scan_after_pipeline"] = st.checkbox(
                "Rescan library after pipeline completes",
                value=config["processing"].get("scan_after_pipeline", True),
                help="Updates catalog flags and Library tab counts when the pipeline finishes. Turn off to save time on very large libraries.",
            )
            config["processing"]["skip_mode"] = st.selectbox(
                "Pipeline skip mode",
                list(SKIP_MODE_OPTIONS.keys()),
                format_func=lambda key: SKIP_MODE_OPTIONS[key],
                index=list(SKIP_MODE_OPTIONS.keys()).index(
                    config["processing"].get("skip_mode", "missing_only")
                ) if config["processing"].get("skip_mode", "missing_only") in SKIP_MODE_OPTIONS else 1,
                help="Missing-only is safest for re-runs. Use stale-only to refresh metadata/index after transcript or vision changes.",
            )

        settings_col1, settings_col2 = st.columns(2)

        with settings_col1:
            with st.container(border=True):
                whisper_head_col, whisper_info_col = st.columns([4, 1])
                with whisper_head_col:
                    st.markdown("**Transcription (Whisper)**")
                with whisper_info_col:
                    whisper_model_info_popover()
                current_model = config["processing"].get("transcription_model", "large-v3")
                if current_model not in WHISPER_MODELS:
                    current_model = "large-v3"
                config["processing"]["transcription_model"] = st.selectbox(
                    "Whisper model",
                    WHISPER_MODELS,
                    index=WHISPER_MODELS.index(current_model),
                    help="Larger models are more accurate but slower and use more VRAM. large-v3 is recommended on a GPU.",
                )
                st.caption(
                    "Speech language is auto-detected per video during transcription "
                    "and shown in the Library tab."
                )

        with settings_col2:
            with st.container(border=True):
                vision_head_col, vision_info_col = st.columns([4, 1])
                with vision_head_col:
                    st.markdown("**Vision analysis**")
                with vision_info_col:
                    vision_model_info_popover()
                current_vision_model = config["processing"].get(
                    "vision_model", DEFAULT_VISION_MODEL_KEY
                )
                if current_vision_model not in VISION_MODEL_OPTIONS:
                    current_vision_model = DEFAULT_VISION_MODEL_KEY
                config["processing"]["vision_model"] = st.selectbox(
                    "Vision model",
                    VISION_MODEL_KEYS,
                    index=VISION_MODEL_KEYS.index(current_vision_model),
                    format_func=lambda key: VISION_MODEL_OPTIONS[key]["label"],
                    help="Local model used for frame descriptions. Larger models need more VRAM and run slower.",
                )
                vision_missing = missing_vision_packages()
                if vision_missing:
                    st.warning(
                        "Vision model dependencies are not installed: "
                        + ", ".join(vision_missing)
                    )
                    if st.button(
                        "Install vision dependencies",
                        key="install_vision_deps_btn",
                        disabled=bool(running_job_summary()),
                    ):
                        run_model_deps_install("vision")

                vision_col1, vision_col2 = st.columns(2)
                with vision_col1:
                    config["processing"]["vision_frame_interval_seconds"] = st.number_input(
                        "Frame interval (sec)",
                        min_value=1,
                        max_value=300,
                        value=int(config["processing"].get("vision_frame_interval_seconds", 30)),
                        help="Extract one frame every N seconds. Lower values greatly increase processing time.",
                    )
                with vision_col2:
                    config["processing"]["min_frames_per_video"] = st.number_input(
                        "Min frames (short clips)",
                        min_value=1,
                        max_value=20,
                        value=int(config["processing"].get("min_frames_per_video", 3)),
                        help="Short videos still get at least this many sampled frames.",
                    )

        save_config(config)

        render_pipeline_estimator_ui(config, DATA_DIR / "system_check.json")

        st.divider()

        st.subheader("Step 1 — Scan library")
        st.write(
            "Scan after adding a folder or when new videos appear on disk. "
            "This updates the catalog and library summary counts."
        )

        if st.button("Scan Library", key="scan_library_btn", disabled=not env_ready):
            run_script("scan_library.py", selected_folder)

        st.divider()

        st.subheader("Step 2 — Run processing pipeline")

        if "pipeline" not in config:
            config["pipeline"] = DEFAULT_CONFIG["pipeline"].copy()

        pipeline_steps = [
            ("normalize", "Normalize old videos", "Remux MOV/VOB/AVI/etc. to MKV"),
            ("transcribe", "Transcribe", "Whisper speech-to-text"),
            ("vision", "Analyze vision", "Local vision model frame descriptions"),
            ("metadata", "Build metadata", "Merge transcript + vision for search"),
            ("index", "Index search DB", "Embed segments into Qdrant"),
        ]

        step_overwrite = config["processing"].setdefault("step_overwrite", {})
        st.caption(
            "Check a step to include it in the run. Check **Force** to re-process that step "
            "even when output already exists."
        )

        step_cols = st.columns(2)
        for i, (key, label, help_text) in enumerate(pipeline_steps):
            with step_cols[i % 2]:
                run_col, force_col = st.columns([4, 1])
                with run_col:
                    config["pipeline"][key] = st.checkbox(
                        label,
                        value=config["pipeline"].get(key, True),
                        help=help_text,
                        key=f"pipeline_step_{key}",
                    )
                with force_col:
                    step_overwrite[key] = st.checkbox(
                        "Force",
                        value=step_overwrite.get(key, False),
                        key=f"step_overwrite_{key}",
                        help=f"Re-run {label.lower()} even if output exists.",
                    )

        save_config(config)

        selected_count = sum(1 for key, _, _ in pipeline_steps if config["pipeline"].get(key, True))
        step_summary = ", ".join(
            label for key, label, _ in pipeline_steps if config["pipeline"].get(key, True)
        ) or "none selected"

        st.caption(f"**{selected_count}** step(s) selected: {step_summary}")

        folder_videos = get_videos(folder=selected_folder)
        file_pick_options = {v["filename"]: v["path"] for v in folder_videos}
        if file_pick_options:
            picked_names = st.multiselect(
                "Or choose specific file(s) in this folder",
                options=sorted(file_pick_options.keys()),
                key="run_jobs_file_picker",
                help="When set, the pipeline runs only on these files instead of the whole folder.",
            )
        else:
            picked_names = []

        pipeline_scope = st.radio(
            "Pipeline scope",
            ["Selected folder only", "All library folders"],
            horizontal=True,
            help="Run the selected steps on one folder or every folder in your library.",
        )

        resume_available = can_resume_pipeline()
        resume_pipeline = False
        if resume_available:
            resume_pipeline = st.checkbox(
                "Resume from last failed/stopped pipeline",
                value=False,
                key="resume_pipeline_checkbox",
                help="Continues from the step where the previous pipeline stopped or failed.",
            )
            st.info("A previous pipeline can be resumed from where it stopped.")
            try:
                resume_state = json.loads(RESUME_FILE.read_text(encoding="utf-8"))
                resume_folders = resume_state.get("folders") or []
            except Exception:
                resume_state = {}
                resume_folders = []
            step_mismatch = find_resume_step_mismatch(config, resume_state)
            if step_mismatch:
                st.warning(
                    "Your selected pipeline steps changed since the interrupted run, "
                    "so resume will **restart from the beginning** instead of continuing "
                    "mid-pipeline.\n\n"
                    f"**Previous steps:** {format_step_key_list(step_mismatch['saved'])}\n\n"
                    f"**Current steps:** {format_step_key_list(step_mismatch['current'])}"
                )
            vision_mismatches = []
            try:
                vision_mismatches = find_vision_resume_mismatches(config, resume_folders)
            except Exception as exc:
                st.caption(
                    f"Could not check vision resume settings ({exc}). "
                    "You can still resume the pipeline."
                )
            if vision_mismatches:
                st.warning(
                    "Partial vision output exists for "
                    f"**{len(vision_mismatches)}** video(s) but your vision model or frame "
                    "settings changed since that run. Those videos will restart vision "
                    "from the beginning (not resume mid-file)."
                )
                with st.expander("Affected videos"):
                    for path in vision_mismatches[:20]:
                        st.code(path)
                    if len(vision_mismatches) > 20:
                        st.caption(f"...and {len(vision_mismatches) - 20} more.")
        elif "resume_pipeline_checkbox" in st.session_state:
            st.session_state["resume_pipeline_checkbox"] = False

        run_pipeline = st.button(
            "Run Selected Pipeline",
            type="primary",
            disabled=selected_count == 0 or not env_ready,
        )

        if run_pipeline:
            if selected_count == 0:
                st.warning("Select at least one pipeline step.")
            else:
                webhook = config.get("notifications", {}).get("discord_webhook_url", "").strip()
                if not webhook:
                    st.caption("Tip: add a Discord webhook under **Tools/System** for pipeline completion alerts.")
                if pipeline_scope == "All library folders":
                    target_folders = list(config["folders"])
                    folder_label = "All_Folders"
                    anchor_folder = target_folders[0]
                else:
                    target_folders = [selected_folder]
                    folder_label = None
                    anchor_folder = selected_folder

                if resume_pipeline and not resume_available:
                    st.warning("Nothing to resume.")
                else:
                    clear_pipeline_video_filter()
                    video_filter = None
                    if picked_names:
                        video_filter = write_pipeline_video_filter(
                            [file_pick_options[name] for name in picked_names]
                        )
                    run_pipeline_job(
                        config,
                        target_folders,
                        anchor_folder,
                        folder_label,
                        resume=resume_pipeline,
                        video_filter_file=video_filter,
                    )

        st.subheader("Quick pipeline actions")
        q_col1, q_col2, q_col3 = st.columns(3)
        saved_pipeline = config["pipeline"].copy()
        saved_skip = config["processing"].get("skip_mode")

        with q_col1:
            if st.button("Index missing only", use_container_width=True, disabled=not env_ready):
                config["pipeline"] = {k: False for k in saved_pipeline}
                config["pipeline"]["index"] = True
                config["processing"]["skip_mode"] = "missing_only"
                save_config(config)
                clear_pipeline_video_filter()
                run_pipeline_job(config, [selected_folder], selected_folder, None)
                config["pipeline"] = saved_pipeline
                config["processing"]["skip_mode"] = saved_skip
                save_config(config)

        with q_col2:
            if st.button("Refresh stale metadata/index", use_container_width=True, disabled=not env_ready):
                config["pipeline"] = {k: False for k in saved_pipeline}
                config["pipeline"]["metadata"] = True
                config["pipeline"]["index"] = True
                config["processing"]["skip_mode"] = "stale_only"
                save_config(config)
                clear_pipeline_video_filter()
                run_pipeline_job(config, [selected_folder], selected_folder, None)
                config["pipeline"] = saved_pipeline
                config["processing"]["skip_mode"] = saved_skip
                save_config(config)

        with q_col3:
            if st.button("Retry incomplete videos", use_container_width=True, disabled=not env_ready):
                folder_videos = get_videos(folder=selected_folder)
                incomplete_paths = [
                    v["path"] for v in folder_videos if video_has_incomplete_steps(v)
                ]
                if not incomplete_paths:
                    st.warning("No incomplete videos in this folder.")
                else:
                    config["processing"]["skip_mode"] = "incomplete_only"
                    save_config(config)
                    run_pipeline_job(
                        config,
                        [selected_folder],
                        selected_folder,
                        None,
                        video_filter_file=write_pipeline_video_filter(incomplete_paths),
                    )
                    config["processing"]["skip_mode"] = saved_skip
                    save_config(config)

        st.caption("Live job progress is on the **Dashboard** tab (auto-refreshes every second).")

        if st.button("Clear Job History", key="jobs_clear_history"):
            for job in JOBS_DIR.glob("*.json"):
                try:
                    job.unlink()
                except Exception:
                    pass
            steps_dir = JOBS_DIR / "steps"
            if steps_dir.exists():
                for job in steps_dir.glob("*.json"):
                    try:
                        job.unlink()
                    except Exception:
                        pass
            st.success("Job history cleared.")
            st.rerun()

with tab_search:
    job_banner_search()
    st.markdown(inject_search_results_styles(), unsafe_allow_html=True)
    st.header("Search")

    search_stats = get_search_index_stats(str(DB_FILE))
    if (
        search_stats["indexed_flags"] == 0
        and not search_stats.get("qdrant_locked")
        and not search_stats["collection_exists"]
    ):
        st.info(
            "No videos are indexed for search yet. Add a folder, run the pipeline on **Run Jobs**, "
            "and include **Index search DB** — then return here."
        )

    st.write(
        "One search box combines semantic matching with keyword overlap. "
        "Use the filters below for dates, file size, and other numeric constraints. "
        "Press **Enter** in the search box to search."
    )

    st.caption("Example queries — click to search:")
    example_cols = st.columns(len(SEARCH_EXAMPLE_QUERIES))
    for col, example in zip(example_cols, SEARCH_EXAMPLE_QUERIES):
        with col:
            if st.button(example, key=f"search_example_{example.replace(' ', '_')}"):
                st.session_state["pending_search_query"] = example
                st.rerun()

    search_folder = None
    date_from = None
    date_to = None
    min_size_mb = None
    max_size_mb = None
    min_score = 0.0
    extension_filter = ""

    if config["folders"]:
        search_scope = st.selectbox(
            "Folder",
            ["All folders"] + config["folders"],
            index=(
                config["folders"].index(config["selected_folder"]) + 1
                if config["selected_folder"] in config["folders"]
                else 0
            ),
            key="search_folder_filter",
            help="Limit search results to one library folder, or search across all folders.",
        )
        if search_scope != "All folders":
            search_folder = search_scope
    else:
        st.info("Add a folder on the Library tab to enable folder-scoped search.")

    with st.expander("More search filters", expanded=False):
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            use_date_from = st.checkbox("Modified from", key="use_search_date_from")
            date_from = (
                st.date_input("Modified from", key="search_date_from")
                if use_date_from
                else None
            )

        with filter_col2:
            use_date_to = st.checkbox("Modified to", key="use_search_date_to")
            date_to = (
                st.date_input("Modified to", key="search_date_to")
                if use_date_to
                else None
            )

        size_col1, size_col2, size_col3, size_col4 = st.columns(4)
        with size_col1:
            use_min_size = st.checkbox("Min size (MB)", key="use_min_size")
            min_size_mb = (
                st.number_input("Min MB", min_value=0.0, value=0.0, key="search_min_mb")
                if use_min_size
                else None
            )
        with size_col2:
            use_max_size = st.checkbox("Max size (MB)", key="use_max_size")
            max_size_mb = (
                st.number_input("Max MB", min_value=0.0, value=10000.0, key="search_max_mb")
                if use_max_size
                else None
            )
        with size_col3:
            min_score = st.slider(
                "Min match score",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
                help="Combined semantic + keyword score.",
            )
        with size_col4:
            extension_filter = st.text_input(
                "Extension",
                placeholder=".mkv",
                help="Optional, e.g. .mkv or .mp4",
            )

    result_limit = st.slider("Max results", min_value=1, max_value=20, value=5)

    with st.form("search_form", clear_on_submit=False):
        search_text = st.text_input(
            "Search your videos",
            value=st.session_state.get("search_query", ""),
            placeholder="Example: Christmas morning, kids swimming, grandpa at the grill...",
        )
        search_col1, search_col2 = st.columns([1, 4])
        with search_col1:
            run_search = st.form_submit_button("Search", type="primary")
        with search_col2:
            clear_search = st.form_submit_button("Clear results")

    if clear_search:
        st.session_state.pop("search_results", None)
        st.session_state.pop("search_error", None)
        st.session_state.pop("search_query", None)
        st.session_state.pop("search_log", None)
        st.session_state.pop("pending_search_query", None)
        st.rerun()

    pending_query = st.session_state.pop("pending_search_query", None)
    if run_search or pending_query:
        query = (search_text.strip() if run_search else str(pending_query or "").strip())
        if not query:
            st.session_state["search_results"] = []
            st.session_state["search_error"] = "Enter a search query."
            st.session_state["search_query"] = ""
            st.session_state["search_log"] = None
        else:
            with st.spinner("Searching..."):
                try:
                    results, error, search_log = search_videos(
                        query,
                        limit=result_limit,
                        folder_filter=search_folder,
                        date_from=date_from,
                        date_to=date_to,
                        min_size_mb=min_size_mb,
                        max_size_mb=max_size_mb,
                        min_score=min_score if min_score > 0 else None,
                        extension_filter=extension_filter,
                        db_file=str(DB_FILE),
                    )
                except Exception as e:
                    results, error, search_log = [], str(e), None

            st.session_state["search_results"] = results
            st.session_state["search_error"] = error
            st.session_state["search_query"] = query
            st.session_state["search_log"] = search_log

    search_error = st.session_state.get("search_error")
    search_results = st.session_state.get("search_results")
    search_query = st.session_state.get("search_query", "")
    search_log = st.session_state.get("search_log")

    if search_log:
        expand_log = bool(search_error or not search_results)
        with st.expander("Search debug log", expanded=expand_log):
            st.code(format_search_log(search_log), language=None)
            st.caption(f"Also appended to **logs/{SEARCH_LOG_FILE.name}** (viewable on the Logs tab).")

    if search_error:
        st.warning(search_error)
    elif search_results:
        st.success(
            f"Found {len(search_results)} matching segments"
            + (f" for **{search_query}**" if search_query else "")
        )

        export_col1, export_col2 = st.columns(2)
        with export_col1:
            st.download_button(
                "Export CSV",
                data=search_results_to_csv(search_results),
                file_name="search_results.csv",
                mime="text/csv",
                key="export_search_csv",
            )
        with export_col2:
            st.download_button(
                "Export Markdown",
                data=search_results_to_markdown(search_results, search_query),
                file_name="search_results.md",
                mime="text/markdown",
                key="export_search_md",
            )

        for i, result in enumerate(search_results, start=1):
            preview = search_result_preview_text(result)
            meta_line = html.escape(search_result_meta_line(result))
            path_exists = Path(result["video_path"]).exists()
            thumb_path = get_search_result_thumbnail(result)

            with st.container(border=True):
                top_col1, top_col2, top_col3, top_col4 = st.columns([1, 4, 2, 1])
                with top_col1:
                    if thumb_path:
                        st.image(str(thumb_path), use_container_width=True)
                with top_col2:
                    safe_name = html.escape(result["filename"])
                    st.markdown(
                        f'<div class="search-result-block">'
                        f'<p class="search-result-title">#{i} {safe_name}</p>'
                        f'<p class="search-result-meta">{meta_line}</p>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    if not path_exists:
                        st.warning("Video file not found at indexed path.")
                with top_col3:
                    st.markdown(
                        format_search_score_badge(result["score"]),
                        unsafe_allow_html=True,
                    )
                with top_col4:
                    if st.button("Play", key=f"play_{i}", use_container_width=True):
                        ok, message = open_video_at_timestamp(
                            result["video_path"],
                            result["start"],
                        )
                        if ok:
                            st.success(message)
                        else:
                            st.error(message)

                st.markdown(
                    format_search_result_summary_html(preview),
                    unsafe_allow_html=True,
                )

                with st.expander("View transcript & details", expanded=False):
                    st.caption(result["video_path"])
                    st.caption(
                        f"Segment: {result['start_label']} – {result['end_label']}"
                    )
                    modified = format_modified_time(result.get("modified_time"))
                    if modified:
                        st.caption(f"File modified: {modified}")
                    if result.get("size_bytes"):
                        st.caption(f"Size: {result['size_bytes'] / (1024 * 1024):.1f} MB")
                    if result.get("semantic_score") is not None:
                        st.caption(
                            f"Semantic: {result['semantic_score']:.0%} · "
                            f"Keyword: {result.get('keyword_score', 0):.0%}"
                        )
                    st.markdown(format_search_result_text(result["text"]))

with tab_tools:
    job_banner_tools()
    st.header("Tools / System")

    if config.get("setup_wizard_complete") and st.button("Show setup wizard again"):
        config["setup_wizard_complete"] = False
        save_config(config)
        st.rerun()

    st.subheader("Catalog maintenance")
    maint_col1, maint_col2 = st.columns(2)
    with maint_col1:
        if st.button("Prune ghost catalog entries", disabled=bool(running_job_summary())):
            removed = prune_missing_videos(str(DB_FILE))
            if removed:
                st.success(f"Removed {len(removed)} catalog entries whose files no longer exist.")
            else:
                st.info("No ghost catalog entries found.")
    with maint_col2:
        if st.button("Reconcile search index", disabled=bool(running_job_summary())):
            removed = prune_orphan_qdrant_points(str(DB_FILE))
            if removed:
                st.success(f"Removed {removed} orphan search segments.")
            else:
                st.info("Search index is already consistent with the catalog.")

    all_duplicates = find_duplicate_groups(str(DB_FILE))
    if all_duplicates:
        with st.expander(
            f"Possible duplicates library-wide ({len(all_duplicates)} group(s))",
            expanded=False,
        ):
            st.caption(
                "Matched by content fingerprint or identical size + duration across all folders."
            )
            for group in all_duplicates[:25]:
                st.markdown(
                    f"**{group['match_type']}** — "
                    + ", ".join(v["filename"] for v in group["videos"])
                )
            if len(all_duplicates) > 25:
                st.caption(f"...and {len(all_duplicates) - 25} more groups.")

    st.subheader("Paths")
    st.write("Project folder:")
    st.code(str(APP_DIR))

    st.write("Database:")
    st.code(str(DB_FILE))

    st.write("Scripts folder:")
    st.code(str(SCRIPT_DIR))

    st.write("Logs folder:")
    st.code(str(LOG_DIR))

    st.write("Qdrant search DB:")
    st.code(str(DATA_DIR / "qdrant"))

    st.subheader("Search index")
    index_stats = get_search_index_stats(str(DB_FILE))
    if index_stats.get("qdrant_locked"):
        st.warning(
            "Search index is busy (indexing job running). "
            "Segment counts and search will work again when the job finishes."
        )
    st.write(
        f"Searchable segments: **{index_stats['segment_count']}** · "
        f"Videos in catalog: **{index_stats['catalog_videos']}** · "
        f"Marked indexed: **{index_stats['indexed_flags']}**"
    )
    if not index_stats["collection_exists"] and not index_stats.get("qdrant_locked"):
        st.info("No search index yet. Run **Index search DB** on the Run Jobs tab.")

    reset_col1, reset_col2 = st.columns([1, 3])
    with reset_col1:
        confirm_reset = st.checkbox(
            "I understand this wipes all search data",
            key="confirm_reset_search_index",
        )
        if st.button(
            "Reset search index",
            disabled=not confirm_reset,
            type="primary",
        ):
            ok, message = reset_search_index(str(DB_FILE))
            st.session_state.pop("search_results", None)
            st.session_state.pop("search_error", None)
            st.session_state.pop("search_query", None)
            if ok:
                st.success(message)
            else:
                st.error(message)
            st.rerun()
    with reset_col2:
        st.caption(
            "Clears the Qdrant vector database and resets indexed flags in the catalog. "
            "Run **Index search DB** afterward to rebuild search from your metadata files."
        )

    st.subheader("Discord Notifications")

    st.write(
        "Paste your own Discord webhook URL here. Each person running this app can use their own webhook. "
        "Nothing is hard-coded — the URL is saved only in your local `config.json`."
    )

    if "notifications" not in config:
        config["notifications"] = DEFAULT_CONFIG["notifications"].copy()

    with st.form("discord_settings_form", clear_on_submit=False):
        webhook_url = st.text_input(
            "Discord webhook URL",
            value=config["notifications"].get("discord_webhook_url", ""),
            placeholder="https://discord.com/api/webhooks/...",
            type="password",
            help="Discord channel → Edit Channel → Integrations → Webhooks → New Webhook → Copy Webhook URL",
        )
        notify_col1, notify_col2, notify_col3 = st.columns(3)
        with notify_col1:
            notify_complete = st.checkbox(
                "Notify on complete",
                value=config["notifications"].get("notify_on_complete", True),
            )
        with notify_col2:
            notify_failed = st.checkbox(
                "Notify on failed",
                value=config["notifications"].get("notify_on_failed", True),
            )
        with notify_col3:
            notify_stopped = st.checkbox(
                "Notify on stopped",
                value=config["notifications"].get("notify_on_stopped", False),
            )
        save_notifications = st.form_submit_button("Save notification settings")

    if save_notifications:
        config["notifications"]["discord_webhook_url"] = webhook_url
        config["notifications"]["notify_on_complete"] = notify_complete
        config["notifications"]["notify_on_failed"] = notify_failed
        config["notifications"]["notify_on_stopped"] = notify_stopped
        save_config(config)
        st.success("Notification settings saved.")

    test_col1, test_col2 = st.columns([1, 4])
    with test_col1:
        if st.button("Test Discord"):
            webhook = config["notifications"].get("discord_webhook_url", "").strip()
            if not webhook:
                st.warning("Enter a Discord webhook URL first.")
            else:
                ok, err = send_discord_webhook(
                    webhook,
                    "Video Indexer test",
                    "Discord notifications are connected.",
                )
                if ok:
                    st.success("Test message sent.")
                else:
                    st.error(err or "Could not send test message.")

    st.subheader("Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Open Project Folder"):
            subprocess.Popen(["explorer", str(APP_DIR)])

    with col2:
        if st.button("Open Logs Folder"):
            subprocess.Popen(["explorer", str(LOG_DIR)])

    with col3:
        if st.button("Open Data Folder"):
            subprocess.Popen(["explorer", str(DATA_DIR)])

    st.subheader("Dependency Manager")

    running_job = running_job_summary()
    if running_job and running_job not in INSTALL_JOB_SCRIPTS:
        st.caption(
            f"Install is disabled while **{running_job}** is running. "
            "Stop it on the Dashboard first."
        )

    if st.button("Install / Update AI Dependencies", disabled=bool(running_job)):
        run_script("install_dependencies.py", config.get("selected_folder", "") or str(APP_DIR))

    render_dependency_install_status()

    st.subheader("System Check")

    if st.button("Refresh System Check"):
        result = subprocess.run(
            [str(APP_DIR / "venv" / "Scripts" / "python.exe"), str(SCRIPT_DIR / "system_check.py")],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            st.success("System check refreshed.")
        else:
            st.error("System check failed.")
            st.code(result.stderr or result.stdout)

    system_check_file = DATA_DIR / "system_check.json"

    if system_check_file.exists():
        check = json.loads(system_check_file.read_text(encoding="utf-8"))

        st.write("### System Information")
        st.dataframe(
            pd.DataFrame(check.get("system", [])),
            width="stretch",
            hide_index=True
        )

        st.write("### Dependency Status")
        st.dataframe(
            pd.DataFrame(check.get("dependencies", [])),
            width="stretch",
            hide_index=True
        )
    else:
        st.info("Click Refresh System Check to generate system/dependency information.")

    st.subheader("Database Summary")

    db_stats = get_library_stats(str(DB_FILE))
    st.write(f"Total scanned videos: **{db_stats['total']}**")
    st.write(f"With transcripts: **{db_stats['transcribed']}**")
    st.write(f"With vision: **{db_stats['vision']}**")
    st.write(f"Indexed in search DB: **{db_stats['indexed']}**")

with tab_logs:
    job_banner_logs()
    st.header("Logs")

    col_clear1, col_clear2 = st.columns([1, 4])

    with col_clear1:
        if st.button("Clear All Logs"):
            for log in LOG_DIR.glob("*.log"):
                try:
                    log.unlink()
                except Exception:
                    pass

            for job in JOBS_DIR.glob("*.json"):
                try:
                    job.unlink()
                except Exception:
                    pass
                
            st.success("Logs cleared.")
            st.rerun()

    render_logs_live()

with tab_help:
    job_banner_help()
    render_help_tab()
