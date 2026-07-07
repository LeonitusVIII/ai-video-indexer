"""Shared UI components for job monitoring."""
from pathlib import Path

import streamlit as st

from app_helpers import compute_job_elapsed

PIPELINE_STEP_KEYS = (
    "scan",
    "normalize",
    "transcribe",
    "vision",
    "metadata",
    "index",
)

STANDALONE_JOB_LABELS = {
    "scan_library.py": "Scan library",
    "install_dependencies.py": "Install dependencies",
    "install_model_deps.py": "Install model dependencies",
    "system_check.py": "System check",
}


def tail_log_file(log_path, max_lines=20):
    path = Path(log_path)
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(lines[-max_lines:])


def is_pipeline_job(job):
    return job.get("script") == "run_pipeline.py"


def infer_running_step_key(job_file, read_job_fn):
    steps_dir = job_file.parent / "steps"
    if not steps_dir.exists():
        return None

    prefix = f"{job_file.stem}_"
    fallback = None
    for step_key in PIPELINE_STEP_KEYS:
        step_path = steps_dir / f"{prefix}{step_key}.json"
        if not step_path.exists():
            continue
        if (read_job_fn(step_path) or {}).get("status") == "running":
            return step_key
        fallback = (step_path.stat().st_mtime, step_key)

    if fallback:
        return fallback[1]
    return None


def resolve_step_status(job, job_file, read_job_fn):
    if is_pipeline_job(job):
        step_key = job.get("pipeline_step_key") or infer_running_step_key(
            job_file, read_job_fn
        )
        if not step_key:
            return None
        step_path = job_file.parent / "steps" / f"{job_file.stem}_{step_key}.json"
        if step_path.exists():
            return read_job_fn(step_path)
        return None
    return job


def _progress_detail(processed, total, unit=""):
    if total:
        suffix = f" {unit}".strip()
        return f"{processed}/{total}{suffix}"
    return ""


def _item_unit(job):
    step_key = job.get("pipeline_step_key") or ""
    script = job.get("script") or ""
    if step_key == "vision" or script == "analyze_vision.py":
        return "frames"
    if step_key == "transcribe" or script == "transcribe.py":
        return "segments"
    if step_key == "scan" or script == "scan_library.py":
        return "files"
    return "parts"


def _standalone_job_label(job):
    script = job.get("script") or ""
    if script in STANDALONE_JOB_LABELS:
        return STANDALONE_JOB_LABELS[script]
    return script.replace(".py", "").replace("_", " ").title() or "Job"


def _standalone_progress_unit(job):
    script = job.get("script") or ""
    if script == "scan_library.py":
        return "files"
    if script in ("install_dependencies.py", "install_model_deps.py"):
        return "steps"
    return "items"


def render_job_progress_layers(job, job_file, read_job_fn, *, compact=False):
    """Render overall, current-step, and optional within-file progress bars."""
    step_status = resolve_step_status(job, job_file, read_job_fn)
    is_pipeline = is_pipeline_job(job)

    if is_pipeline:
        overall_percent = int(job.get("percent", 0))
        st.caption(
            "Overall pipeline · "
            f"{_progress_detail(job.get('processed', 0), job.get('total', 0), 'steps')} · "
            f"{overall_percent}%"
        )
        st.progress(overall_percent / 100)
    elif job.get("status") == "running" or int(job.get("percent", 0)) > 0:
        label = _standalone_job_label(job)
        processed = int(job.get("processed", 0) or 0)
        total = int(job.get("total", 0) or 0)
        percent = int(job.get("percent", 0) or 0)
        unit = _standalone_progress_unit(job)
        current = (job.get("current") or "").strip()
        if total > 0:
            st.caption(
                f"{label} · {_progress_detail(processed, total, unit)} · {percent}%"
            )
        elif current:
            st.caption(f"{label} · {current[:120]}")
        else:
            st.caption(f"{label} · Working...")
        st.progress(max(min(percent, 100), 0) / 100)

    active = step_status or job
    step_total = int(active.get("total", 0) or 0)
    if is_pipeline and step_total > 0:
        step_percent = int(active.get("percent", 0))
        step_label = job.get("pipeline_step") or "Current step"
        step_unit = _item_unit(job)
        st.caption(
            f"{step_label} · "
            f"{_progress_detail(active.get('processed', 0), step_total, step_unit)} · "
            f"{step_percent}%"
        )
        st.progress(step_percent / 100)

    item_total = int(active.get("item_total", 0) or 0)
    if item_total > 0:
        item_percent = int(active.get("item_percent", 0) or 0)
        item_label = active.get("item_label") or "Current file"
        if compact and len(item_label) > 48:
            item_label = item_label[:45] + "…"
        item_unit = _item_unit(job)
        st.caption(
            f"{item_label} · "
            f"{_progress_detail(active.get('item_processed', 0), item_total, item_unit)} · "
            f"{item_percent}%"
        )
        st.progress(item_percent / 100)


def render_job_status_banner(job_files, read_job_fn, stop_job_fn, *, key_prefix=""):
    """Compact running-job strip for non-dashboard tabs."""
    if not job_files:
        return

    job_file = job_files[0]
    job = read_job_fn(job_file)
    if not job or job.get("status") != "running":
        return

    percent = int(job.get("percent", 0))
    script_name = job.get("script") or job.get("job_id", "Job")
    folder = job.get("folder") or ""
    if len(folder) > 72:
        folder = folder[:69] + "…"
    elapsed = compute_job_elapsed(job) or "—"
    processed = job.get("processed", 0)
    total = job.get("total", 0)
    current = job.get("current") or ""

    with st.container(border=True):
        banner_col1, banner_col2, banner_col3, banner_col4 = st.columns([2.2, 3.2, 2.2, 0.7])
        with banner_col1:
            st.markdown(f"**Running:** {script_name}")
            if folder:
                st.caption(folder)
            if job.get("pipeline_step"):
                st.caption(f"Step: **{job['pipeline_step']}**")
        with banner_col2:
            render_job_progress_layers(job, job_file, read_job_fn, compact=True)
            if current and not int(job.get("item_total", 0) or 0):
                st.caption(f"Current: {current[:120]}")
        with banner_col3:
            unit = "steps" if is_pipeline_job(job) else "files"
            st.caption(
                f"**{percent}%** complete · {processed} / {total} {unit} · Elapsed **{elapsed}**"
            )
        with banner_col4:
            if st.button("Stop", key=f"{key_prefix}banner_stop", use_container_width=True):
                stop_job_fn(job_file)
                st.warning("Stop requested.")


def render_job_panel(
    job_files,
    read_job_fn,
    stop_job_fn,
    *,
    jobs_dir,
    show_clear=False,
    key_prefix="",
    empty_message="No jobs have run yet.",
):
    if show_clear and st.button("Clear Job History", key=f"{key_prefix}clear_jobs"):
        for job in Path(jobs_dir).glob("*.json"):
            try:
                job.unlink()
            except Exception:
                pass
        steps_dir = Path(jobs_dir) / "steps"
        if steps_dir.exists():
            for job in steps_dir.glob("*.json"):
                try:
                    job.unlink()
                except Exception:
                    pass
        st.success("Job history cleared.")
        st.rerun()

    if not job_files:
        st.info(empty_message)
        return

    job_file = job_files[0]
    job = read_job_fn(job_file)
    if not job:
        st.warning("Could not read job status.")
        return

    with st.container(border=True):
        st.write(f"**{job.get('script') or job.get('job_id', 'Unknown job')}**")
        st.caption(job.get("folder", ""))

        pipeline_steps = job.get("pipeline_steps")
        if pipeline_steps:
            st.caption("Pipeline: " + " → ".join(pipeline_steps))

        skip_mode = job.get("skip_mode")
        if skip_mode and skip_mode != "all":
            st.caption(f"Skip mode: **{skip_mode.replace('_', ' ')}**")

        render_job_progress_layers(job, job_file, read_job_fn)

        percent = int(job.get("percent", 0))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Status", job.get("status", "unknown"))
        col2.metric("Progress", f"{percent}%")
        unit_label = "Steps" if is_pipeline_job(job) else "Files"
        col3.metric(
            unit_label,
            f"{job.get('processed', 0)} / {job.get('total', 0)}",
        )
        col4.metric("Elapsed", compute_job_elapsed(job) or "—")

        if job.get("pid") and job.get("status") == "running":
            st.caption(f"PID {job.get('pid')}")

        current = job.get("current")
        if current:
            st.write("Current item:")
            st.code(current)

        log_file = job.get("log_file")
        if log_file:
            st.caption(f"Log: {Path(log_file).name}")
            tail = tail_log_file(log_file)
            if tail:
                st.text_area(
                    "Recent log output",
                    tail,
                    height=160,
                    disabled=True,
                    key=f"{key_prefix}job_log_tail",
                )

        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            if job.get("status") == "running":
                if st.button("Stop Job", key=f"{key_prefix}stop_job"):
                    stop_job_fn(job_file)
                    st.warning("Stop requested.")
        with btn_col2:
            if st.button("Refresh", key=f"{key_prefix}refresh_job"):
                st.rerun()
