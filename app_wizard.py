"""First-run setup wizard for new installs."""
import shutil
import subprocess
from pathlib import Path

import streamlit as st


def check_python():
    import sys
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 11)
    return ok, f"{major}.{minor}"


def check_venv(app_dir):
    venv_python = Path(app_dir) / "venv" / "Scripts" / "python.exe"
    return venv_python.exists(), str(venv_python)


def check_ffmpeg():
    return shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "Not found"


def check_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return True, result.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return False, "No NVIDIA GPU detected (CPU mode only)"


def wizard_complete(config):
    return config.get("setup_wizard_complete", False)


def mark_wizard_complete(config):
    config["setup_wizard_complete"] = True
    return config


def environment_ready(app_dir):
    """True when venv and FFmpeg are available for processing jobs."""
    venv_ok, _ = check_venv(app_dir)
    ffmpeg_ok, _ = check_ffmpeg()
    return venv_ok and ffmpeg_ok


def render_setup_wizard(config, app_dir, save_config_fn):
    if wizard_complete(config):
        return config

    st.info(
        "Welcome to AI Video Indexer. Complete these checks, then add a folder on the "
        "**Library** tab and run **Scan Library** on **Run Jobs**."
    )

    py_ok, py_ver = check_python()
    venv_ok, venv_path = check_venv(app_dir)
    ffmpeg_ok, ffmpeg_path = check_ffmpeg()
    gpu_ok, gpu_info = check_gpu()

    checks = [
        ("Python 3.11+", py_ok, py_ver),
        ("Virtual environment (run setup.bat)", venv_ok, venv_path if venv_ok else "Missing"),
        ("FFmpeg on PATH", ffmpeg_ok, ffmpeg_path),
        ("GPU (optional)", gpu_ok, gpu_info),
    ]

    for label, ok, detail in checks:
        icon = "✓" if ok else "·"
        st.write(f"{icon} **{label}** — `{detail}`")

    folders_ok = bool(config.get("folders"))
    st.write(
        f"{'✓' if folders_ok else '·'} **Video folder added** — "
        f"{len(config.get('folders', []))} folder(s) configured"
    )

    if st.button("Mark setup complete", type="primary"):
        config = mark_wizard_complete(config)
        save_config_fn(config)
        st.success("Setup wizard dismissed. You can reopen it from Tools/System.")
        st.rerun()

    return config
