import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_utils import read_status, should_stop, write_status


def run_step(name, cmd, status, status_file, optional=False):
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(cmd), flush=True)

    status["current"] = name
    write_status(status_file, status)

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0 and not optional:
        status = read_status(status_file)
        status.update({
            "status": "failed",
            "current": f"Failed: {name}",
            "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        write_status(status_file, status)
        sys.exit(result.returncode)


def build_steps(python):
    """Return ordered install steps covering all project Python dependencies."""
    return [
        (
            "Upgrade pip",
            [python, "-m", "pip", "install", "--upgrade", "pip"],
        ),
        (
            "Install core app packages",
            [
                python,
                "-m",
                "pip",
                "install",
                "streamlit",
                "pandas",
                "psutil",
                "requests",
            ],
        ),
        (
            "Install transcription stack",
            [
                python,
                "-m",
                "pip",
                "install",
                "faster-whisper",
                "nvidia-cublas-cu12",
                "nvidia-cudnn-cu12",
            ],
        ),
        (
            "Install PyTorch CUDA build",
            [
                python,
                "-m",
                "pip",
                "install",
                "torch",
                "torchvision",
                "torchaudio",
                "--index-url",
                "https://download.pytorch.org/whl/cu128",
            ],
        ),
        (
            "Remove torchcodec (broken on Windows; not needed)",
            [python, "-m", "pip", "uninstall", "-y", "torchcodec"],
            True,
        ),
        (
            "Install video processing",
            [
                python,
                "-m",
                "pip",
                "install",
                "opencv-python",
                "pillow",
                "ffmpeg-python",
            ],
        ),
        (
            "Install search and embeddings",
            [
                python,
                "-m",
                "pip",
                "install",
                "qdrant-client",
                "sentence-transformers",
            ],
        ),
        (
            "Install vision stack",
            [
                python,
                "-m",
                "pip",
                "install",
                "transformers",
                "accelerate",
                "qwen-vl-utils",
            ],
        ),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=False, default="")
    parser.add_argument("--db", required=False, default="")
    parser.add_argument("--status-file", required=True)
    args = parser.parse_args()

    status_file = Path(args.status_file)
    python = sys.executable
    steps = build_steps(python)

    status = read_status(status_file)
    status.update({
        "status": "running",
        "percent": 0,
        "processed": 0,
        "total": len(steps),
        "current": "Starting dependency install",
        "stop_requested": status.get("stop_requested", False),
    })
    if not status.get("started_at"):
        status["started_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    status["finished_at"] = ""
    write_status(status_file, status)

    for i, step in enumerate(steps, start=1):
        if should_stop(status_file):
            status = read_status(status_file)
            status.update({
                "status": "stopped",
                "current": "Stopped by user.",
                "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
            })
            write_status(status_file, status)
            print("Stopped by user.", flush=True)
            sys.exit(0)

        if len(step) == 3:
            name, cmd, optional = step
        else:
            name, cmd = step
            optional = False

        status["processed"] = i - 1
        status["percent"] = int(((i - 1) / len(steps)) * 100)
        write_status(status_file, status)

        run_step(name, cmd, status, status_file, optional=optional)

        status = read_status(status_file)
        status["processed"] = i
        status["percent"] = int((i / len(steps)) * 100)
        write_status(status_file, status)

    status = read_status(status_file)
    status.update({
        "status": "complete",
        "percent": 100,
        "current": "Dependency install complete",
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    write_status(status_file, status)

    print("\nDependency install complete.", flush=True)


if __name__ == "__main__":
    main()
