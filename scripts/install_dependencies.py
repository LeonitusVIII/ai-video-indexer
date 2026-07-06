import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path


def write_status(status_file, status):
    Path(status_file).write_text(json.dumps(status, indent=2), encoding="utf-8")


def run_step(name, cmd, status, status_file, optional=False):
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(cmd), flush=True)

    status["current"] = name
    write_status(status_file, status)

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0 and not optional:
        status["status"] = "failed"
        status["current"] = f"Failed: {name}"
        status["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
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

    status = {
        "status": "running",
        "percent": 0,
        "processed": 0,
        "total": len(steps),
        "current": "Starting dependency install",
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "finished_at": "",
        "stop_requested": False,
    }
    write_status(status_file, status)

    for i, step in enumerate(steps, start=1):
        if len(step) == 3:
            name, cmd, optional = step
        else:
            name, cmd = step
            optional = False

        status["processed"] = i - 1
        status["percent"] = int(((i - 1) / len(steps)) * 100)
        write_status(status_file, status)

        run_step(name, cmd, status, status_file, optional=optional)

        status["processed"] = i
        status["percent"] = int((i / len(steps)) * 100)
        write_status(status_file, status)

    status["status"] = "complete"
    status["percent"] = 100
    status["current"] = "Dependency install complete"
    status["finished_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    write_status(status_file, status)

    print("\nDependency install complete.", flush=True)


if __name__ == "__main__":
    main()
