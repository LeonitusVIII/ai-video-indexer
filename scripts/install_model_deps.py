import argparse
import datetime
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from job_utils import read_status, should_stop, write_status

STACKS = {
    "vision": [
        (
            "Install PyTorch CUDA build",
            [
                sys.executable,
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
            "Install vision model stack",
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "transformers",
                "accelerate",
                "qwen-vl-utils",
                "opencv-python",
                "pillow",
            ],
        ),
    ],
}


def run_step(name, cmd, status, status_file):
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(cmd), flush=True)
    status["current"] = name
    write_status(status_file, status)
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        status = read_status(status_file)
        status.update({
            "status": "failed",
            "current": f"Failed: {name}",
            "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        })
        write_status(status_file, status)
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", choices=["vision"], required=True)
    parser.add_argument("--folder", default="")
    parser.add_argument("--db", default="")
    parser.add_argument("--status-file", required=True)
    args = parser.parse_args()

    steps = STACKS[args.stack]
    status_file = Path(args.status_file)

    status = read_status(status_file)
    status.update({
        "status": "running",
        "percent": 0,
        "processed": 0,
        "total": len(steps),
        "current": f"Installing {args.stack} dependencies",
        "stop_requested": status.get("stop_requested", False),
    })
    if not status.get("started_at"):
        status["started_at"] = datetime.datetime.now().isoformat(timespec="seconds")
    status["finished_at"] = ""
    write_status(status_file, status)

    for i, (name, cmd) in enumerate(steps, start=1):
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

        status = read_status(status_file)
        status["processed"] = i - 1
        status["percent"] = int(((i - 1) / len(steps)) * 100)
        write_status(status_file, status)
        run_step(name, cmd, status, status_file)
        status = read_status(status_file)
        status["processed"] = i
        status["percent"] = int((i / len(steps)) * 100)
        write_status(status_file, status)

    status = read_status(status_file)
    status.update({
        "status": "complete",
        "percent": 100,
        "current": f"{args.stack.title()} dependencies installed",
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    })
    write_status(status_file, status)
    print(f"\n{args.stack.title()} dependency install complete.", flush=True)


if __name__ == "__main__":
    main()
