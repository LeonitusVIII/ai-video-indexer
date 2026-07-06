import json
import platform
import shutil
import subprocess
import sys
import importlib.util
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
OUT_FILE = APP_DIR / "data" / "system_check.json"
OUT_FILE.parent.mkdir(exist_ok=True)


PACKAGES = {
    "streamlit": "streamlit",
    "pandas": "pandas",
    "psutil": "psutil",
    "requests": "requests",
    "faster-whisper": "faster_whisper",
    "ctranslate2": "ctranslate2",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
    "opencv-python": "cv2",
    "pillow": "PIL",
    "ffmpeg-python": "ffmpeg",
    "qdrant-client": "qdrant_client",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "sentence-transformers": "sentence_transformers",
    "qwen-vl-utils": "qwen_vl_utils",
    "nvidia-cublas-cu12": "nvidia.cublas",
    "nvidia-cudnn-cu12": "nvidia.cudnn",
}


def run_cmd(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return (r.stdout or r.stderr or "").strip()
    except Exception as e:
        return str(e)


def package_status():
    rows = []

    for package_name, import_name in PACKAGES.items():
        found = importlib.util.find_spec(import_name) is not None
        version = ""

        if found:
            try:
                mod = __import__(import_name.split(".")[0])
                version = getattr(mod, "__version__", "")
            except Exception:
                version = ""

        rows.append({
            "Dependency": package_name,
            "Import": import_name,
            "Installed": "Yes" if found else "No",
            "Version": version
        })

    return rows


def system_info():
    rows = []

    rows.append({"Item": "Python", "Value": sys.version.replace("\n", " ")})
    rows.append({"Item": "Python executable", "Value": sys.executable})
    rows.append({"Item": "Operating system", "Value": platform.platform()})
    rows.append({"Item": "CPU", "Value": platform.processor() or "Unknown"})

    try:
        import psutil
        rows.append({"Item": "CPU cores", "Value": str(psutil.cpu_count(logical=True))})
        rows.append({"Item": "RAM total", "Value": f"{psutil.virtual_memory().total / (1024**3):.1f} GB"})
        rows.append({"Item": "RAM available", "Value": f"{psutil.virtual_memory().available / (1024**3):.1f} GB"})
        disk = psutil.disk_usage(str(APP_DIR.drive + "\\"))
        rows.append({"Item": "Disk total", "Value": f"{disk.total / (1024**3):.1f} GB"})
        rows.append({"Item": "Disk free", "Value": f"{disk.free / (1024**3):.1f} GB"})
    except Exception as e:
        rows.append({"Item": "psutil error", "Value": str(e)})

    ffmpeg_path = shutil.which("ffmpeg")
    rows.append({"Item": "FFmpeg path", "Value": ffmpeg_path or "Not found"})
    if ffmpeg_path:
        ffmpeg_ver = run_cmd(["ffmpeg", "-version"]).splitlines()[0]
        rows.append({"Item": "FFmpeg version", "Value": ffmpeg_ver})

    nvidia_smi = shutil.which("nvidia-smi")
    rows.append({"Item": "nvidia-smi", "Value": nvidia_smi or "Not found"})

    if nvidia_smi:
        gpu_query = run_cmd([
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits"
        ])
        rows.append({"Item": "GPU", "Value": gpu_query or "No GPU output"})

    try:
        import torch
        rows.append({"Item": "Torch CUDA available", "Value": str(torch.cuda.is_available())})
        if torch.cuda.is_available():
            rows.append({"Item": "Torch CUDA device", "Value": torch.cuda.get_device_name(0)})
            rows.append({"Item": "Torch CUDA version", "Value": str(torch.version.cuda)})
    except Exception as e:
        rows.append({"Item": "Torch check", "Value": str(e)})

    if importlib.util.find_spec("torchcodec") is not None:
        try:
            import torchcodec  # noqa: F401
            rows.append({"Item": "torchcodec", "Value": "Installed and loadable"})
        except Exception as exc:
            rows.append({
                "Item": "torchcodec",
                "Value": (
                    "Installed but broken — run: pip uninstall -y torchcodec "
                    f"({type(exc).__name__})"
                ),
            })

    return rows


def main():
    data = {
        "system": system_info(),
        "dependencies": package_status()
    }

    OUT_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()