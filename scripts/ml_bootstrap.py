"""Shared ML environment setup for Video Indexer scripts.

A broken torchcodec install raises RuntimeError on import, which breaks
sentence-transformers even though text-only models do not need torchcodec.
"""
import importlib.util
import subprocess
import sys


def _torchcodec_is_broken():
    if importlib.util.find_spec("torchcodec") is None:
        return False

    try:
        import torchcodec  # noqa: F401
    except Exception:
        return True

    return False


def remove_broken_torchcodec():
    """Uninstall torchcodec when its native libraries fail to load."""
    if not _torchcodec_is_broken():
        return False

    print(
        "Removing broken torchcodec package (not required for this app)...",
        flush=True,
    )
    result = subprocess.run(
        [sys.executable, "-m", "pip", "uninstall", "-y", "torchcodec"],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "torchcodec is installed but cannot load its native libraries, and "
            "automatic removal failed. Run manually: pip uninstall -y torchcodec"
            + (f"\n{stderr}" if stderr else "")
        )

    if "torchcodec" in sys.modules:
        del sys.modules["torchcodec"]

    return True


def prepare_ml_environment(*, remove_torchcodec=True):
    """Apply environment fixes before importing sentence-transformers."""
    if remove_torchcodec:
        remove_broken_torchcodec()
