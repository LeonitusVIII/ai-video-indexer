"""Disk space helpers for local paths and Windows UNC/network shares."""
from __future__ import annotations

import ctypes
import shutil
from pathlib import Path


def _usage_via_ctypes(path: str):
    """Windows GetDiskFreeSpaceExW — often works on UNC paths when shutil fails."""
    free_user = ctypes.c_ulonglong()
    total = ctypes.c_ulonglong()
    free_total = ctypes.c_ulonglong()
    ok = ctypes.windll.kernel32.GetDiskFreeSpaceExW(
        ctypes.c_wchar_p(path),
        ctypes.byref(free_user),
        ctypes.byref(total),
        ctypes.byref(free_total),
    )
    if not ok:
        raise OSError(ctypes.get_last_error(), f"GetDiskFreeSpaceExW failed for {path}")
    total_bytes = int(total.value)
    free_bytes = int(free_total.value)
    used_bytes = max(0, total_bytes - free_bytes)
    return total_bytes, used_bytes, free_bytes


def _resolve_query_path(path: str | Path):
    """Pick an existing directory to query, walking up parents if needed."""
    candidate = Path(path)
    if candidate.is_file():
        candidate = candidate.parent
    for item in [candidate, *candidate.parents]:
        if item.exists() and item.is_dir():
            return item
    return None


def get_disk_space(path: str | Path):
    """
    Return volume space for the drive/share backing ``path``.

    Works for local folders and many Windows UNC shares (``\\\\server\\share``).
    """
    query = _resolve_query_path(path)
    if query is None:
        return {
            "available": False,
            "path_queried": str(path),
            "error": "Path is not accessible from this PC.",
        }

    query_str = str(query)
    last_error = None
    for attempt in (query_str, query_str.rstrip("\\") + "\\" if query_str.startswith("\\\\") else None):
        if not attempt:
            continue
        try:
            usage = shutil.disk_usage(attempt)
            return _pack_usage(attempt, usage.total, usage.total - usage.free, usage.free, "shutil")
        except OSError as exc:
            last_error = str(exc)

    try:
        total, used, free = _usage_via_ctypes(query_str)
        return _pack_usage(query_str, total, used, free, "winapi")
    except OSError as exc:
        last_error = str(exc)

    return {
        "available": False,
        "path_queried": query_str,
        "error": last_error or "Could not read free space for this path.",
    }


def _pack_usage(path_queried, total, used, free, method):
    total = int(total)
    used = int(used)
    free = int(free)
    pct_used = (used / total * 100) if total else 0
    return {
        "available": True,
        "path_queried": str(path_queried),
        "method": method,
        "total_bytes": total,
        "used_bytes": used,
        "free_bytes": free,
        "percent_used": round(pct_used, 1),
    }


def format_bytes(num_bytes, *, precision=1):
    value = float(num_bytes or 0)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    unit = units[0]
    for unit in units:
        if value < 1024 or unit == units[-1]:
            break
        value /= 1024
    if unit == "B":
        return f"{int(value)} {unit}"
    return f"{value:.{precision}f} {unit}"


def format_disk_line(info):
    if not info.get("available"):
        return f"Disk space unavailable — {info.get('error', 'unknown error')}"
    return (
        f"{format_bytes(info['free_bytes'])} free of {format_bytes(info['total_bytes'])} "
        f"({info['percent_used']}% used)"
    )


def estimate_transcode_bytes(video_paths, *, output_mode="suffix", safety_factor=1.15):
    """
    Estimate temporary space needed for a transcode batch.

    Suffix mode keeps originals, so outputs add to used space (conservative sum).
    Replace mode only needs headroom for the largest single encode at a time.
    """
    sizes = []
    for path in video_paths:
        try:
            sizes.append(Path(path).stat().st_size)
        except OSError:
            continue
    if not sizes:
        return 0
    if output_mode == "replace":
        return int(max(sizes) * safety_factor)
    return int(sum(sizes) * safety_factor)


def transcode_space_check(folder, video_paths, *, output_mode="suffix", min_free_gb=5):
    """
    Return whether a transcode batch likely fits, with human-readable detail.
    """
    disk = get_disk_space(folder)
    needed = estimate_transcode_bytes(video_paths, output_mode=output_mode)
    headroom = int(max(0, float(min_free_gb)) * 1024 ** 3)
    required = needed + headroom
    result = {
        "disk": disk,
        "needed_bytes": needed,
        "headroom_bytes": headroom,
        "required_bytes": required,
        "ok": False,
        "message": "",
    }
    if not disk.get("available"):
        result["ok"] = True
        result["message"] = (
            "Could not verify free space for this path (common on some network shares). "
            "Proceed with caution."
        )
        return result
    free = disk["free_bytes"]
    result["ok"] = free >= required
    if result["ok"]:
        result["message"] = (
            f"Estimated need {format_bytes(needed)} (+ {format_bytes(headroom)} headroom); "
            f"{format_bytes(free)} available."
        )
    else:
        shortfall = required - free
        result["message"] = (
            f"Need about {format_bytes(required)} ({format_bytes(needed)} encode workspace + "
            f"{format_bytes(headroom)} headroom) but only {format_bytes(free)} free "
            f"({format_bytes(shortfall)} short)."
        )
    return result
