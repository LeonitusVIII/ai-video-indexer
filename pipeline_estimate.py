"""Rough pipeline duration estimates from system profile and library size."""
from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from job_utils import DEFAULT_VISION_MODEL_KEY, VISION_MODEL_OPTIONS

LIBRARY_SIZE_LABELS = (
    ("100 MB", 100 / 1024),
    ("1 GB", 1.0),
    ("10 GB", 10.0),
    ("100 GB", 100.0),
)

# Seconds per GB (min, max) at gpu_high tier, large-v3, vision every 30s.
STEP_SEC_PER_GB = {
    "scan": (8, 25),
    "normalize": (40, 180),
    "transcribe": (480, 2400),
    "vision": (720, 4800),
    "metadata": (5, 25),
    "index": (15, 90),
}

TIER_MULTIPLIERS = {
    "gpu_high": 1.0,
    "gpu_mid": 1.35,
    "gpu_low": 2.0,
    "cpu_only": 4.5,
}

WHISPER_MODEL_MULTIPLIERS = {
    "tiny": 0.22,
    "base": 0.32,
    "small": 0.45,
    "medium": 0.7,
    "large-v2": 1.0,
    "large-v3": 1.0,
}

# Relative to Qwen2.5-VL 3B at gpu_high tier.
VISION_MODEL_MULTIPLIERS = {
    "qwen2-vl-2b": 0.6,
    "qwen2.5-vl-3b": 1.0,
    "qwen2.5-vl-7b": 2.3,
}


def _parse_system_rows(rows):
    data = {}
    for row in rows or []:
        item = row.get("Item")
        value = row.get("Value")
        if item and value is not None:
            data[item] = str(value)
    return data


def _gpu_vram_gb(system_rows):
    gpu_value = _parse_system_rows(system_rows).get("GPU", "")
    parts = [part.strip() for part in gpu_value.split(",")]
    if len(parts) >= 3:
        try:
            return float(parts[2]) / 1024
        except ValueError:
            pass
    return None


def _performance_tier(system_rows, use_gpu):
    rows = _parse_system_rows(system_rows)
    cuda = rows.get("Torch CUDA available", "").lower() == "true"
    if not use_gpu or not cuda:
        return "cpu_only"

    vram_gb = _gpu_vram_gb(system_rows)
    if vram_gb is None:
        return "gpu_mid"
    if vram_gb >= 10:
        return "gpu_high"
    if vram_gb >= 6:
        return "gpu_mid"
    return "gpu_low"


def _format_duration_range(min_seconds, max_seconds):
    from app_helpers import format_duration

    if min_seconds <= 0 and max_seconds <= 0:
        return "—"
    low = format_duration(min_seconds)
    high = format_duration(max_seconds)
    if low == high:
        return low
    return f"{low} – {high}"


def _vision_interval_multiplier(interval_seconds):
    interval = max(int(interval_seconds or 30), 5)
    return 30 / interval


def _vision_model_key(config):
    processing = config.get("processing") or {}
    key = processing.get("vision_model", DEFAULT_VISION_MODEL_KEY) or DEFAULT_VISION_MODEL_KEY
    if key in VISION_MODEL_OPTIONS:
        return key
    for option_key, option in VISION_MODEL_OPTIONS.items():
        if option["model_id"] == key:
            return option_key
    return DEFAULT_VISION_MODEL_KEY


def _vision_model_multiplier(config):
    return VISION_MODEL_MULTIPLIERS.get(_vision_model_key(config), 1.0)


def _vision_model_label(config):
    return VISION_MODEL_OPTIONS[_vision_model_key(config)]["label"].split(" (")[0]


def build_system_summary(system_rows, use_gpu):
    rows = _parse_system_rows(system_rows)
    tier = _performance_tier(system_rows, use_gpu)
    tier_labels = {
        "gpu_high": "High-end GPU",
        "gpu_mid": "Mid-range GPU",
        "gpu_low": "Limited GPU VRAM",
        "cpu_only": "CPU only",
    }
    parts = [tier_labels[tier]]
    if rows.get("GPU"):
        parts.append(rows["GPU"].split(",")[0].strip())
    elif rows.get("CPU"):
        parts.append(rows["CPU"])
    cores = rows.get("CPU cores")
    if cores:
        parts.append(f"{cores} cores")
    ram = rows.get("RAM total")
    if ram:
        parts.append(f"{ram} RAM")
    return " · ".join(parts)


def enabled_pipeline_steps(config, *, include_vision=None):
    """Return enabled step keys; scan-after is included when enabled in processing settings."""
    pipeline = config.get("pipeline") or {}
    processing = config.get("processing") or {}
    steps = []
    for key in ("normalize", "transcribe", "vision", "metadata", "index"):
        if not pipeline.get(key, True):
            continue
        if key == "vision" and include_vision is False:
            continue
        steps.append(key)
    if processing.get("scan_after_pipeline", True):
        steps.append("scan")
    return steps


def _estimate_steps(config, system_rows, size_gb, steps):
    processing = config.get("processing") or {}
    use_gpu = bool(processing.get("use_gpu", True))
    tier = _performance_tier(system_rows, use_gpu)
    tier_mult = TIER_MULTIPLIERS[tier]

    model = processing.get("transcription_model", "large-v3")
    model_mult = WHISPER_MODEL_MULTIPLIERS.get(model, 1.0)

    vision_interval_mult = _vision_interval_multiplier(
        processing.get("vision_frame_interval_seconds", 30)
    )
    vision_model_mult = _vision_model_multiplier(config)

    min_total = 0.0
    max_total = 0.0
    for step in steps:
        step_min, step_max = STEP_SEC_PER_GB.get(step, (0, 0))
        step_mult = tier_mult
        if step == "transcribe":
            step_mult *= model_mult
        elif step == "vision":
            step_mult *= vision_interval_mult * vision_model_mult
        min_total += step_min * size_gb * step_mult
        max_total += step_max * size_gb * step_mult

    return int(min_total), int(max_total)


def estimate_library_size(config, system_rows, size_gb, *, include_vision=True):
    steps = enabled_pipeline_steps(config, include_vision=include_vision)
    return _estimate_steps(config, system_rows, size_gb, steps)


def estimate_vision_addon(config, system_rows, size_gb):
    """Extra time from the vision step using current vision settings."""
    return _estimate_steps(config, system_rows, size_gb, ["vision"])


def build_estimate_table(config, system_rows):
    rows = []
    for label, size_gb in LIBRARY_SIZE_LABELS:
        min_without, max_without = estimate_library_size(
            config, system_rows, size_gb, include_vision=False
        )
        min_vision, max_vision = estimate_vision_addon(config, system_rows, size_gb)
        min_with = min_without + min_vision
        max_with = max_without + max_vision

        rows.append(
            {
                "Library size": label,
                "Without vision": _format_duration_range(min_without, max_without),
                "Vision add-on": _format_duration_range(min_vision, max_vision),
                "With vision": _format_duration_range(min_with, max_with),
            }
        )
    return rows


def load_system_rows(system_check_file):
    path = Path(system_check_file)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data.get("system", [])


def _step_summary(config, *, include_vision):
    steps = enabled_pipeline_steps(config, include_vision=include_vision)
    labels = {
        "normalize": "normalize",
        "transcribe": "transcribe",
        "vision": "vision",
        "metadata": "metadata",
        "index": "index",
        "scan": "rescan",
    }
    return ", ".join(labels[key] for key in steps)


def render_pipeline_estimator_ui(config, system_check_file, *, expanded=False):
    import pandas as pd
    import streamlit as st

    with st.expander("Pipeline Run Time Estimator", expanded=expanded):
        st.caption(
            "Rough wall-clock ranges for your selected pipeline steps on this PC. "
            "**Without vision** includes transcribe and any other enabled steps (plus rescan). "
            "**Vision add-on** is the extra time for your selected vision model and frame interval. "
            "**With vision** is the full total. Lowering frame interval greatly increases vision time. "
            "Actual time varies with codecs, clip length, and disk speed."
        )

        estimate_system_rows = load_system_rows(system_check_file)
        if not estimate_system_rows:
            st.info(
                "Run **Refresh System Check** on the Tools/System tab for GPU-aware estimates. "
                "Using CPU-only assumptions until then."
            )
            estimate_system_rows = []

        processing_cfg = config.get("processing") or {}
        vision_selected = bool((config.get("pipeline") or {}).get("vision", True))

        st.write(
            f"**System profile:** {build_system_summary(estimate_system_rows, processing_cfg.get('use_gpu', True))}"
        )
        st.write(
            f"**Without vision:** {_step_summary(config, include_vision=False)} · "
            f"Whisper **{processing_cfg.get('transcription_model', 'large-v3')}** · "
            f"GPU **{'on' if processing_cfg.get('use_gpu', True) else 'off'}**"
        )
        vision_label = _vision_model_label(config)
        vision_interval = processing_cfg.get("vision_frame_interval_seconds", 30)
        if vision_selected:
            st.write(
                f"**With vision:** **{vision_label}** · "
                f"**{vision_interval}s** frame interval."
            )
        else:
            st.write(
                f"Vision is unchecked in Step 2 — **Vision add-on** uses **{vision_label}** at "
                f"**{vision_interval}s** if you enable it."
            )

        estimate_rows = build_estimate_table(config, estimate_system_rows)
        st.dataframe(
            pd.DataFrame(estimate_rows),
            width="stretch",
            hide_index=True,
        )
