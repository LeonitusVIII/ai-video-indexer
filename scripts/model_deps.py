"""Check Python packages required for the vision model stack."""
import importlib.util

VISION_PACKAGES = {
    "torch": "torch",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "qwen-vl-utils": "qwen_vl_utils",
}


def missing_packages(package_map):
    missing = []
    for label, import_name in package_map.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(label)
    return missing


def missing_vision_packages():
    return missing_packages(VISION_PACKAGES)


def vision_stack_ready():
    return not missing_vision_packages()
