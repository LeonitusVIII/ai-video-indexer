"""CLI entry for Windows Task Scheduler — checks overnight schedule rules."""
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from schedule_manager import tick_schedule


def main():
    result = tick_schedule()
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
