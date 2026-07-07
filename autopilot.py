"""Cron entry point: render one autopilot video and exit.

Usage:
    .venv/bin/python autopilot.py [--niche "..."] [--language vi]

Exit code 0 when a video was rendered, 1 otherwise — so cron mail / wrappers
can tell success from failure. All state (product history, log) lives in
storage/autopilot/; per-run outputs land in the normal storage/tasks/<id>/
folder next to an autopilot_report.txt with the publish copy.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from loguru import logger

from app.services import autopilot


def main() -> int:
    parser = argparse.ArgumentParser(description="Unattended affiliate video run")
    parser.add_argument("--niche", default="", help="override config channel_niche")
    parser.add_argument("--language", default="vi")
    args = parser.parse_args()

    logger.add(
        os.path.join(autopilot.autopilot_dir(), "autopilot.log"),
        rotation="10 MB",
        retention=5,
        encoding="utf-8",
    )

    result = autopilot.run_autopilot(niche=args.niche, language=args.language)
    if result.get("error"):
        logger.error(f"autopilot failed: {result['error']}")
        return 1
    logger.success(
        f"autopilot done: {result['product']!r} -> {result['videos']} "
        f"(report: {result['report']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
