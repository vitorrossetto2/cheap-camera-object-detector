from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def main() -> None:
    _configure_logging(DEFAULT_LOG_LEVEL)
    logger.info("application_start mode=ui log_level=%s", DEFAULT_LOG_LEVEL)

    from cheap_camera_object_detector.ui import run_desktop_ui

    run_desktop_ui()


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
        force=True,
    )


if __name__ == "__main__":
    main()
