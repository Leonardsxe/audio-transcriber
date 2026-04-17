"""
logging_setup.py — Centralised logging configuration
=====================================================

Call :func:`configure_logging` once at the application entry-point
(``main.py`` or CLI).  All loggers in the ``transcriber`` hierarchy
inherit this configuration automatically.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """
    Configure root-level logging for the transcriber package.

    Parameters
    ----------
    level:
        One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``.
        Defaults to ``"INFO"``.
    """
    numeric = getattr(logging, level.upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(numeric)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root = logging.getLogger()
    root.setLevel(numeric)
    root.handlers.clear()
    root.addHandler(handler)

    # Silence noisy third-party loggers unless in DEBUG mode.
    if numeric > logging.DEBUG:
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)
