"""
halt.py — Graceful halt controller (Ctrl+C / SIGTERM / 'q' keystroke)
======================================================================

Provides a single shared :class:`HaltController` that coordinates a clean
stop across the transcription loop.

How it works
------------
- Registers handlers for ``SIGINT`` (Ctrl+C) and ``SIGTERM`` (kill / systemd).
- Sets a threading ``Event`` when either signal is received.
- The transcription loop polls ``.should_halt()`` before starting each new
  chunk.  The *current* chunk is always allowed to finish.
- Pressing **q + Enter** in the terminal also triggers a halt (via an
  optional background reader thread).

This design avoids ``sys.exit()`` inside the signal handler — which can
leave files in an inconsistent state — and lets the caller run any cleanup
(checkpoint flush, chunk deletion) before exiting.

Usage
-----
::

    controller = HaltController()
    controller.install()            # register SIGINT / SIGTERM handlers

    for chunk in chunks:
        if controller.should_halt():
            break
        result = engine.transcribe(chunk.path)
        checkpoint.save_chunk(chunk, result)

    if controller.was_halted():
        print("Paused. Re-run the same command to resume.")
    else:
        checkpoint.delete()
"""

from __future__ import annotations

import logging
import signal
import sys
import threading
from types import FrameType

logger = logging.getLogger(__name__)


class HaltController:
    """
    Thread-safe stop flag with signal and keyboard support.

    Parameters
    ----------
    prompt_keyboard:
        If True, start a background thread that reads stdin and halts when
        the user types ``q`` and presses Enter.  Disable in non-interactive
        environments (e.g. when piping stdin).

    Attributes
    ----------
    halt_reason:
        Human-readable string describing why the halt was triggered, or
        empty string if no halt occurred.
    """

    def __init__(self, *, prompt_keyboard: bool = True) -> None:
        self._event = threading.Event()
        self._prompt_keyboard = prompt_keyboard
        self._halt_reason = ""
        self._original_sigint: signal.Handlers = signal.SIG_DFL
        self._original_sigterm: signal.Handlers = signal.SIG_DFL
        self._keyboard_thread: threading.Thread | None = None

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def halt_reason(self) -> str:
        return self._halt_reason

    def install(self) -> None:
        """
        Register signal handlers and (optionally) the keyboard thread.

        Must be called from the main thread.
        """
        self._original_sigint = signal.signal(signal.SIGINT, self._handle_signal)
        self._original_sigterm = signal.signal(signal.SIGTERM, self._handle_signal)

        if self._prompt_keyboard and sys.stdin.isatty():
            self._keyboard_thread = threading.Thread(
                target=self._keyboard_reader,
                daemon=True,
                name="halt-keyboard-reader",
            )
            self._keyboard_thread.start()
            logger.info(
                "Press Ctrl+C or type 'q' + Enter to pause after the current chunk."
            )

    def uninstall(self) -> None:
        """Restore the original signal handlers."""
        signal.signal(signal.SIGINT, self._original_sigint)
        signal.signal(signal.SIGTERM, self._original_sigterm)

    def should_halt(self) -> bool:
        """Return True if a halt was requested (check before each new chunk)."""
        return self._event.is_set()

    def was_halted(self) -> bool:
        """Alias for ``should_halt()`` — reads better at the end of a loop."""
        return self._event.is_set()

    def trigger(self, reason: str = "manual") -> None:
        """Programmatically trigger a halt (useful in tests)."""
        self._halt_reason = reason
        self._event.set()

    # ── signal handler ────────────────────────────────────────────────────────

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        sig_name = "Ctrl+C" if signum == signal.SIGINT else "SIGTERM"
        if not self._event.is_set():
            self._halt_reason = sig_name
            self._event.set()
            # Print on its own line so it doesn't collide with the progress log.
            print(
                f"\n[halt] {sig_name} received — finishing current chunk then pausing …",
                flush=True,
            )
        else:
            # Second signal: exit immediately (user is impatient).
            print("\n[halt] Second signal — forcing exit.", flush=True)
            sys.exit(1)

    # ── keyboard reader ───────────────────────────────────────────────────────

    def _keyboard_reader(self) -> None:
        """
        Background thread: read stdin lines and halt on 'q'.

        Runs as a daemon thread — automatically dies when the main thread
        exits, so it never needs explicit cleanup.
        """
        try:
            for line in sys.stdin:
                if line.strip().lower() in {"q", "quit", "stop", "pause"}:
                    if not self._event.is_set():
                        self._halt_reason = "keyboard 'q'"
                        self._event.set()
                        print(
                            "\n[halt] Pause requested — finishing current chunk …",
                            flush=True,
                        )
                    break
        except (OSError, EOFError):
            pass  # stdin closed — nothing to do
