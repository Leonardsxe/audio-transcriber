"""
checkpoint.py — Persistent checkpoint manager for resumable transcription
=========================================================================

Saves progress after every completed chunk so a long transcription job can
be interrupted (Ctrl+C or SIGTERM) and resumed exactly where it stopped —
without re-transcribing chunks that already finished.

Checkpoint file format (JSON)
------------------------------
One file per source audio, stored alongside the output or in a configurable
directory.  Example::

    {
      "version": 1,
      "source": "/abs/path/to/interview.mp3",
      "source_size_bytes": 2831155200,
      "total_chunks": 15,
      "completed_chunks": [0, 1, 2],
      "created_at": "2026-04-06T15:30:00",
      "updated_at": "2026-04-06T15:47:23",
      "chunk_results": {
        "0": { "segments": [...], "language": "es", "duration": 612.4 },
        "1": { ...}
      }
    }

The ``source_size_bytes`` field is a basic integrity guard: if the file on
disk has a different size than when the checkpoint was created, we warn and
offer to start fresh rather than silently mixing results from different files.

Design
------
- ``CheckpointManager`` is a plain class with no threading — safe to call
  from a signal handler as long as the write is atomic (write to .tmp then
  rename).
- Atomic rename guarantees we never leave a half-written checkpoint.
- If the checkpoint file is corrupted (bad JSON), we log a warning and
  treat it as absent rather than crashing.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from transcriber.audio.chunker import AudioChunk
from transcriber.protocols import TranscriptionResult
from transcriber.transcription.result_serialization import dict_to_result, result_to_dict

logger = logging.getLogger(__name__)

_CHECKPOINT_VERSION = 1


class CheckpointManager:
    """
    Saves and loads per-chunk transcription progress.

    Parameters
    ----------
    source_path:
        The original audio file being transcribed.
    checkpoint_dir:
        Directory where the ``.checkpoint.json`` file is stored.
        Defaults to the same directory as ``source_path``.

    Example
    -------
    >>> mgr = CheckpointManager(Path("audio/interview.mp3"))
    >>> mgr.load()                          # loads existing progress if any
    >>> for chunk in chunks:
    ...     if mgr.is_done(chunk): continue
    ...     result = engine.transcribe(chunk.path)
    ...     mgr.save_chunk(chunk, result)
    >>> mgr.delete()                        # clean up after success
    """

    def __init__(
        self,
        source_path: Path,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self._source = source_path.resolve()
        base = checkpoint_dir or source_path.parent
        base.mkdir(parents=True, exist_ok=True)
        self._path = base / f".{source_path.stem}.checkpoint.json"

        # In-memory state
        self._total_chunks: int = 0
        self._completed: set[int] = set()
        self._chunk_results: dict[int, TranscriptionResult] = {}
        self._created_at: str = ""
        self._loaded = False

    # ── public interface ──────────────────────────────────────────────────────

    @property
    def checkpoint_path(self) -> Path:
        """Filesystem path of the checkpoint file."""
        return self._path

    @property
    def completed_count(self) -> int:
        """Number of chunks successfully transcribed so far."""
        return len(self._completed)

    @property
    def is_fresh(self) -> bool:
        """True if no checkpoint exists (this is a new job)."""
        return not self._path.exists()

    def load(self, total_chunks: int) -> bool:
        """
        Load an existing checkpoint if present and valid.

        Parameters
        ----------
        total_chunks:
            The total number of chunks in the current split.  Used to
            detect if the audio was re-split differently since the last run.

        Returns
        -------
        bool
            True if a valid checkpoint was loaded, False if starting fresh.
        """
        self._total_chunks = total_chunks

        if not self._path.exists():
            logger.info("No checkpoint found — starting fresh.")
            return False

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Checkpoint file unreadable (%s) — starting fresh.", exc)
            return False

        # Integrity checks
        if data.get("version") != _CHECKPOINT_VERSION:
            logger.warning("Checkpoint version mismatch — starting fresh.")
            return False

        saved_size = data.get("source_size_bytes", -1)
        actual_size = self._source.stat().st_size
        if saved_size != actual_size:
            logger.warning(
                "Source file size changed (saved %d, now %d) — starting fresh.",
                saved_size,
                actual_size,
            )
            return False

        saved_total = data.get("total_chunks", 0)
        if saved_total != total_chunks:
            logger.warning(
                "Chunk count changed (saved %d, now %d) — starting fresh.",
                saved_total,
                total_chunks,
            )
            return False

        # Restore state
        self._completed = set(data.get("completed_chunks", []))
        self._created_at = data.get("created_at", "")
        raw_results = data.get("chunk_results", {})
        self._chunk_results = {
            int(k): _dict_to_result(v) for k, v in raw_results.items()
        }
        self._loaded = True

        logger.info(
            "Checkpoint loaded: %d/%d chunks already done (created %s).",
            len(self._completed),
            total_chunks,
            self._created_at,
        )
        return True

    def is_done(self, chunk: AudioChunk) -> bool:
        """Return True if *chunk* was already transcribed in a previous run."""
        return chunk.index in self._completed

    def save_chunk(self, chunk: AudioChunk, result: TranscriptionResult) -> None:
        """
        Persist the result for *chunk* and flush to disk atomically.

        Safe to call from a signal handler — uses write-then-rename so the
        checkpoint file is never left in a partial state.
        """
        self._completed.add(chunk.index)
        self._chunk_results[chunk.index] = result
        self._flush()
        logger.debug("Checkpoint saved: %d/%d done.", len(self._completed), self._total_chunks)

    def get_result(self, chunk: AudioChunk) -> TranscriptionResult | None:
        """Return the cached result for *chunk*, or None if not yet done."""
        return self._chunk_results.get(chunk.index)

    def all_results_in_order(self, chunks: list[AudioChunk]) -> list[TranscriptionResult]:
        """
        Return results for all *chunks* in index order.

        Raises
        ------
        RuntimeError
            If any chunk has no cached result (only possible if called before
            all chunks are marked done).
        """
        results = []
        for chunk in chunks:
            result = self._chunk_results.get(chunk.index)
            if result is None:
                raise RuntimeError(
                    f"Chunk {chunk.index} has no cached result. "
                    "Did you call save_chunk() for every completed chunk?"
                )
            results.append(result)
        return results

    def delete(self) -> None:
        """Remove the checkpoint file after a successful run."""
        try:
            self._path.unlink(missing_ok=True)
            logger.debug("Checkpoint deleted: %s", self._path)
        except OSError as exc:
            logger.warning("Could not delete checkpoint: %s", exc)

    def summary(self) -> str:
        """Human-readable progress line."""
        return (
            f"Progress: {len(self._completed)}/{self._total_chunks} chunks "
            f"({'%.0f' % (100 * len(self._completed) / max(self._total_chunks, 1))}%)"
        )

    # ── private ───────────────────────────────────────────────────────────────

    def _flush(self) -> None:
        """Write checkpoint to a temp file then atomically rename it."""
        now = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")

        payload: dict[str, Any] = {
            "version": _CHECKPOINT_VERSION,
            "source": str(self._source),
            "source_size_bytes": self._source.stat().st_size,
            "total_chunks": self._total_chunks,
            "completed_chunks": sorted(self._completed),
            "created_at": self._created_at or now,
            "updated_at": now,
            "chunk_results": {
                str(idx): _result_to_dict(res)
                for idx, res in self._chunk_results.items()
            },
        }
        if not self._created_at:
            self._created_at = now

        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self._path)  # atomic on POSIX


def _result_to_dict(result: TranscriptionResult) -> dict[str, Any]:
    """Compatibility wrapper for existing tests and callers."""
    return result_to_dict(result)


def _dict_to_result(data: dict[str, Any]) -> TranscriptionResult:
    """Compatibility wrapper for existing tests and callers."""
    return dict_to_result(data)
