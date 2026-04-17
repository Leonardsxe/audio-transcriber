"""
stage_checkpoint.py — Resume state for multi-stage transcription pipelines
==========================================================================

Persists the completed transcription so downstream stages such as diarization
can be retried without re-running Whisper from scratch.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from transcriber.protocols import TranscriptionResult
from transcriber.transcription.result_serialization import dict_to_result, result_to_dict

logger = logging.getLogger(__name__)

_CHECKPOINT_VERSION = 1


class DiarizationStageCheckpoint:
    """
    Persist the transcription result between the transcription and diarization stages.

    The checkpoint is deliberately separate from the chunk-level checkpoint so each
    class has one reason to change.
    """

    def __init__(self, source_path: Path, checkpoint_dir: Path | None = None) -> None:
        self._source = source_path.resolve()
        base = checkpoint_dir or source_path.parent
        base.mkdir(parents=True, exist_ok=True)
        self._path = base / f".{source_path.stem}.diarization.checkpoint.json"

    @property
    def checkpoint_path(self) -> Path:
        """Filesystem path of the stage checkpoint."""
        return self._path

    def load_transcript(self) -> TranscriptionResult | None:
        """
        Restore a previously completed transcription result if still valid.

        Returns ``None`` when no usable checkpoint exists.
        """
        if not self._path.exists():
            return None

        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Stage checkpoint unreadable (%s) — starting fresh.", exc)
            return None

        if data.get("version") != _CHECKPOINT_VERSION:
            logger.warning("Stage checkpoint version mismatch — starting fresh.")
            return None

        saved_size = data.get("source_size_bytes", -1)
        actual_size = self._source.stat().st_size
        if saved_size != actual_size:
            logger.warning(
                "Source file size changed (saved %d, now %d) — starting fresh.",
                saved_size,
                actual_size,
            )
            return None

        transcript_data = data.get("transcript")
        if not isinstance(transcript_data, dict):
            logger.warning("Stage checkpoint missing transcript data — starting fresh.")
            return None

        transcript = dict_to_result(transcript_data)
        logger.info(
            "Stage checkpoint loaded — reusing completed transcription from '%s'.",
            self._path,
        )
        return transcript

    def save_transcript(self, transcript: TranscriptionResult) -> None:
        """Persist the completed transcription result atomically."""
        now = datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
        payload = {
            "version": _CHECKPOINT_VERSION,
            "source": str(self._source),
            "source_size_bytes": self._source.stat().st_size,
            "created_at": now,
            "updated_at": now,
            "transcript": result_to_dict(transcript),
        }
        tmp_path = self._path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp_path, self._path)

    def delete(self) -> None:
        """Remove the stage checkpoint after the full pipeline succeeds."""
        try:
            self._path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete stage checkpoint: %s", exc)
