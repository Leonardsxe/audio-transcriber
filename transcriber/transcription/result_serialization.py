"""
result_serialization.py — Shared serialisation helpers for TranscriptionResult
=============================================================================

Keeps JSON persistence concerns out of the service layer and lets multiple
checkpoint managers reuse the same conversion logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from transcriber.protocols import TranscriptionResult, TranscriptionSegment


def result_to_dict(result: TranscriptionResult) -> dict[str, Any]:
    """Convert a ``TranscriptionResult`` into JSON-safe primitives."""
    return {
        "language": result.language,
        "duration": result.duration,
        "source_path": str(result.source_path) if result.source_path else None,
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "confidence": seg.confidence,
                "speaker": seg.speaker,
            }
            for seg in result.segments
        ],
    }


def dict_to_result(data: dict[str, Any]) -> TranscriptionResult:
    """Rebuild a ``TranscriptionResult`` from persisted JSON data."""
    segments = [
        TranscriptionSegment(
            start=segment["start"],
            end=segment["end"],
            text=segment["text"],
            confidence=segment["confidence"],
            speaker=segment.get("speaker"),
        )
        for segment in data.get("segments", [])
    ]
    source_path = data.get("source_path")
    return TranscriptionResult(
        segments=segments,
        language=data.get("language", "es"),
        duration=data.get("duration", 0.0),
        source_path=Path(source_path) if source_path else None,
    )
