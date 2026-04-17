"""
protocols.py — Domain objects and structural interfaces
=======================================================

Dependency-Inversion Principle: all concrete implementations depend on
these abstractions, never on each other.

Value objects
-------------
- ``TranscriptionSegment``    — one timed utterance from Whisper
- ``TranscriptionResult``     — full transcript of one audio file
- ``SpeakerTurn``             — a block of speech attributed to one speaker
- ``DiarizedTranscript``      — full transcript with speaker-labelled turns

Protocols (interfaces)
----------------------
- ``SpeechTranscriber``  — transcribe(path) → TranscriptionResult
- ``ResultExporter``     — export(result, dest) → Path

Exceptions
----------
- ``HaltException``  — raised when the user halts a chunked job;
  distinguishes a deliberate pause from an engine crash so the batch
  loop can treat them differently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable


# ─────────────────────────────────────────────
#  Custom exceptions
# ─────────────────────────────────────────────


class HaltException(RuntimeError):
    """
    Raised when the user deliberately pauses a chunked transcription job.

    Distinct from a plain ``RuntimeError`` so ``transcribe_batch`` can
    stop the entire batch (not just skip the current file) while still
    letting ordinary engine errors be skipped.
    """


# ─────────────────────────────────────────────
#  Core transcription value objects
# ─────────────────────────────────────────────


@dataclass(frozen=True)
class TranscriptionSegment:
    """
    One timed chunk of transcribed speech produced by Whisper.

    Attributes
    ----------
    start:
        Segment start time in seconds (absolute, within the original file).
    end:
        Segment end time in seconds.
    text:
        Recognised text for this segment.
    confidence:
        Probability in [0.0, 1.0] derived from the model's avg_logprob.
    speaker:
        Optional speaker label assigned by diarization (e.g. ``"INTERVIEWER"``).
        ``None`` when diarization has not been run.
    """

    start: float
    end: float
    text: str
    confidence: float
    speaker: str | None = field(default=None)

    @property
    def duration(self) -> float:
        """Segment length in seconds."""
        return self.end - self.start

    def __str__(self) -> str:
        prefix = f"[{self.speaker}] " if self.speaker else ""
        return (
            f"{prefix}[{self.start:.1f}s → {self.end:.1f}s]"
            f" ({self.confidence:.0%}) {self.text.strip()}"
        )


@dataclass(frozen=True)
class TranscriptionResult:
    """
    Complete transcription of one audio file.

    Attributes
    ----------
    segments:
        Ordered list of timed transcript segments.
    language:
        ISO 639-1 code detected / used by the model (e.g. ``"es"``).
    duration:
        Total audio duration in seconds.
    source_path:
        Optional path to the originating audio file (for traceability).
    """

    segments: list[TranscriptionSegment]
    language: str
    duration: float
    source_path: Path | None = field(default=None)

    @property
    def full_text(self) -> str:
        """Concatenate all segment texts into a single paragraph."""
        return " ".join(seg.text.strip() for seg in self.segments)

    @property
    def average_confidence(self) -> float:
        """Mean confidence across all segments (0.0 if no segments)."""
        if not self.segments:
            return 0.0
        return sum(s.confidence for s in self.segments) / len(self.segments)

    def __str__(self) -> str:
        return (
            f"TranscriptionResult("
            f"language={self.language}, "
            f"duration={self.duration:.1f}s, "
            f"segments={len(self.segments)}, "
            f"avg_confidence={self.average_confidence:.0%})"
        )


# ─────────────────────────────────────────────
#  Diarized transcript value objects
# ─────────────────────────────────────────────


@dataclass(frozen=True)
class SpeakerTurn:
    """
    A contiguous block of speech attributed to one speaker.

    Multiple ``TranscriptionSegment`` objects that share the same speaker
    label and are temporally adjacent are merged into a single turn for
    readability and downstream processing.

    Attributes
    ----------
    speaker:
        Human-readable label, e.g. ``"INTERVIEWER"`` or ``"INTERVIEWEE"``.
    start:
        Turn start time in seconds.
    end:
        Turn end time in seconds.
    text:
        Full text of the turn (segments joined with spaces).
    segments:
        The individual Whisper segments that make up this turn.
    """

    speaker: str
    start: float
    end: float
    text: str
    segments: list[TranscriptionSegment]

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def __str__(self) -> str:
        return f"[{self.speaker}] {self.start:.1f}s–{self.end:.1f}s: {self.text[:80]}"


@dataclass(frozen=True)
class SpeakerStats:
    """
    Aggregate speaking statistics for one speaker.

    Attributes
    ----------
    label:
        Speaker identifier (``"INTERVIEWER"`` / ``"INTERVIEWEE"``).
    total_speech_s:
        Total seconds of speech attributed to this speaker.
    turn_count:
        Number of conversational turns.
    word_count:
        Approximate word count across all turns.
    """

    label: str
    total_speech_s: float
    turn_count: int
    word_count: int

    @property
    def speech_ratio(self) -> float:
        """Fraction of total identified speech belonging to this speaker."""
        return self.total_speech_s  # caller normalises against total


@dataclass(frozen=True)
class DiarizedTranscript:
    """
    Full interview transcript with speaker attribution.

    This is the primary output object for downstream auto-coding and
    thematic synthesis pipelines.  It contains:

    - All speaker turns in chronological order.
    - Per-speaker statistics.
    - The underlying ``TranscriptionResult`` for segment-level access.

    Attributes
    ----------
    turns:
        Speaker turns in order — the main unit for auto-coding.
    speakers:
        Dict mapping label → ``SpeakerStats``.
    source:
        The underlying transcription (with speaker labels on segments).
    """

    turns: list[SpeakerTurn]
    speakers: dict[str, SpeakerStats]
    source: TranscriptionResult

    @property
    def interviewer_turns(self) -> list[SpeakerTurn]:
        return [t for t in self.turns if t.speaker == "INTERVIEWER"]

    @property
    def interviewee_turns(self) -> list[SpeakerTurn]:
        return [t for t in self.turns if t.speaker == "INTERVIEWEE"]

    @property
    def full_text(self) -> str:
        """Plain text with inline speaker labels — for quick review."""
        lines = [f"[{t.speaker}] {t.text.strip()}" for t in self.turns]
        return "\n\n".join(lines)

    def __str__(self) -> str:
        stats = ", ".join(
            f"{lbl}: {s.turn_count} turns / {s.total_speech_s:.0f}s"
            for lbl, s in self.speakers.items()
        )
        return f"DiarizedTranscript({stats})"


# ─────────────────────────────────────────────
#  Interfaces / Protocols
# ─────────────────────────────────────────────


@runtime_checkable
class SpeechTranscriber(Protocol):
    """Contract for any transcription engine."""

    def transcribe(self, audio_path: Path) -> TranscriptionResult: ...


@runtime_checkable
class ResultExporter(Protocol):
    """Contract for writing a ``TranscriptionResult`` to disk."""

    def export(self, result: TranscriptionResult, destination: Path) -> Path: ...
