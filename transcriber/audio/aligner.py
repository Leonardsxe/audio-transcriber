"""
aligner.py — Align Whisper segments with diarization output
============================================================

Whisper produces text segments with timestamps.
Pyannote produces speaker segments with timestamps.
These two are *independent* — they don't share boundaries.

This module fuses them by computing temporal overlap: for each Whisper
segment we find the diarization segment that covers the *majority* of
its time span and assign that speaker label.

Then adjacent segments sharing the same speaker are merged into
``SpeakerTurn`` objects, which become the primary unit for downstream
auto-coding and thematic synthesis.

Overlap strategy
----------------
For each Whisper segment [ws, we]:
  1. Find all diarization segments that overlap it.
  2. Pick the one with the *most overlap* (seconds of intersection).
  3. Assign that speaker to the segment.
  4. If no overlap exists (gap in diarization), carry forward the last
     seen speaker to avoid orphaned segments.

This is more robust than a simple midpoint lookup — it handles the case
where a Whisper segment spans a speaker boundary (both speakers partially
overlap it) and picks the dominant one.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from transcriber.audio.diarizer import DiarizationSegment
from transcriber.protocols import (
    DiarizedTranscript,
    SpeakerStats,
    SpeakerTurn,
    TranscriptionResult,
    TranscriptionSegment,
)

logger = logging.getLogger(__name__)

_FALLBACK_SPEAKER = "UNKNOWN"


def align(
    transcript: TranscriptionResult,
    diarization: list[DiarizationSegment],
) -> DiarizedTranscript:
    """
    Fuse a ``TranscriptionResult`` with diarization segments.

    Parameters
    ----------
    transcript:
        Whisper output with timed text segments (no speaker info yet).
    diarization:
        Ordered list of speaker segments from ``SpeakerDiarizer.diarize()``.

    Returns
    -------
    DiarizedTranscript
        Full transcript with per-segment speaker labels and grouped turns.

    Notes
    -----
    The function is pure (no I/O, no side effects) and therefore trivially
    testable with synthetic data.
    """
    if not diarization:
        logger.warning("Diarization is empty — all segments labelled as UNKNOWN.")
        labelled = [_label(seg, _FALLBACK_SPEAKER) for seg in transcript.segments]
        return _build_transcript(labelled, transcript)

    labelled_segments = _assign_speakers(transcript.segments, diarization)
    return _build_transcript(labelled_segments, transcript)


# ─────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────


def _assign_speakers(
    segments: list[TranscriptionSegment],
    diarization: list[DiarizationSegment],
) -> list[TranscriptionSegment]:
    """Label each Whisper segment with the dominant diarization speaker."""
    labelled: list[TranscriptionSegment] = []
    last_speaker: str = _FALLBACK_SPEAKER

    for seg in segments:
        speaker = _dominant_speaker(seg.start, seg.end, diarization) or last_speaker
        last_speaker = speaker
        labelled.append(_label(seg, speaker))

    return labelled


def _dominant_speaker(
    start: float,
    end: float,
    diarization: list[DiarizationSegment],
) -> str | None:
    """
    Return the speaker with the most overlap with the window [start, end].

    Returns ``None`` if there is no overlap at all (gap in diarization).
    """
    overlap_by_speaker: dict[str, float] = {}

    for dseg in diarization:
        # Intersection of [start, end] and [dseg.start, dseg.end]
        overlap_start = max(start, dseg.start)
        overlap_end = min(end, dseg.end)
        overlap = overlap_end - overlap_start

        if overlap > 0:
            overlap_by_speaker[dseg.speaker] = (
                overlap_by_speaker.get(dseg.speaker, 0.0) + overlap
            )

    if not overlap_by_speaker:
        return None

    return max(overlap_by_speaker, key=lambda k: overlap_by_speaker[k])


def _label(seg: TranscriptionSegment, speaker: str) -> TranscriptionSegment:
    """Return a new segment with the speaker field set (frozen dataclass)."""
    return TranscriptionSegment(
        start=seg.start,
        end=seg.end,
        text=seg.text,
        confidence=seg.confidence,
        speaker=speaker,
    )


def _build_transcript(
    labelled: list[TranscriptionSegment],
    source: TranscriptionResult,
) -> DiarizedTranscript:
    """
    Group labelled segments into speaker turns and compute statistics.

    Consecutive segments with the same speaker are merged into one turn.
    A new turn starts whenever the speaker changes.
    """
    # Rebuild TranscriptionResult with speaker-labelled segments.
    labelled_result = TranscriptionResult(
        segments=labelled,
        language=source.language,
        duration=source.duration,
        source_path=source.source_path,
    )

    turns = _group_into_turns(labelled)
    stats = _compute_stats(turns)

    _log_balance(stats, source.duration)

    return DiarizedTranscript(
        turns=turns,
        speakers=stats,
        source=labelled_result,
    )


def _group_into_turns(segments: list[TranscriptionSegment]) -> list[SpeakerTurn]:
    """Merge adjacent same-speaker segments into turns."""
    if not segments:
        return []

    turns: list[SpeakerTurn] = []
    current_segs: list[TranscriptionSegment] = [segments[0]]

    for seg in segments[1:]:
        if seg.speaker == current_segs[-1].speaker:
            current_segs.append(seg)
        else:
            turns.append(_make_turn(current_segs))
            current_segs = [seg]

    turns.append(_make_turn(current_segs))
    return turns


def _make_turn(segments: list[TranscriptionSegment]) -> SpeakerTurn:
    """Collapse a list of same-speaker segments into one SpeakerTurn."""
    text = " ".join(s.text.strip() for s in segments)
    return SpeakerTurn(
        speaker=segments[0].speaker or _FALLBACK_SPEAKER,
        start=segments[0].start,
        end=segments[-1].end,
        text=text,
        segments=segments,
    )


def _compute_stats(turns: list[SpeakerTurn]) -> dict[str, SpeakerStats]:
    """Aggregate per-speaker statistics from the turn list."""
    raw: dict[str, dict[str, float | int]] = {}

    for turn in turns:
        entry = raw.setdefault(turn.speaker, {"speech_s": 0.0, "turns": 0, "words": 0})
        entry["speech_s"] = float(entry["speech_s"]) + turn.duration
        entry["turns"] = int(entry["turns"]) + 1
        entry["words"] = int(entry["words"]) + turn.word_count

    return {
        label: SpeakerStats(
            label=label,
            total_speech_s=float(data["speech_s"]),
            turn_count=int(data["turns"]),
            word_count=int(data["words"]),
        )
        for label, data in raw.items()
    }


def _log_balance(stats: dict[str, SpeakerStats], total_duration: float) -> None:
    """Log a human-readable speaking balance summary."""
    total_speech = sum(s.total_speech_s for s in stats.values()) or 1.0
    for lbl, s in stats.items():
        pct = s.total_speech_s / total_speech * 100
        logger.info(
            "  %s — %.0f s speech (%.0f%%), %d turns, ~%d words",
            lbl, s.total_speech_s, pct, s.turn_count, s.word_count,
        )
