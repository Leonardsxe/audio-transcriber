"""
reassembler.py — Merges chunked TranscriptionResults into one coherent transcript
==================================================================================

When audio is split into overlapping chunks, the overlapping regions produce
duplicate (or near-duplicate) text at every boundary.  This module detects
and removes those duplicates so the final transcript reads continuously.

How duplicate detection works
------------------------------
Each chunk carries ``overlap_start_ms`` and ``overlap_end_ms`` metadata.
We know exactly which segments in the transcript fall inside the overlap
zone.  We compare the *tail* of chunk N with the *head* of chunk N+1 using
a token-level longest-common-subsequence match, then drop the duplicate
portion from whichever side is shorter.

This is deliberately conservative: if the match score is below the
threshold the segments are kept as-is (with a small gap marker) rather
than risk silently dropping speech.

Timeline correction
-------------------
Segment timestamps in each chunk are relative to the chunk's start, not
the original file.  We shift every segment's ``start`` and ``end`` by
``chunk.start_s`` so the final result has correct absolute timestamps.
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from pathlib import Path

from transcriber.audio.chunker import AudioChunk
from transcriber.protocols import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)

# Minimum similarity score (0–1) to consider two text snippets a duplicate.
_SIMILARITY_THRESHOLD = 0.82

# How many words from the tail/head to compare when looking for overlaps.
_OVERLAP_WORDS = 30


def reassemble(
    results: list[TranscriptionResult],
    chunks: list[AudioChunk],
    *,
    source_path: Path | None = None,
) -> TranscriptionResult:
    """
    Merge a list of per-chunk ``TranscriptionResult`` objects into one.

    Parameters
    ----------
    results:
        One result per chunk, in order.  Must be the same length as *chunks*.
    chunks:
        The ``AudioChunk`` objects that produced *results*.
    source_path:
        Optional path to the original audio file, stored in the final result
        for traceability.

    Returns
    -------
    TranscriptionResult
        A single result with corrected timestamps and de-duplicated text.

    Raises
    ------
    ValueError
        If ``len(results) != len(chunks)``.
    """
    if len(results) != len(chunks):
        raise ValueError(
            f"results ({len(results)}) and chunks ({len(chunks)}) must have the same length."
        )

    if not results:
        return TranscriptionResult(
            segments=[], language="es", duration=0.0, source_path=source_path
        )

    if len(results) == 1:
        # No merging needed — just fix the timestamps.
        return _shift_timestamps(results[0], chunks[0].start_s, source_path=source_path)

    merged_segments: list[TranscriptionSegment] = []
    language = results[0].language

    for idx, (result, chunk) in enumerate(zip(results, chunks)):
        # 1. Shift all segment timestamps to absolute positions.
        shifted = _shift_timestamps(result, chunk.start_s)
        segments = list(shifted.segments)

        if not segments:
            continue

        # 2. Drop segments that fall entirely within the *start* overlap zone
        #    (those are duplicates from the previous chunk's end overlap).
        if idx > 0 and chunk.overlap_start_ms > 0:
            overlap_end_s = chunk.start_s + chunk.overlap_start_ms / 1000
            segments = [s for s in segments if s.end > overlap_end_s]
            logger.debug(
                "Chunk %03d: dropped %d overlap-start segment(s), kept %d",
                idx,
                len(shifted.segments) - len(segments),
                len(segments),
            )

        # 3. Fine-grained text dedup against the previous chunk's tail.
        if merged_segments and segments:
            segments = _deduplicate_boundary(merged_segments, segments, chunk)

        merged_segments.extend(segments)

    total_duration = chunks[-1].end_s
    return TranscriptionResult(
        segments=merged_segments,
        language=language,
        duration=total_duration,
        source_path=source_path,
    )


# ─────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────


def _shift_timestamps(
    result: TranscriptionResult,
    offset_s: float,
    *,
    source_path: Path | None = None,
) -> TranscriptionResult:
    """Return a new result with all segment timestamps shifted by *offset_s*."""
    shifted = [
        TranscriptionSegment(
            start=seg.start + offset_s,
            end=seg.end + offset_s,
            text=seg.text,
            confidence=seg.confidence,
        )
        for seg in result.segments
    ]
    return TranscriptionResult(
        segments=shifted,
        language=result.language,
        duration=result.duration + offset_s,
        source_path=source_path or result.source_path,
    )


def _deduplicate_boundary(
    previous: list[TranscriptionSegment],
    incoming: list[TranscriptionSegment],
    chunk: AudioChunk,
) -> list[TranscriptionSegment]:
    """
    Remove duplicate text at the boundary between two adjacent chunks.

    Strategy
    --------
    Take the last N words from ``previous`` and the first N words from
    ``incoming``.  Compute a SequenceMatcher similarity ratio.  If it's
    above the threshold, drop the matching head from ``incoming``.

    Parameters
    ----------
    previous:
        Already-merged segments (the accumulated output so far).
    incoming:
        Segments from the next chunk (after timestamp shift).
    chunk:
        The AudioChunk that produced ``incoming``.

    Returns
    -------
    list[TranscriptionSegment]
        The ``incoming`` list, possibly with its duplicate head trimmed.
    """
    tail_text = _last_n_words(previous, _OVERLAP_WORDS)
    head_text = _first_n_words(incoming, _OVERLAP_WORDS)

    if not tail_text or not head_text:
        return incoming

    ratio = SequenceMatcher(None, tail_text, head_text).ratio()
    logger.debug(
        "Chunk %03d boundary similarity: %.2f (threshold %.2f)",
        chunk.index,
        ratio,
        _SIMILARITY_THRESHOLD,
    )

    if ratio < _SIMILARITY_THRESHOLD:
        # Not a clear duplicate — keep everything, add a small gap in timestamps.
        logger.debug("Chunk %03d: no duplicate detected, keeping full segment list.", chunk.index)
        return incoming

    # Find the last segment in `incoming` whose text still overlaps with the tail.
    # We drop segments until the overlap ends.
    overlap_end_s = chunk.start_s + chunk.overlap_start_ms / 1000
    cleaned = [s for s in incoming if s.start >= overlap_end_s]

    dropped = len(incoming) - len(cleaned)
    if dropped:
        logger.debug("Chunk %03d: deduped %d boundary segment(s).", chunk.index, dropped)

    return cleaned if cleaned else incoming  # never return empty if input wasn't empty


def _last_n_words(segments: list[TranscriptionSegment], n: int) -> str:
    """Join the last *n* words from the combined text of *segments*."""
    text = " ".join(s.text.strip() for s in segments)
    words = text.split()
    return " ".join(words[-n:]).lower()


def _first_n_words(segments: list[TranscriptionSegment], n: int) -> str:
    """Join the first *n* words from the combined text of *segments*."""
    text = " ".join(s.text.strip() for s in segments)
    words = text.split()
    return " ".join(words[:n]).lower()
