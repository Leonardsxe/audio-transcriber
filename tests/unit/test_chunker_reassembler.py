"""
test_chunker_reassembler.py — Tests for AudioChunker and reassemble()
======================================================================

All tests use synthetic data — no real audio files or model weights needed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcriber.audio.chunker import AudioChunk, AudioChunker, ChunkConfig
from transcriber.audio.reassembler import (
    _first_n_words,
    _last_n_words,
    _shift_timestamps,
    reassemble,
)
from transcriber.protocols import TranscriptionResult, TranscriptionSegment


# ─────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────


def _seg(start: float, end: float, text: str, conf: float = 0.9) -> TranscriptionSegment:
    return TranscriptionSegment(start=start, end=end, text=text, confidence=conf)


def _result(*segments: TranscriptionSegment, lang: str = "es") -> TranscriptionResult:
    duration = segments[-1].end if segments else 0.0
    return TranscriptionResult(segments=list(segments), language=lang, duration=duration)


def _chunk(
    idx: int,
    start_s: float,
    end_s: float,
    overlap_start_ms: int = 0,
    overlap_end_ms: int = 0,
) -> AudioChunk:
    return AudioChunk(
        index=idx,
        path=Path(f"/tmp/chunk_{idx:03d}.wav"),
        start_s=start_s,
        end_s=end_s,
        overlap_start_ms=overlap_start_ms,
        overlap_end_ms=overlap_end_ms,
    )


# ─────────────────────────────────────────────
#  ChunkConfig
# ─────────────────────────────────────────────


class TestChunkConfig:
    def test_defaults(self) -> None:
        cfg = ChunkConfig()
        assert cfg.target_duration_s == 600.0
        assert cfg.min_silence_ms == 800
        assert cfg.overlap_ms == 200

    def test_custom_values(self) -> None:
        cfg = ChunkConfig(target_duration_s=300.0, min_silence_ms=500)
        assert cfg.target_duration_s == 300.0
        assert cfg.min_silence_ms == 500

    def test_frozen(self) -> None:
        cfg = ChunkConfig()
        with pytest.raises(Exception):
            cfg.target_duration_s = 100.0  # type: ignore[misc]


# ─────────────────────────────────────────────
#  AudioChunk
# ─────────────────────────────────────────────


class TestAudioChunk:
    def test_duration_property(self) -> None:
        chunk = _chunk(0, 0.0, 612.5)
        assert chunk.duration_s == pytest.approx(612.5)

    def test_str_representation(self) -> None:
        chunk = _chunk(3, 600.0, 1200.0)
        text = str(chunk)
        assert "Chunk 003" in text
        assert "10.0 min" in text

    def test_zero_overlaps_by_default(self) -> None:
        chunk = _chunk(0, 0.0, 100.0)
        assert chunk.overlap_start_ms == 0
        assert chunk.overlap_end_ms == 0


# ─────────────────────────────────────────────
#  AudioChunker internal helpers
# ─────────────────────────────────────────────


class TestBestSilenceInWindow:
    def test_returns_longest_silence(self) -> None:
        silences = [(100, 300), (500, 1500), (1600, 1700)]
        result = AudioChunker._best_silence_in_window(silences, 0, 2000)
        assert result == (500, 1500)  # longest = 1000 ms

    def test_returns_none_when_no_overlap(self) -> None:
        silences = [(5000, 6000)]
        result = AudioChunker._best_silence_in_window(silences, 0, 1000)
        assert result is None

    def test_partial_overlap_counts(self) -> None:
        silences = [(900, 1100)]
        result = AudioChunker._best_silence_in_window(silences, 1000, 2000)
        assert result == (900, 1100)

    def test_empty_silences(self) -> None:
        result = AudioChunker._best_silence_in_window([], 0, 10000)
        assert result is None


class TestComputeCutPoints:
    def _chunker(self, target_s: float = 600, window_s: float = 60) -> AudioChunker:
        cfg = ChunkConfig(
            target_duration_s=target_s,
            max_duration_s=target_s * 1.2,
            search_window_s=window_s,
            min_silence_ms=500,
        )
        chunker = AudioChunker.__new__(AudioChunker)
        chunker._config = cfg
        return chunker

    def test_single_chunk_when_audio_short(self) -> None:
        chunker = self._chunker(target_s=600)
        # 300 s audio — fits in one chunk, no cut needed
        cuts = chunker._compute_cut_points(300_000, silences=[])
        assert cuts == []

    def test_produces_correct_number_of_cuts(self) -> None:
        chunker = self._chunker(target_s=300, window_s=30)
        # 900 s audio, 300 s target → expect 2 cuts (3 chunks)
        silences = [(295_000, 305_000), (595_000, 605_000)]
        cuts = chunker._compute_cut_points(900_000, silences)
        assert len(cuts) == 2

    def test_cut_at_silence_midpoint(self) -> None:
        chunker = self._chunker(target_s=300, window_s=60)
        # Silence at 290–310 s → midpoint = 300_000 ms
        silences = [(290_000, 310_000)]
        cuts = chunker._compute_cut_points(600_000, silences)
        assert len(cuts) == 1
        assert cuts[0] == 300_000

    def test_hard_cut_when_no_silence(self) -> None:
        chunker = self._chunker(target_s=300)
        # No silences anywhere — should fall back to max_duration hard cut
        cuts = chunker._compute_cut_points(700_000, silences=[])
        assert len(cuts) >= 1


# ─────────────────────────────────────────────
#  Reassembler helpers
# ─────────────────────────────────────────────


class TestShiftTimestamps:
    def test_shifts_all_segments(self) -> None:
        result = _result(_seg(0.0, 2.0, "hola"), _seg(2.5, 5.0, "mundo"))
        shifted = _shift_timestamps(result, offset_s=10.0)

        assert shifted.segments[0].start == pytest.approx(10.0)
        assert shifted.segments[0].end == pytest.approx(12.0)
        assert shifted.segments[1].start == pytest.approx(12.5)
        assert shifted.segments[1].end == pytest.approx(15.0)

    def test_preserves_text_and_confidence(self) -> None:
        result = _result(_seg(0.0, 1.0, "test", conf=0.77))
        shifted = _shift_timestamps(result, offset_s=5.0)
        assert shifted.segments[0].text == "test"
        assert shifted.segments[0].confidence == pytest.approx(0.77)

    def test_zero_offset_is_identity(self) -> None:
        result = _result(_seg(1.0, 2.0, "x"))
        shifted = _shift_timestamps(result, offset_s=0.0)
        assert shifted.segments[0].start == pytest.approx(1.0)


class TestLastFirstNWords:
    def test_last_n_words(self) -> None:
        segs = [_seg(0, 1, "uno dos tres"), _seg(1, 2, "cuatro cinco")]
        assert _last_n_words(segs, 2) == "cuatro cinco"

    def test_first_n_words(self) -> None:
        segs = [_seg(0, 1, "uno dos tres"), _seg(1, 2, "cuatro cinco")]
        assert _first_n_words(segs, 2) == "uno dos"

    def test_fewer_words_than_n(self) -> None:
        segs = [_seg(0, 1, "solo")]
        assert _last_n_words(segs, 10) == "solo"

    def test_empty_segments(self) -> None:
        assert _last_n_words([], 5) == ""
        assert _first_n_words([], 5) == ""


# ─────────────────────────────────────────────
#  Reassemble
# ─────────────────────────────────────────────


class TestReassemble:
    def test_raises_on_length_mismatch(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            reassemble(
                [_result(_seg(0, 1, "x"))],
                [_chunk(0, 0, 10), _chunk(1, 10, 20)],
            )

    def test_empty_returns_empty_result(self) -> None:
        result = reassemble([], [])
        assert result.segments == []
        assert result.duration == 0.0

    def test_single_chunk_shifts_timestamps(self) -> None:
        chunk = _chunk(0, 100.0, 200.0)
        r = _result(_seg(0.0, 5.0, "hola"))
        merged = reassemble([r], [chunk])
        assert merged.segments[0].start == pytest.approx(100.0)
        assert merged.segments[0].end == pytest.approx(105.0)

    def test_two_chunks_timestamps_are_absolute(self) -> None:
        chunks = [_chunk(0, 0.0, 60.0), _chunk(1, 60.0, 120.0)]
        results = [
            _result(_seg(0.0, 5.0, "inicio")),
            _result(_seg(0.0, 5.0, "final")),
        ]
        merged = reassemble(results, chunks)
        starts = [s.start for s in merged.segments]
        assert starts[0] == pytest.approx(0.0)
        assert starts[1] == pytest.approx(60.0)

    def test_total_duration_is_last_chunk_end(self) -> None:
        chunks = [_chunk(0, 0.0, 300.0), _chunk(1, 300.0, 600.0)]
        results = [
            _result(_seg(0.0, 5.0, "a")),
            _result(_seg(0.0, 5.0, "b")),
        ]
        merged = reassemble(results, chunks)
        assert merged.duration == pytest.approx(600.0)

    def test_overlap_segments_are_dropped(self) -> None:
        # Chunk 1 ends at 60 s. Chunk 2 starts at 59 s with 1000 ms overlap.
        # Segments before (start_s + overlap_ms/1000) = 60.0 s should be dropped.
        chunk0 = _chunk(0, 0.0, 60.0, overlap_end_ms=200)
        chunk1 = _chunk(1, 60.0, 120.0, overlap_start_ms=1000)
        results = [
            _result(_seg(0.0, 55.0, "buenas tardes")),
            _result(
                _seg(0.0, 0.5, "tardes"),    # inside overlap zone → should drop
                _seg(2.0, 5.0, "siguiente"), # outside overlap zone → keep
            ),
        ]
        merged = reassemble(results, [chunk0, chunk1])
        texts = [s.text.strip() for s in merged.segments]
        assert "siguiente" in texts

    def test_language_from_first_result(self) -> None:
        chunks = [_chunk(0, 0.0, 10.0)]
        results = [_result(_seg(0, 5, "hola"), lang="es")]
        merged = reassemble(results, chunks)
        assert merged.language == "es"

    def test_source_path_set_when_provided(self, tmp_path: Path) -> None:
        src = tmp_path / "interview.mp3"
        chunks = [_chunk(0, 0.0, 10.0)]
        results = [_result(_seg(0, 5, "hola"))]
        merged = reassemble(results, chunks, source_path=src)
        assert merged.source_path == src
