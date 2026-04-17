"""
test_protocols.py — Tests for domain value objects
====================================================
"""

from __future__ import annotations

from transcriber.protocols import TranscriptionResult, TranscriptionSegment


class TestTranscriptionSegment:
    def test_duration_is_end_minus_start(self) -> None:
        seg = TranscriptionSegment(start=1.0, end=4.5, text="hola", confidence=0.9)
        assert seg.duration == pytest.approx(3.5)

    def test_str_contains_text(self) -> None:
        seg = TranscriptionSegment(start=0.0, end=2.0, text=" test ", confidence=0.8)
        assert "test" in str(seg)

    def test_frozen_raises_on_mutation(self) -> None:
        seg = TranscriptionSegment(start=0.0, end=1.0, text="x", confidence=1.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            seg.text = "y"  # type: ignore[misc]


class TestTranscriptionResult:
    def test_full_text_joins_segments(self, sample_result: TranscriptionResult) -> None:
        text = sample_result.full_text
        assert "Hola" in text
        assert "IA" in text

    def test_average_confidence(self, sample_result: TranscriptionResult) -> None:
        expected = (0.95 + 0.88) / 2
        assert sample_result.average_confidence == pytest.approx(expected)

    def test_average_confidence_empty(self) -> None:
        result = TranscriptionResult(segments=[], language="es", duration=0.0)
        assert result.average_confidence == 0.0

    def test_str_representation(self, sample_result: TranscriptionResult) -> None:
        text = str(sample_result)
        assert "es" in text
        assert "7.2" in text


import pytest  # noqa: E402 — kept at bottom for readability
