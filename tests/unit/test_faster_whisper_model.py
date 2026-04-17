"""
test_faster_whisper_model.py — Tests for FasterWhisperTranscriber helpers
=========================================================================

We test the pure helper function and constructor-level behaviour without
loading the actual model weights (that would require a GPU / large download).
"""

from __future__ import annotations

import math

import pytest

from transcriber.models.faster_whisper_model import _logprob_to_confidence


class TestLogprobToConfidence:
    """Unit tests for the log-probability → confidence converter."""

    def test_zero_logprob_gives_one(self) -> None:
        # exp(0) == 1.0 → maximum confidence
        assert _logprob_to_confidence(0.0) == pytest.approx(1.0)

    def test_negative_logprob_below_one(self) -> None:
        result = _logprob_to_confidence(-0.5)
        assert 0.0 < result < 1.0

    def test_very_negative_logprob_clamps_to_zero(self) -> None:
        # exp(-100) ≈ 0 → should not be negative
        assert _logprob_to_confidence(-100.0) == pytest.approx(0.0, abs=1e-4)

    def test_positive_logprob_clamped_to_one(self) -> None:
        # Shouldn't happen in practice, but we guard against it.
        assert _logprob_to_confidence(5.0) == 1.0

    def test_known_value(self) -> None:
        # exp(-0.1) ≈ 0.9048
        expected = round(min(1.0, math.exp(-0.1)), 4)
        assert _logprob_to_confidence(-0.1) == pytest.approx(expected)

    def test_output_is_rounded_to_4_decimals(self) -> None:
        result = _logprob_to_confidence(-0.12345)
        decimal_places = len(str(result).split(".")[-1])
        assert decimal_places <= 4
