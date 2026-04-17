"""
test_config.py — Tests for TranscriberConfig
=============================================
"""

from __future__ import annotations

import pytest

from transcriber.config import ComputeType, TranscriberConfig, WhisperModelSize


class TestTranscriberConfig:
    def test_defaults_are_sensible(self) -> None:
        cfg = TranscriberConfig()
        assert cfg.language == "es"
        assert cfg.device == "cpu"
        assert cfg.model_size == WhisperModelSize.LARGE_V3
        assert cfg.compute_type == ComputeType.INT8
        assert cfg.vad_filter is True

    def test_language_is_lowercased(self) -> None:
        cfg = TranscriberConfig(language="ES")
        assert cfg.language == "es"

    def test_invalid_device_raises(self) -> None:
        with pytest.raises(Exception):
            TranscriberConfig(device="tpu")

    def test_invalid_language_raises(self) -> None:
        with pytest.raises(Exception):
            TranscriberConfig(language="x")

    def test_beam_size_bounds(self) -> None:
        with pytest.raises(Exception):
            TranscriberConfig(beam_size=0)
        with pytest.raises(Exception):
            TranscriberConfig(beam_size=11)

    def test_model_size_enum(self) -> None:
        cfg = TranscriberConfig(model_size=WhisperModelSize.MEDIUM)
        assert cfg.model_size == "medium"
