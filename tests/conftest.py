"""
conftest.py — Shared pytest fixtures
=====================================

Fixtures defined here are available to all test modules automatically.

Key design decisions
--------------------
- ``audio_file`` uses a ``.wav`` extension so ``AudioNormalizer`` passes
  it through unchanged (no pydub decoding or ffmpeg call needed).
- ``mock_normalizer`` is a stub that returns the file path as-is, making
  service tests independent of ffmpeg/pydub and of temp file creation.
- ``mock_transcriber`` always returns ``sample_result`` so service tests
  can assert on identity (``is``) not just equality.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from transcriber.config import ComputeType, TranscriberConfig, WhisperModelSize
from transcriber.protocols import TranscriptionResult, TranscriptionSegment


# ─────────────────────────────────────────────
#  Config fixtures
# ─────────────────────────────────────────────


@pytest.fixture()
def tiny_config() -> TranscriberConfig:
    """Fastest config — suitable for unit tests that don't load the model."""
    return TranscriberConfig(
        model_size=WhisperModelSize.TINY,
        language="es",
        device="cpu",
        compute_type=ComputeType.INT8,
        beam_size=1,
        vad_filter=False,
    )


# ─────────────────────────────────────────────
#  Domain object fixtures
# ─────────────────────────────────────────────


@pytest.fixture()
def sample_segment() -> TranscriptionSegment:
    return TranscriptionSegment(
        start=0.0, end=3.5, text="Hola, bienvenidos.", confidence=0.95
    )


@pytest.fixture()
def sample_result(sample_segment: TranscriptionSegment, tmp_path: Path) -> TranscriptionResult:
    audio = tmp_path / "sample.mp3"
    audio.touch()
    return TranscriptionResult(
        segments=[
            sample_segment,
            TranscriptionSegment(
                start=3.5, end=7.2, text="Hoy hablamos de IA.", confidence=0.88
            ),
        ],
        language="es",
        duration=7.2,
        source_path=audio,
    )


# ─────────────────────────────────────────────
#  Mock normalizer fixture
# ─────────────────────────────────────────────


@pytest.fixture()
def mock_normalizer(audio_file: Path) -> MagicMock:
    """
    Stub AudioNormalizer that returns the audio_file unchanged.

    Injecting this into ``TranscriptionService(normalizer=mock_normalizer)``
    prevents any real ffmpeg / pydub calls in unit tests, so:

    - ``transcribe_file`` calls ``transcribe(audio_file)`` directly
      (not a temp WAV path)
    - The service's source_path stamping is a no-op (paths already match)
    - ``result is sample_result`` identity assertions hold
    """
    prepared = MagicMock()
    prepared.path = audio_file
    prepared.cleanup_required = False

    normalizer = MagicMock()
    normalizer.prepare.return_value = prepared
    normalizer.cleanup.return_value = None
    return normalizer


# ─────────────────────────────────────────────
#  Mock transcriber fixture
# ─────────────────────────────────────────────


@pytest.fixture()
def mock_transcriber(sample_result: TranscriptionResult) -> MagicMock:
    """A mock SpeechTranscriber that always returns sample_result."""
    mock = MagicMock()
    mock.transcribe.return_value = sample_result
    return mock


# ─────────────────────────────────────────────
#  Audio file fixture
# ─────────────────────────────────────────────


@pytest.fixture()
def audio_file(tmp_path: Path) -> Path:
    """
    A dummy WAV file for unit tests.

    Using ``.wav`` means AudioNormalizer passes it through unchanged
    (no pydub decoding needed), so tests don't require a real audio file
    or ffmpeg invocation when a real normalizer is used.
    """
    path = tmp_path / "test_audio.wav"
    path.write_bytes(b"\x00" * 100)
    return path
