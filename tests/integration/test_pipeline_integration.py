"""
test_pipeline_integration.py — End-to-end pipeline integration tests
=====================================================================

These tests exercise the full pipeline from service → model → exporter.
They are skipped by default because they require:
  - The faster-whisper package installed
  - A real audio file (even a tiny synthetic WAV works)
  - ~1 GB RAM for the tiny model

Run them explicitly::

    pytest tests/integration/ -v

Or with the mark::

    pytest -m integration
"""

from __future__ import annotations

import wave
import struct
import math
from pathlib import Path

import pytest

# Mark all tests in this module as integration tests so they can be
# selectively skipped in CI with: pytest -m "not integration"
pytestmark = pytest.mark.integration


def _make_sine_wav(path: Path, duration_s: float = 1.0, freq: float = 440.0) -> Path:
    """
    Generate a minimal mono 16-bit PCM WAV file at 16 kHz.

    Whisper expects 16 kHz mono audio — this produces exactly that.
    The content is a pure sine wave (no speech), so the transcript will
    likely be empty or contain hallucinations, but the pipeline itself
    will run end-to-end without errors.

    Parameters
    ----------
    path:
        Destination file path.
    duration_s:
        Duration in seconds.
    freq:
        Sine wave frequency in Hz.
    """
    sample_rate = 16_000
    n_samples = int(sample_rate * duration_s)
    amplitude = 32767 * 0.3  # 30% of max to avoid clipping

    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        samples = [
            int(amplitude * math.sin(2 * math.pi * freq * i / sample_rate))
            for i in range(n_samples)
        ]
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))

    return path


@pytest.fixture(scope="module")
def synthetic_wav(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A short synthetic WAV file reused across all integration tests."""
    tmp = tmp_path_factory.mktemp("audio")
    return _make_sine_wav(tmp / "synthetic.wav", duration_s=2.0)


@pytest.mark.integration
def test_full_pipeline_runs_without_error(synthetic_wav: Path, tmp_path: Path) -> None:
    """
    Verify the full pipeline executes without raising exceptions.

    This test loads the real tiny model (~74 MB download on first run).
    """
    from transcriber.config import TranscriberConfig, WhisperModelSize, ComputeType
    from transcriber.models.faster_whisper_model import FasterWhisperTranscriber
    from transcriber.transcription.service import TranscriptionService

    config = TranscriberConfig(
        model_size=WhisperModelSize.TINY,
        language="es",
        device="cpu",
        compute_type=ComputeType.INT8,
        vad_filter=False,
    )

    engine = FasterWhisperTranscriber(config)
    service = TranscriptionService(engine)

    result = service.transcribe_file(synthetic_wav)

    assert result is not None
    assert result.language is not None
    assert result.duration > 0
    assert isinstance(result.full_text, str)


@pytest.mark.integration
def test_pipeline_exports_to_all_formats(synthetic_wav: Path, tmp_path: Path) -> None:
    """Verify that all three exporters write valid files after a real transcription."""
    from transcriber.config import TranscriberConfig, WhisperModelSize, ComputeType
    from transcriber.models.faster_whisper_model import FasterWhisperTranscriber
    from transcriber.output.exporters import JsonExporter, PlainTextExporter, SrtExporter
    from transcriber.transcription.service import TranscriptionService
    import json

    config = TranscriberConfig(
        model_size=WhisperModelSize.TINY,
        language="es",
        device="cpu",
        compute_type=ComputeType.INT8,
        vad_filter=False,
    )

    engine = FasterWhisperTranscriber(config)
    service = TranscriptionService(engine)
    result = service.transcribe_file(synthetic_wav)

    # Plain text
    txt_path = PlainTextExporter().export(result, tmp_path / "out")
    assert txt_path.exists()

    # JSON — must be valid
    json_path = JsonExporter().export(result, tmp_path / "out")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "segments" in data

    # SRT
    srt_path = SrtExporter().export(result, tmp_path / "out")
    assert srt_path.exists()
