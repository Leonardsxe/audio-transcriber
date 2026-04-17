"""
config.py — Centralised configuration (Open/Closed + Single-Responsibility)
============================================================================

All tuneable parameters live here.  ``TranscriberConfig`` is loaded from
environment variables (or a ``.env`` file) via *pydantic-settings*, so the
same codebase works in dev, CI, and production without code changes.

Usage
-----
Load defaults (reads `.env` automatically if present)::

    config = TranscriberConfig()

Override specific fields at call-site (useful in tests)::

    config = TranscriberConfig(model_size=WhisperModelSize.TINY, device="cpu")

Override via environment::

    TRANSCRIBER_MODEL_SIZE=medium python -m transcriber.main audio/file.mp3
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class WhisperModelSize(StrEnum):
    """
    Available Whisper model sizes ordered from fastest to most accurate.

    +----------+--------+-------+-----------------------------------+
    | Name     | RAM    | Speed | Spanish WER (approx.)             |
    +==========+========+=======+===================================+
    | tiny     | ~1 GB  | ████  | ~15 %  — quick drafts             |
    | base     | ~1 GB  | ███   | ~12 %  — prototyping              |
    | small    | ~2 GB  | ██    | ~8 %   — lightweight production   |
    | medium   | ~5 GB  | █     | ~6 %   — balanced production      |
    | large-v3 | ~10 GB | ░     | ~3 %   — best accuracy ✓          |
    +----------+--------+-------+-----------------------------------+
    """

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE_V3 = "large-v3"


class ComputeType(StrEnum):
    """
    Numerical precision used by CTranslate2 (faster-whisper backend).

    Choose based on your hardware:

    - ``INT8``    → CPU — smallest memory footprint, fastest on modern CPUs.
    - ``FLOAT16`` → GPU — best speed/quality trade-off on NVIDIA cards.
    - ``FLOAT32`` → CPU/GPU — highest accuracy, most memory.
    """

    INT8 = "int8"
    FLOAT16 = "float16"
    FLOAT32 = "float32"


class TranscriberConfig(BaseSettings):
    """
    Runtime configuration for the transcription pipeline.

    All fields can be overridden via environment variables prefixed with
    ``TRANSCRIBER_`` or via a ``.env`` file in the working directory.

    Attributes
    ----------
    model_size:
        Whisper model variant to load.
    language:
        BCP-47 / ISO 639-1 language code for forced decoding.
        ``"es"`` = Spanish.  Set to ``None`` to let the model auto-detect.
    device:
        ``"cpu"`` or ``"cuda"``.  Auto-detection is intentionally skipped
        so behaviour stays predictable across machines.
    compute_type:
        CTranslate2 precision.  ``INT8`` is the safe default for CPU.
    beam_size:
        Beam-search width.  Higher values improve accuracy at the cost
        of speed.  5 is a good balance for Spanish.
    vad_filter:
        When ``True``, a Voice Activity Detection pass removes silent
        regions before transcription, reducing hallucinations.
    vad_min_speech_duration_ms:
        Minimum speech segment length (ms) kept by the VAD filter.
    output_dir:
        Default directory where exporters write results.
    """

    model_config = SettingsConfigDict(
        env_prefix="TRANSCRIBER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    model_size: WhisperModelSize = Field(
        default=WhisperModelSize.LARGE_V3,
        description="Whisper model variant.",
    )
    language: str = Field(
        default="es",
        description="ISO 639-1 language code, e.g. 'es' for Spanish.",
    )
    device: str = Field(
        default="cpu",
        description="Inference device: 'cpu' or 'cuda'.",
    )
    compute_type: ComputeType = Field(
        default=ComputeType.INT8,
        description="Numerical precision for CTranslate2.",
    )
    beam_size: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Beam-search width (1 = greedy).",
    )
    vad_filter: bool = Field(
        default=True,
        description="Enable Voice Activity Detection pre-processing.",
    )
    vad_min_speech_duration_ms: int = Field(
        default=250,
        ge=0,
        description="Minimum speech duration kept by VAD (milliseconds).",
    )
    output_dir: Path = Field(
        default=Path("./output"),
        description="Directory for exported transcription files.",
    )

    @field_validator("device")
    @classmethod
    def _validate_device(cls, value: str) -> str:
        allowed = {"cpu", "cuda"}
        if value not in allowed:
            raise ValueError(f"device must be one of {allowed}, got {value!r}")
        return value

    @field_validator("language")
    @classmethod
    def _validate_language(cls, value: str) -> str:
        if len(value) < 2:  # noqa: PLR2004
            raise ValueError(f"language must be a valid ISO code, got {value!r}")
        return value.lower()
