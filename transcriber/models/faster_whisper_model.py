"""
faster_whisper_model.py — faster-whisper transcription engine
=============================================================

Wraps the ``faster-whisper`` library (CTranslate2 backend) and adapts
its output to the shared ``TranscriptionResult`` / ``TranscriptionSegment``
domain model.

Responsibilities (SRP)
-----------------------
- Load and hold the Whisper model.
- Run inference on a single audio file.
- Convert raw faster-whisper output into domain objects.
- Nothing else.

Why faster-whisper?
-------------------
- Same accuracy as OpenAI Whisper large-v3.
- ~4× faster inference on CPU via CTranslate2 INT8 quantisation.
- Lower peak RAM.
- Built-in VAD (Voice Activity Detection) via Silero.
- MIT-licensed.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Imported only for type-checking; at runtime the import is deferred
    # to __init__ so the module can be loaded without faster-whisper installed
    # (useful for running unit tests in a lightweight environment).
    from faster_whisper import WhisperModel
    from faster_whisper.transcribe import Segment as FWSegment

from transcriber.config import TranscriberConfig
from transcriber.protocols import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber:
    """
    ``SpeechTranscriber`` implementation backed by faster-whisper.

    The model is loaded once on construction and reused for every call to
    :meth:`transcribe`, so instantiate this class once and share it.

    Parameters
    ----------
    config:
        Runtime settings (model size, device, language, …).

    Examples
    --------
    >>> from transcriber.config import TranscriberConfig, WhisperModelSize
    >>> from transcriber.models.faster_whisper_model import FasterWhisperTranscriber
    >>> from pathlib import Path
    >>>
    >>> cfg = TranscriberConfig(model_size=WhisperModelSize.TINY)
    >>> engine = FasterWhisperTranscriber(cfg)
    >>> result = engine.transcribe(Path("audio/sample.wav"))
    >>> print(result.full_text)
    """

    def __init__(self, config: TranscriberConfig) -> None:
        self._config = config
        self._model: Any = self._load_model()

    # ── private ──────────────────────────────────────────────────────────────

    def _load_model(self) -> Any:
        """
        Instantiate and return a ``WhisperModel``.

        The model weights are downloaded automatically on first use and
        cached at ``~/.cache/huggingface/hub/``.
        """
        try:
            from faster_whisper import WhisperModel  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "faster-whisper is required. Install it with: pip install faster-whisper"
            ) from exc

        logger.info(
            "Loading Whisper model '%s' on %s (%s) …",
            self._config.model_size,
            self._config.device,
            self._config.compute_type,
        )
        return WhisperModel(
            model_size_or_path=self._config.model_size,
            device=self._config.device,
            compute_type=self._config.compute_type,
        )

    @staticmethod
    def _to_segment(raw: FWSegment) -> TranscriptionSegment:
        """
        Convert a faster-whisper ``Segment`` into our domain object.

        Parameters
        ----------
        raw:
            Segment as returned by ``WhisperModel.transcribe``.

        Returns
        -------
        TranscriptionSegment
            Immutable domain object.
        """
        return TranscriptionSegment(
            start=raw.start,
            end=raw.end,
            text=raw.text,
            confidence=_logprob_to_confidence(raw.avg_logprob),
        )

    def _build_vad_params(self) -> dict[str, int]:
        """Return VAD keyword arguments derived from config."""
        return {"min_speech_duration_ms": self._config.vad_min_speech_duration_ms}

    # ── public API (satisfies SpeechTranscriber protocol) ────────────────────

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        """
        Transcribe the audio file at *audio_path*.

        Parameters
        ----------
        audio_path:
            Path to a supported audio file (mp3, wav, m4a, ogg, flac, …).

        Returns
        -------
        TranscriptionResult
            Immutable result with segments and metadata.

        Raises
        ------
        FileNotFoundError
            If *audio_path* does not exist on disk.
        RuntimeError
            If the underlying faster-whisper engine raises an error.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Transcribing '%s' …", audio_path.name)

        segments_iter, info = self._model.transcribe(
            str(audio_path),
            language=self._config.language,
            beam_size=self._config.beam_size,
            vad_filter=self._config.vad_filter,
            vad_parameters=self._build_vad_params(),
        )

        # Materialise the lazy iterator so errors surface here, not later.
        segments = [self._to_segment(seg) for seg in segments_iter]

        logger.info(
            "Done — %d segment(s), %.1f s, detected language: %s",
            len(segments),
            info.duration,
            info.language,
        )

        return TranscriptionResult(
            segments=segments,
            language=info.language,
            duration=info.duration,
            source_path=audio_path.resolve(),
        )


# ─────────────────────────────────────────────
#  Pure helper (no side-effects, easy to test)
# ─────────────────────────────────────────────


def _logprob_to_confidence(avg_logprob: float) -> float:
    """
    Convert a model log-probability to a human-readable confidence score.

    Whisper reports ``avg_logprob`` (average log-probability per token).
    ``exp(avg_logprob)`` maps it back to a probability in ``(0, 1]``.
    We clamp to ``[0.0, 1.0]`` to guard against floating-point edge cases.

    Parameters
    ----------
    avg_logprob:
        Average log-probability as returned by faster-whisper
        (typically a negative float close to 0).

    Returns
    -------
    float
        Confidence score in ``[0.0, 1.0]``, rounded to 4 decimal places.

    Examples
    --------
    >>> round(_logprob_to_confidence(-0.1), 2)
    0.9
    >>> _logprob_to_confidence(-10.0)  # very uncertain
    0.0
    """
    return round(min(1.0, max(0.0, math.exp(avg_logprob))), 4)
