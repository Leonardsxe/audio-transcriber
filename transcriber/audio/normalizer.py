"""
normalizer.py — audio preparation helpers for transcription
===========================================================

Converts arbitrary audio inputs into a Whisper-friendly WAV file when the
caller needs a normalized intermediate file before inference.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedAudio:
    """Represents a normalized audio file plus whether it should be deleted."""

    path: Path
    cleanup_required: bool = False


class AudioNormalizer:
    """
    Convert audio into 16 kHz mono WAV for direct model transcription.

    WAV inputs are passed through unchanged to avoid unnecessary work.
    """

    def __init__(self) -> None:
        self._check_ffmpeg()

    def prepare(self, audio_path: Path) -> PreparedAudio:
        """Return a transcription-ready audio path."""
        if audio_path.suffix.lower() == ".wav":
            return PreparedAudio(path=audio_path)

        try:
            from pydub import AudioSegment  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "pydub is required for audio conversion. Install: pip install pydub"
            ) from exc

        logger.info("Converting '%s' to 16 kHz mono WAV before transcription.", audio_path.name)
        audio = AudioSegment.from_file(str(audio_path))
        normalized = audio.set_frame_rate(16_000).set_channels(1)

        tmp_dir = Path(tempfile.gettempdir()) / "transcriber_prepared_audio"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        tmp_file = tempfile.NamedTemporaryFile(
            prefix=f"{audio_path.stem}_",
            suffix=".wav",
            dir=tmp_dir,
            delete=False,
        )
        tmp_path = Path(tmp_file.name)
        tmp_file.close()

        normalized.export(tmp_path, format="wav")
        return PreparedAudio(path=tmp_path, cleanup_required=True)

    @staticmethod
    def cleanup(prepared: PreparedAudio) -> None:
        """Delete a temporary normalized file if this instance created it."""
        if not prepared.cleanup_required:
            return
        try:
            prepared.path.unlink(missing_ok=True)
        except OSError:
            logger.warning("Failed to delete temporary audio '%s'.", prepared.path, exc_info=True)

    @staticmethod
    def _check_ffmpeg() -> None:
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH.\n"
                "Install it with:  sudo apt install ffmpeg"
            )
