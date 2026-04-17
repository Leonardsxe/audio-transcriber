"""
audio-transcriber
=================
Spanish audio transcription package built on faster-whisper.

Public re-exports keep the consumer API stable even if internal
module layout changes.

Example
-------
>>> from transcriber import TranscriptionService, TranscriberConfig
>>> from transcriber.models.faster_whisper_model import FasterWhisperTranscriber
>>> from pathlib import Path
>>>
>>> config  = TranscriberConfig()
>>> engine  = FasterWhisperTranscriber(config)
>>> service = TranscriptionService(engine)
>>> result  = service.transcribe_file(Path("audio/interview.mp3"))
>>> print(result.full_text)
"""

from transcriber.config import ComputeType, TranscriberConfig, WhisperModelSize
from transcriber.protocols import (
    SpeechTranscriber,
    TranscriptionResult,
    TranscriptionSegment,
)
from transcriber.transcription.service import TranscriptionService

__all__ = [
    # Configuration
    "TranscriberConfig",
    "WhisperModelSize",
    "ComputeType",
    # Protocols (interfaces)
    "SpeechTranscriber",
    # Domain models
    "TranscriptionResult",
    "TranscriptionSegment",
    # Service
    "TranscriptionService",
]
