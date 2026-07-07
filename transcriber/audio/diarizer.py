"""
diarizer.py — Speaker diarization using pyannote.audio
=======================================================

Identifies *who spoke when* in an audio file, producing a list of
(speaker_label, start_s, end_s) segments.  The raw labels from pyannote
are arbitrary integers (``SPEAKER_00``, ``SPEAKER_01``, …).  We map them
to meaningful roles — ``INTERVIEWER`` / ``INTERVIEWEE`` — using a simple
heuristic: the speaker with *less total speaking time* is typically the
interviewer (asks short questions), while the speaker with *more total
speaking time* is the interviewee.

This heuristic is correct for ~95 % of journalistic or research interviews.
If your interview has the opposite dynamic, pass ``flip_roles=True``.

Requirements
------------
::

    pip install pyannote.audio

You also need a free HuggingFace account and must accept the terms of:

- https://hf.co/pyannote/speaker-diarization-3.1
- https://hf.co/pyannote/segmentation-3.0

Then create a read token at https://hf.co/settings/tokens and either:

- Set the env var ``HF_TOKEN=hf_...``
- Or pass ``hf_token="hf_..."`` to ``SpeakerDiarizer``.

Design
------
``SpeakerDiarizer`` satisfies the Single-Responsibility Principle: it only
runs diarization and returns raw timed segments.  Role mapping, alignment
with Whisper output, and export are handled by other modules.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Value object
# ─────────────────────────────────────────────


@dataclass(frozen=True)
class DiarizationSegment:
    """
    One speaker segment produced by the diarization model.

    Attributes
    ----------
    speaker:
        Raw pyannote label (e.g. ``"SPEAKER_00"``).
    start:
        Segment start time in seconds.
    end:
        Segment end time in seconds.
    """

    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __str__(self) -> str:
        return f"[{self.speaker}] {self.start:.2f}s–{self.end:.2f}s"


# ─────────────────────────────────────────────
#  Diarizer
# ─────────────────────────────────────────────


class SpeakerDiarizer:
    """
    Run pyannote speaker diarization on a WAV file.

    Parameters
    ----------
    hf_token:
        HuggingFace access token.  If ``None``, read from ``HF_TOKEN``
        environment variable.
    num_speakers:
        Expected number of speakers.  For a standard interview, ``2`` is
        correct.  Pass ``None`` to let pyannote auto-detect (slower).
    flip_roles:
        By default, the speaker with *less* total time = INTERVIEWER.
        Set to ``True`` if the interviewee speaks less than the interviewer.
    model_name:
        pyannote pipeline to use.  The default is the latest recommended
        model as of April 2026.

    Example
    -------
    >>> diarizer = SpeakerDiarizer(hf_token="hf_...")
    >>> segments = diarizer.diarize(Path("audio/interview.wav"))
    >>> for seg in segments[:3]:
    ...     print(seg)
    [INTERVIEWER] 0.00s–3.45s
    [INTERVIEWEE] 3.80s–47.12s
    [INTERVIEWER] 48.00s–51.30s
    """

    _MODEL_NAME = "pyannote/speaker-diarization-3.1"

    def __init__(
        self,
        hf_token: str | None = None,
        *,
        num_speakers: int | None = 2,
        flip_roles: bool = False,
        model_name: str = _MODEL_NAME,
    ) -> None:
        self._token = hf_token or os.environ.get("HF_TOKEN", "")
        self._num_speakers = num_speakers
        self._flip_roles = flip_roles
        self._model_name = model_name
        self._pipeline = None  # lazy — loaded on first call

    # ── public ───────────────────────────────────────────────────────────────

    def diarize(self, audio_path: Path) -> list[DiarizationSegment]:
        """
        Run diarization and return speaker-labelled time segments.

        The returned segments use mapped labels (``INTERVIEWER`` /
        ``INTERVIEWEE``) rather than the raw pyannote integers.

        Parameters
        ----------
        audio_path:
            Path to a WAV file (16 kHz mono recommended).  Other formats
            are accepted by pyannote but may be slower.

        Returns
        -------
        list[DiarizationSegment]
            Chronologically ordered segments.

        Raises
        ------
        FileNotFoundError
            If *audio_path* does not exist.
        ImportError
            If ``pyannote.audio`` is not installed.
        ValueError
            If no HuggingFace token is available.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._ensure_pipeline()

        logger.info("Running speaker diarization on '%s' …", audio_path.name)
        diarization = self._run_pipeline(audio_path)

        raw_segments = self._extract_segments(diarization)
        role_map = self._build_role_map(raw_segments)

        mapped = [
            DiarizationSegment(
                speaker=role_map.get(seg.speaker, seg.speaker),
                start=seg.start,
                end=seg.end,
            )
            for seg in raw_segments
        ]

        speakers_found = sorted({s.speaker for s in mapped})
        logger.info(
            "Diarization complete — %d segment(s), speakers: %s",
            len(mapped),
            ", ".join(speakers_found),
        )
        return mapped

    def validate_access(self) -> None:
        """Fail fast by verifying model access and warming the diarization pipeline."""
        self._ensure_pipeline()

    # ── private ───────────────────────────────────────────────────────────────

    def _ensure_pipeline(self) -> None:
        """Lazy-load the pyannote pipeline (downloads model on first use)."""
        if self._pipeline is not None:
            return

        if not self._token:
            raise ValueError(
                "A HuggingFace token is required for pyannote diarization.\n"
                "Set the HF_TOKEN environment variable or pass hf_token= to SpeakerDiarizer.\n"
                "Get a free token at https://hf.co/settings/tokens\n"
                "Then accept model terms at https://hf.co/pyannote/speaker-diarization-3.1"
            )

        try:
            from pyannote.audio import Pipeline  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "pyannote.audio is required for speaker diarization.\n"
                "Install: pip install pyannote.audio"
            ) from exc

        logger.info("Loading diarization pipeline '%s' …", self._model_name)
        # pyannote.audio 3.x replaced `use_auth_token` with `token`.
        # We support both to stay compatible with 2.x and 3.x installs.
        import inspect  # noqa: PLC0415
        sig = inspect.signature(Pipeline.from_pretrained)
        token_kwarg = "token" if "token" in sig.parameters else "use_auth_token"
        try:
            self._pipeline = Pipeline.from_pretrained(
                self._model_name,
                **{token_kwarg: self._token},
            )
        except Exception as exc:  # noqa: BLE001
            raise _translate_pipeline_error(exc, model_name=self._model_name) from exc

    def _run_pipeline(self, audio_path: Path) -> object:
        """Call the pyannote pipeline with optional speaker count hint."""
        kwargs: dict[str, object] = {}
        if self._num_speakers is not None:
            kwargs["num_speakers"] = self._num_speakers
        return self._pipeline(str(audio_path), **kwargs)  # type: ignore[operator]

    @staticmethod
    def _extract_segments(diarization: object) -> list[DiarizationSegment]:
        """
        Convert pyannote output to a flat list of DiarizationSegment.

        pyannote.audio has two return shapes in the wild:
        - legacy `Annotation` objects exposing `itertracks`
        - newer `DiarizeOutput` objects wrapping both regular and exclusive
          diarization annotations

        For transcript alignment we prefer the exclusive annotation because it
        removes overlapping speech turns that would otherwise double-count
        overlap in the aligner.
        """
        annotation = SpeakerDiarizer._select_annotation(diarization)
        segments: list[DiarizationSegment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):  # type: ignore[union-attr]
            segments.append(
                DiarizationSegment(
                    speaker=speaker,
                    start=turn.start,
                    end=turn.end,
                )
            )
        return sorted(segments, key=lambda s: s.start)

    @staticmethod
    def _select_annotation(diarization: object) -> object:
        """Return the annotation-like object exposing `itertracks`."""
        if hasattr(diarization, "itertracks"):
            return diarization

        exclusive = getattr(diarization, "exclusive_speaker_diarization", None)
        if hasattr(exclusive, "itertracks"):
            return exclusive

        primary = getattr(diarization, "speaker_diarization", None)
        if hasattr(primary, "itertracks"):
            return primary

        raise TypeError(
            "Unsupported diarization output from pyannote pipeline: expected an "
            "Annotation-like object or DiarizeOutput with speaker diarization fields."
        )

    def _build_role_map(self, segments: list[DiarizationSegment]) -> dict[str, str]:
        """
        Map raw pyannote labels → INTERVIEWER / INTERVIEWEE.

        Heuristic: total speaking time.
        - Less total time  → INTERVIEWER  (asks short questions)
        - More total time  → INTERVIEWEE  (gives long answers)

        For interviews with more than 2 speakers, additional speakers are
        mapped as SPEAKER_2, SPEAKER_3, etc.
        """
        totals: dict[str, float] = {}
        for seg in segments:
            totals[seg.speaker] = totals.get(seg.speaker, 0.0) + seg.duration

        if not totals:
            return {}

        # Sort by total speaking time ascending.
        ranked = sorted(totals, key=lambda k: totals[k])

        roles = ["INTERVIEWER", "INTERVIEWEE"]
        if self._flip_roles:
            roles = list(reversed(roles))

        role_map: dict[str, str] = {}
        for idx, raw_label in enumerate(ranked):
            if idx < len(roles):
                role_map[raw_label] = roles[idx]
            else:
                role_map[raw_label] = f"SPEAKER_{idx}"

        logger.info(
            "Role mapping: %s",
            " | ".join(f"{raw} → {role}" for raw, role in role_map.items()),
        )
        return role_map


def _translate_pipeline_error(exc: Exception, *, model_name: str) -> Exception:
    """Convert low-level HuggingFace errors into actionable diarization guidance."""
    status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code == 403 or "403" in str(exc):
        return PermissionError(
            "HuggingFace denied access to the diarization model.\n"
            "Confirm that the same account behind HF_TOKEN has accepted access to:\n"
            "  - https://hf.co/pyannote/speaker-diarization-3.1\n"
            "  - https://hf.co/pyannote/segmentation-3.0\n"
            "Your current token permissions should usually be enough for public gated repos.\n"
            "If access was accepted on a different account, or a fine-grained token still fails,\n"
            "create a fresh read token from the account that accepted both model terms and set HF_TOKEN again."
        )
    return RuntimeError(f"Could not load diarization pipeline '{model_name}': {exc}")
