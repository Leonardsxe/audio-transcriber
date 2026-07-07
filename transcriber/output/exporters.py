"""
exporters.py — ResultExporter implementations (Open/Closed Principle)
======================================================================

Three concrete exporters ship out of the box.  Each satisfies the
:class:`~transcriber.protocols.ResultExporter` protocol without inheriting
from any shared base class — pure duck-typing.

Adding a new format:
    1. Create a class with ``export(result, destination) -> Path``.
    2. Register it in :func:`exporter_for` (or inject it directly).
    3. No existing code changes required.

Output formats
--------------
PlainTextExporter  → .txt   — Human-readable transcript.
JsonExporter       → .json  — Machine-readable with full metadata.
SrtExporter        → .srt   — SubRip subtitle format for video players.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from transcriber.protocols import TranscriptionResult, TranscriptionSegment

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────


def _ensure_parent(path: Path) -> None:
    """Create parent directories for *path* if they don't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _seconds_to_srt_time(seconds: float) -> str:
    """
    Format *seconds* as an SRT timestamp ``HH:MM:SS,mmm``.

    Parameters
    ----------
    seconds:
        Time in seconds (may include fractional part).

    Returns
    -------
    str
        SRT-formatted timestamp, e.g. ``"00:01:23,456"``.

    Examples
    --------
    >>> _seconds_to_srt_time(83.456)
    '00:01:23,456'
    """
    total_ms = int(seconds * 1000)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    secs = total_s % 60
    total_m = total_s // 60
    mins = total_m % 60
    hours = total_m // 60
    return f"{hours:02d}:{mins:02d}:{secs:02d},{ms:03d}"


# ─────────────────────────────────────────────
#  Exporters
# ─────────────────────────────────────────────


class PlainTextExporter:
    """
    Write the full transcript as a ``.txt`` file.

    The file contains the plain concatenated text — no timestamps,
    no metadata.  Ideal for feeding into downstream NLP pipelines.

    Example output::

        Hola, bienvenidos al programa de hoy.
        Hoy vamos a hablar sobre inteligencia artificial.
    """

    def export(self, result: TranscriptionResult, destination: Path) -> Path:
        """
        Write *result* to *destination* (suffix forced to ``.txt``).

        Parameters
        ----------
        result:
            The transcription to serialise.
        destination:
            Target path.  The ``.txt`` extension is appended if missing.

        Returns
        -------
        Path
            The resolved path of the written file.
        """
        path = destination.with_suffix(".txt")
        _ensure_parent(path)
        path.write_text(result.full_text, encoding="utf-8")
        logger.debug("PlainTextExporter → %s", path)
        return path


class JsonExporter:
    """
    Write the full transcription as a ``.json`` file.

    The JSON includes every segment with start/end times, text, and
    confidence score — useful for integration with downstream systems.

    Example output::

        {
          "language": "es",
          "duration": 42.3,
          "average_confidence": 0.91,
          "source": "/abs/path/to/audio.mp3",
          "segments": [
            {"start": 0.0, "end": 3.2, "text": "Hola,", "confidence": 0.95},
            ...
          ]
        }
    """

    def export(self, result: TranscriptionResult, destination: Path) -> Path:
        """
        Write *result* to *destination* (suffix forced to ``.json``).

        Returns
        -------
        Path
            The resolved path of the written file.
        """
        path = destination.with_suffix(".json")
        _ensure_parent(path)

        payload = {
            "language": result.language,
            "duration": round(result.duration, 3),
            "average_confidence": round(result.average_confidence, 4),
            "source": str(result.source_path) if result.source_path else None,
            "segments": [_segment_to_dict(seg) for seg in result.segments],
        }

        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("JsonExporter → %s", path)
        return path


class SrtExporter:
    """
    Write the transcription as a ``.srt`` (SubRip) subtitle file.

    SRT is universally supported by video players (VLC, mpv, …) and
    editing tools (DaVinci Resolve, Premiere, …).

    Example output::

        1
        00:00:00,000 --> 00:00:03,200
        Hola, bienvenidos al programa.

        2
        00:00:03,500 --> 00:00:07,100
        Hoy vamos a hablar sobre IA.
    """

    def export(self, result: TranscriptionResult, destination: Path) -> Path:
        """
        Write *result* to *destination* (suffix forced to ``.srt``).

        Returns
        -------
        Path
            The resolved path of the written file.
        """
        path = destination.with_suffix(".srt")
        _ensure_parent(path)

        lines: list[str] = []
        for idx, seg in enumerate(result.segments, start=1):
            lines.append(str(idx))
            lines.append(
                f"{_seconds_to_srt_time(seg.start)} --> {_seconds_to_srt_time(seg.end)}"
            )
            lines.append(seg.text.strip())
            lines.append("")  # blank line between entries

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.debug("SrtExporter → %s", path)
        return path


# ─────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────

_EXPORTERS: dict[str, type[PlainTextExporter | JsonExporter | SrtExporter]] = {
    ".txt": PlainTextExporter,
    ".json": JsonExporter,
    ".srt": SrtExporter,
}


def exporter_for(fmt: str) -> PlainTextExporter | JsonExporter | SrtExporter:
    """
    Return the appropriate exporter for a given file extension.

    Parameters
    ----------
    fmt:
        File extension including the dot: ``".txt"``, ``".json"``, ``".srt"``.

    Returns
    -------
    PlainTextExporter | JsonExporter | SrtExporter
        A fresh exporter instance.

    Raises
    ------
    ValueError
        If *fmt* is not a recognised extension.

    Example
    -------
    >>> exp = exporter_for(".srt")
    >>> exp.export(result, Path("output/clip"))
    """
    key = fmt.lower()
    if key not in _EXPORTERS:
        raise ValueError(f"No exporter for format {fmt!r}. Choose from {list(_EXPORTERS)}")
    return _EXPORTERS[key]()


# ─────────────────────────────────────────────
#  Private helpers
# ─────────────────────────────────────────────


def _segment_to_dict(seg: TranscriptionSegment) -> dict[str, float | str]:
    return {
        "start": round(seg.start, 3),
        "end": round(seg.end, 3),
        "text": seg.text.strip(),
        "confidence": seg.confidence,
    }


# ═════════════════════════════════════════════
#  Diarized transcript exporters
# ═════════════════════════════════════════════
#
# These exporters operate on ``DiarizedTranscript`` objects and produce
# the two recommended output formats for qualitative research pipelines:
#
#  TranscriptJsonExporter   → .transcript.json   (machine-readable)
#  SpeakerTextExporter      → .transcript.txt    (human-readable)
#
# Why NOT plain .txt for auto-coding / thematic synthesis
# -------------------------------------------------------
# Plain .txt loses all structure:
#   - No speaker identity → cannot split interviewer vs interviewee turns
#   - No timestamps → cannot anchor quotes to the audio
#   - No confidence scores → cannot flag uncertain passages for review
#   - No turn boundaries → sentence segmentation must be re-computed
#
# The .transcript.json format is the canonical machine-readable form.
# The .transcript.txt is the human-readable companion for NVivo / ATLAS.ti
# import or manual annotation — it preserves speaker turns as labelled
# paragraphs so coders can see the conversational flow.


from transcriber.protocols import DiarizedTranscript, SpeakerTurn  # noqa: E402


class TranscriptJsonExporter:
    """
    Export a ``DiarizedTranscript`` as a structured JSON file.

    This is the **recommended format** for downstream auto-coding and
    thematic synthesis pipelines.  The schema is designed to be:

    - Directly iterable as speaker turns (the atomic unit for coding)
    - Self-describing (metadata included)
    - Lossless (all timestamps, confidence scores, and per-segment text)

    Output schema::

        {
          "schema_version": "1.0",
          "metadata": {
            "source": "/abs/path/to/interview.mp3",
            "duration_s": 2847.3,
            "language": "es",
            "avg_confidence": 0.91
          },
          "speakers": {
            "INTERVIEWER": {
              "total_speech_s": 342.1,
              "turn_count": 47,
              "word_count": 1820
            },
            "INTERVIEWEE": {
              "total_speech_s": 2505.2,
              "turn_count": 44,
              "word_count": 18340
            }
          },
          "turns": [
            {
              "turn_index": 0,
              "speaker": "INTERVIEWEE",
              "start": 0.0,
              "end": 12.4,
              "duration": 12.4,
              "word_count": 23,
              "text": "Buenos días, muchas gracias por recibirme hoy.",
              "segments": [
                {
                  "start": 0.0, "end": 5.1,
                  "text": "Buenos días, muchas gracias",
                  "confidence": 0.94
                },
                ...
              ]
            },
            ...
          ]
        }

    Usage for auto-coding
    ----------------------
    ::

        import json
        data = json.loads(Path("interview.transcript.json").read_text())
        interviewee_turns = [t for t in data["turns"] if t["speaker"] == "INTERVIEWEE"]
        for turn in interviewee_turns:
            print(f"[{turn['start']:.1f}s] {turn['text']}")
    """

    def export(self, result: DiarizedTranscript, destination: Path) -> Path:
        """
        Write *result* to *destination* (suffix forced to ``.transcript.json``).

        Parameters
        ----------
        result:
            The diarized transcript to serialise.
        destination:
            Target file path stem (extension is set automatically).

        Returns
        -------
        Path
            The resolved path of the written file.
        """
        path = destination.with_suffix(".transcript.json")
        _ensure_parent(path)

        payload = {
            "schema_version": "1.0",
            "metadata": {
                "source": str(result.source.source_path) if result.source.source_path else None,
                "duration_s": round(result.source.duration, 3),
                "language": result.source.language,
                "avg_confidence": round(result.source.average_confidence, 4),
            },
            "speakers": {
                label: {
                    "total_speech_s": round(stats.total_speech_s, 2),
                    "turn_count": stats.turn_count,
                    "word_count": stats.word_count,
                }
                for label, stats in result.speakers.items()
            },
            "turns": [
                _turn_to_dict(turn, idx)
                for idx, turn in enumerate(result.turns)
            ],
        }

        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("TranscriptJsonExporter → %s", path)
        return path


class SpeakerTextExporter:
    """
    Export a ``DiarizedTranscript`` as a speaker-labelled plain-text file.

    This is the **recommended human-readable format** — suitable for:

    - Manual import into NVivo, ATLAS.ti, Dedoose, or MaxQDA
    - Distributing to collaborators for annotation
    - Quick review without a JSON viewer

    Format::

        ══════════════════════════════════════════════════
        Interview transcript
        Duration : 47.5 min  |  Language: es
        Speakers : INTERVIEWER (47 turns) · INTERVIEWEE (44 turns)
        ══════════════════════════════════════════════════

        [INTERVIEWER] 0:00:00
        Buenos días. Para empezar, ¿me puede contar un poco sobre su historia?

        [INTERVIEWEE] 0:00:08
        Claro, con mucho gusto. Nací en Bogotá en 1978 y…

        [INTERVIEWER] 0:04:32
        ¿Y cómo fue esa experiencia para usted?

        ...

    Each block is a single conversational turn.  The timestamp anchors the
    speaker to the audio file for verification.

    Why not plain .txt?
    -------------------
    A plain .txt file has no speaker boundaries — it is just a wall of
    text.  Auto-coders and manual coders both need the turn structure to:

    1. Know whose words they are reading.
    2. Find the passage in the audio for playback.
    3. Separate interviewer prompts from interviewee responses before
       thematic analysis.
    """

    def export(self, result: DiarizedTranscript, destination: Path) -> Path:
        """
        Write *result* to *destination* (suffix forced to ``.transcript.txt``).

        Returns
        -------
        Path
            The resolved path of the written file.
        """
        path = destination.with_suffix(".transcript.txt")
        _ensure_parent(path)

        lines: list[str] = []

        # ── Header ──────────────────────────────────────────────────────────
        sep = "═" * 56
        lines.append(sep)
        lines.append("Interview transcript")
        dur_min = result.source.duration / 60
        lines.append(
            f"Duration : {dur_min:.1f} min  |  Language: {result.source.language}"
        )
        speaker_summary = " · ".join(
            f"{lbl} ({s.turn_count} turns, ~{s.word_count} words)"
            for lbl, s in result.speakers.items()
        )
        lines.append(f"Speakers : {speaker_summary}")
        if result.source.source_path:
            lines.append(f"Source   : {result.source.source_path.name}")
        lines.append(sep)
        lines.append("")

        # ── Turns ────────────────────────────────────────────────────────────
        for turn in result.turns:
            timestamp = _seconds_to_hms(turn.start)
            lines.append(f"[{turn.speaker}] {timestamp}")
            lines.append(turn.text.strip())
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.debug("SpeakerTextExporter → %s", path)
        return path


# ─────────────────────────────────────────────
#  Update factory to include new formats
# ─────────────────────────────────────────────

_EXPORTERS["transcript.json"] = TranscriptJsonExporter  # type: ignore[index]
_EXPORTERS["transcript.txt"] = SpeakerTextExporter      # type: ignore[index]


# ─────────────────────────────────────────────
#  Private helpers for diarized exporters
# ─────────────────────────────────────────────


def _turn_to_dict(turn: SpeakerTurn, idx: int) -> dict:
    return {
        "turn_index": idx,
        "speaker": turn.speaker,
        "start": round(turn.start, 3),
        "end": round(turn.end, 3),
        "duration": round(turn.duration, 3),
        "word_count": turn.word_count,
        "text": turn.text.strip(),
        "segments": [
            {
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "text": s.text.strip(),
                "confidence": s.confidence,
            }
            for s in turn.segments
        ],
    }


def _seconds_to_hms(seconds: float) -> str:
    """Format seconds as ``H:MM:SS`` for the human-readable transcript."""
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h}:{m:02d}:{s:02d}"
