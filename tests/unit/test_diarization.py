"""
test_diarization.py — Tests for diarizer, aligner, and diarized exporters
=========================================================================

All tests use synthetic data — no real model or HuggingFace token needed.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from transcriber.audio.aligner import (
    _assign_speakers,
    _compute_stats,
    _dominant_speaker,
    _group_into_turns,
    align,
)
from transcriber.audio.diarizer import DiarizationSegment, SpeakerDiarizer
from transcriber.output.exporters import SpeakerTextExporter, TranscriptJsonExporter
from transcriber.protocols import (
    DiarizedTranscript,
    SpeakerStats,
    SpeakerTurn,
    TranscriptionResult,
    TranscriptionSegment,
)


# ─────────────────────────────────────────────
#  Test helpers
# ─────────────────────────────────────────────


def _seg(start: float, end: float, text: str, speaker: str | None = None) -> TranscriptionSegment:
    return TranscriptionSegment(start=start, end=end, text=text, confidence=0.9, speaker=speaker)


def _dseg(speaker: str, start: float, end: float) -> DiarizationSegment:
    return DiarizationSegment(speaker=speaker, start=start, end=end)


def _transcript(*segs: TranscriptionSegment) -> TranscriptionResult:
    return TranscriptionResult(
        segments=list(segs),
        language="es",
        duration=segs[-1].end if segs else 0.0,
    )


def _turn(speaker: str, start: float, end: float, text: str) -> SpeakerTurn:
    return SpeakerTurn(
        speaker=speaker, start=start, end=end, text=text,
        segments=[_seg(start, end, text, speaker)],
    )


def _diarized(turns: list[SpeakerTurn], transcript: TranscriptionResult) -> DiarizedTranscript:
    stats = _compute_stats(turns)
    return DiarizedTranscript(turns=turns, speakers=stats, source=transcript)


# ─────────────────────────────────────────────
#  DiarizationSegment
# ─────────────────────────────────────────────


class TestDiarizationSegment:
    def test_duration(self) -> None:
        seg = _dseg("SPEAKER_00", 1.0, 4.5)
        assert seg.duration == pytest.approx(3.5)

    def test_str(self) -> None:
        seg = _dseg("INTERVIEWER", 0.0, 10.0)
        assert "INTERVIEWER" in str(seg)
        assert "0.00" in str(seg)


# ─────────────────────────────────────────────
#  SpeakerDiarizer role mapping
# ─────────────────────────────────────────────


class TestSpeakerDiarizerRoleMap:
    def _diarizer(self, flip: bool = False) -> SpeakerDiarizer:
        d = SpeakerDiarizer.__new__(SpeakerDiarizer)
        d._flip_roles = flip
        d._num_speakers = 2
        return d

    def test_less_speaking_time_is_interviewer(self) -> None:
        """Speaker with less total time → INTERVIEWER."""
        segs = [
            _dseg("SPEAKER_00", 0.0, 5.0),    # 5 s → less → INTERVIEWER
            _dseg("SPEAKER_01", 5.0, 55.0),   # 50 s → more → INTERVIEWEE
        ]
        role_map = self._diarizer()._build_role_map(segs)
        assert role_map["SPEAKER_00"] == "INTERVIEWER"
        assert role_map["SPEAKER_01"] == "INTERVIEWEE"

    def test_flip_roles_swaps_assignment(self) -> None:
        segs = [
            _dseg("SPEAKER_00", 0.0, 5.0),
            _dseg("SPEAKER_01", 5.0, 55.0),
        ]
        role_map = self._diarizer(flip=True)._build_role_map(segs)
        assert role_map["SPEAKER_00"] == "INTERVIEWEE"
        assert role_map["SPEAKER_01"] == "INTERVIEWER"

    def test_three_speakers_gets_extra_label(self) -> None:
        segs = [
            _dseg("A", 0.0, 10.0),
            _dseg("B", 10.0, 50.0),
            _dseg("C", 50.0, 200.0),
        ]
        role_map = self._diarizer()._build_role_map(segs)
        assert "INTERVIEWER" in role_map.values()
        assert "INTERVIEWEE" in role_map.values()
        assert "SPEAKER_2" in role_map.values()

    def test_empty_segments_returns_empty_map(self) -> None:
        role_map = self._diarizer()._build_role_map([])
        assert role_map == {}

    def test_no_token_raises(self) -> None:
        d = SpeakerDiarizer(hf_token="")
        d._token = ""
        with pytest.raises(ValueError, match="HuggingFace token"):
            d._ensure_pipeline()

    def test_missing_pyannote_raises(self) -> None:
        d = SpeakerDiarizer(hf_token="hf_fake")
        d._token = "hf_fake"
        with patch.dict("sys.modules", {"pyannote.audio": None}):
            with pytest.raises(ImportError, match="pyannote.audio"):
                d._ensure_pipeline()

    def test_hf_403_raises_actionable_permission_error(self) -> None:
        class FakeResponse:
            status_code = 403

        class FakeHubError(Exception):
            def __init__(self) -> None:
                super().__init__("403 Forbidden")
                self.response = FakeResponse()

        def _from_pretrained(model_name: str, token: str) -> object:
            raise FakeHubError()

        fake_pipeline = types.SimpleNamespace(from_pretrained=_from_pretrained)
        fake_module = types.SimpleNamespace(Pipeline=fake_pipeline)

        d = SpeakerDiarizer(hf_token="hf_fake")
        d._token = "hf_fake"
        with patch.dict("sys.modules", {"pyannote.audio": fake_module}):
            with pytest.raises(PermissionError, match="segmentation-3.0"):
                d._ensure_pipeline()


# ─────────────────────────────────────────────
#  SpeakerDiarizer output compatibility
# ─────────────────────────────────────────────


class TestSpeakerDiarizerExtractSegments:
    @staticmethod
    def _annotation(*rows: tuple[float, float, str]) -> object:
        class FakeAnnotation:
            def __init__(self, values: tuple[tuple[float, float, str], ...]) -> None:
                self._values = values

            def itertracks(self, *, yield_label: bool = False):
                assert yield_label is True
                for start, end, speaker in self._values:
                    yield types.SimpleNamespace(start=start, end=end), "_", speaker

        return FakeAnnotation(tuple(rows))

    def test_extract_segments_supports_legacy_annotation(self) -> None:
        diarization = self._annotation(
            (3.0, 4.0, "SPEAKER_01"),
            (0.0, 1.5, "SPEAKER_00"),
        )

        result = SpeakerDiarizer._extract_segments(diarization)

        assert [(seg.speaker, seg.start, seg.end) for seg in result] == [
            ("SPEAKER_00", 0.0, 1.5),
            ("SPEAKER_01", 3.0, 4.0),
        ]

    def test_extract_segments_prefers_exclusive_annotation_from_diarize_output(self) -> None:
        diarization = types.SimpleNamespace(
            speaker_diarization=self._annotation((0.0, 3.0, "SPEAKER_00")),
            exclusive_speaker_diarization=self._annotation(
                (0.0, 1.0, "SPEAKER_00"),
                (1.0, 3.0, "SPEAKER_01"),
            ),
        )

        result = SpeakerDiarizer._extract_segments(diarization)

        assert [(seg.speaker, seg.start, seg.end) for seg in result] == [
            ("SPEAKER_00", 0.0, 1.0),
            ("SPEAKER_01", 1.0, 3.0),
        ]

    def test_extract_segments_falls_back_to_primary_annotation(self) -> None:
        diarization = types.SimpleNamespace(
            speaker_diarization=self._annotation((2.0, 5.0, "SPEAKER_00")),
        )

        result = SpeakerDiarizer._extract_segments(diarization)

        assert [(seg.speaker, seg.start, seg.end) for seg in result] == [
            ("SPEAKER_00", 2.0, 5.0),
        ]


# ─────────────────────────────────────────────
#  Aligner — _dominant_speaker
# ─────────────────────────────────────────────


class TestDominantSpeaker:
    def test_full_coverage(self) -> None:
        dsegs = [_dseg("INTERVIEWER", 0.0, 10.0)]
        assert _dominant_speaker(2.0, 8.0, dsegs) == "INTERVIEWER"

    def test_majority_wins(self) -> None:
        dsegs = [
            _dseg("INTERVIEWER", 0.0, 3.0),   # 2 s overlap with [1,5]
            _dseg("INTERVIEWEE", 3.0, 10.0),  # 2 s overlap with [1,5] — tie → first listed
        ]
        # Segment [1.0, 5.0]: interviewer covers [1,3] = 2s, interviewee covers [3,5] = 2s
        # Equal overlap → whichever comes first in the dict wins (non-deterministic tie)
        result = _dominant_speaker(1.0, 5.0, dsegs)
        assert result in {"INTERVIEWER", "INTERVIEWEE"}

    def test_interviewee_more_overlap(self) -> None:
        dsegs = [
            _dseg("INTERVIEWER", 0.0, 1.0),   # 0.5 s overlap with [0.5, 5.0]
            _dseg("INTERVIEWEE", 1.0, 10.0),  # 4.0 s overlap with [0.5, 5.0]
        ]
        assert _dominant_speaker(0.5, 5.0, dsegs) == "INTERVIEWEE"

    def test_no_overlap_returns_none(self) -> None:
        dsegs = [_dseg("INTERVIEWER", 100.0, 200.0)]
        assert _dominant_speaker(0.0, 5.0, dsegs) is None

    def test_empty_diarization_returns_none(self) -> None:
        assert _dominant_speaker(0.0, 5.0, []) is None


# ─────────────────────────────────────────────
#  Aligner — _group_into_turns
# ─────────────────────────────────────────────


class TestGroupIntoTurns:
    def test_same_speaker_merged(self) -> None:
        segs = [
            _seg(0.0, 2.0, "uno", "INTERVIEWER"),
            _seg(2.0, 4.0, "dos", "INTERVIEWER"),
        ]
        turns = _group_into_turns(segs)
        assert len(turns) == 1
        assert turns[0].text == "uno dos"

    def test_speaker_change_creates_new_turn(self) -> None:
        segs = [
            _seg(0.0, 2.0, "pregunta", "INTERVIEWER"),
            _seg(2.0, 8.0, "respuesta", "INTERVIEWEE"),
        ]
        turns = _group_into_turns(segs)
        assert len(turns) == 2
        assert turns[0].speaker == "INTERVIEWER"
        assert turns[1].speaker == "INTERVIEWEE"

    def test_empty_returns_empty(self) -> None:
        assert _group_into_turns([]) == []

    def test_alternating_speakers(self) -> None:
        segs = [
            _seg(0, 1, "a", "INTERVIEWER"),
            _seg(1, 3, "b", "INTERVIEWEE"),
            _seg(3, 4, "c", "INTERVIEWER"),
            _seg(4, 8, "d", "INTERVIEWEE"),
        ]
        turns = _group_into_turns(segs)
        assert len(turns) == 4

    def test_turn_timestamps_correct(self) -> None:
        segs = [
            _seg(1.0, 3.0, "uno", "INTERVIEWEE"),
            _seg(3.0, 7.0, "dos", "INTERVIEWEE"),
        ]
        turns = _group_into_turns(segs)
        assert turns[0].start == pytest.approx(1.0)
        assert turns[0].end == pytest.approx(7.0)


# ─────────────────────────────────────────────
#  Aligner — full align()
# ─────────────────────────────────────────────


class TestAlign:
    def test_basic_two_speaker_alignment(self) -> None:
        transcript = _transcript(
            _seg(0.0, 5.0, "Buenos días"),
            _seg(5.0, 30.0, "Soy de Bogotá y nací en…"),
        )
        diarization = [
            _dseg("INTERVIEWER", 0.0, 5.5),
            _dseg("INTERVIEWEE", 5.5, 35.0),
        ]
        result = align(transcript, diarization)
        assert result.turns[0].speaker == "INTERVIEWER"
        assert result.turns[1].speaker == "INTERVIEWEE"

    def test_empty_diarization_labels_unknown(self) -> None:
        transcript = _transcript(_seg(0.0, 5.0, "test"))
        result = align(transcript, [])
        assert result.turns[0].speaker == "UNKNOWN"

    def test_fallback_carries_last_speaker(self) -> None:
        """Gap in diarization → last known speaker assigned."""
        transcript = _transcript(
            _seg(0.0, 5.0, "a"),
            _seg(10.0, 15.0, "b"),  # gap at 5–10 s in diarization
        )
        diarization = [_dseg("INTERVIEWER", 0.0, 5.0)]
        result = align(transcript, diarization)
        speakers = [t.speaker for t in result.turns]
        assert all(s == "INTERVIEWER" for s in speakers)

    def test_stats_computed_correctly(self) -> None:
        transcript = _transcript(
            _seg(0.0, 10.0, "pregunta interviewer"),
            _seg(10.0, 60.0, "respuesta larga del entrevistado y muchas palabras"),
        )
        diarization = [
            _dseg("INTERVIEWER", 0.0, 10.0),
            _dseg("INTERVIEWEE", 10.0, 60.0),
        ]
        result = align(transcript, diarization)
        assert result.speakers["INTERVIEWEE"].total_speech_s > result.speakers["INTERVIEWER"].total_speech_s
        assert result.speakers["INTERVIEWER"].turn_count == 1
        assert result.speakers["INTERVIEWEE"].turn_count == 1

    def test_source_path_preserved(self, tmp_path: Path) -> None:
        src = tmp_path / "interview.mp3"
        transcript = TranscriptionResult(
            segments=[_seg(0.0, 5.0, "test")],
            language="es",
            duration=5.0,
            source_path=src,
        )
        result = align(transcript, [_dseg("INTERVIEWER", 0.0, 5.0)])
        assert result.source.source_path == src


# ─────────────────────────────────────────────
#  DiarizedTranscript convenience props
# ─────────────────────────────────────────────


class TestDiarizedTranscript:
    def _make(self) -> DiarizedTranscript:
        turns = [
            _turn("INTERVIEWER", 0.0, 5.0, "pregunta"),
            _turn("INTERVIEWEE", 5.0, 40.0, "respuesta larga"),
            _turn("INTERVIEWER", 40.0, 45.0, "gracias"),
        ]
        transcript = _transcript(_seg(0, 45, "all"))
        return _diarized(turns, transcript)

    def test_interviewer_turns_filter(self) -> None:
        d = self._make()
        assert len(d.interviewer_turns) == 2

    def test_interviewee_turns_filter(self) -> None:
        d = self._make()
        assert len(d.interviewee_turns) == 1

    def test_full_text_has_labels(self) -> None:
        d = self._make()
        assert "[INTERVIEWER]" in d.full_text
        assert "[INTERVIEWEE]" in d.full_text


# ─────────────────────────────────────────────
#  TranscriptJsonExporter
# ─────────────────────────────────────────────


class TestTranscriptJsonExporter:
    def _make_diarized(self, tmp_path: Path) -> DiarizedTranscript:
        src = tmp_path / "interview.mp3"
        src.write_bytes(b"\x00")
        turns = [
            _turn("INTERVIEWER", 0.0, 5.0, "¿Cómo se llama?"),
            _turn("INTERVIEWEE", 5.0, 30.0, "Me llamo Ana García."),
        ]
        transcript = TranscriptionResult(
            segments=[_seg(0, 30, "test")],
            language="es",
            duration=30.0,
            source_path=src,
        )
        stats = _compute_stats(turns)
        return DiarizedTranscript(turns=turns, speakers=stats, source=transcript)

    def test_creates_file(self, tmp_path: Path) -> None:
        d = self._make_diarized(tmp_path)
        path = TranscriptJsonExporter().export(d, tmp_path / "out")
        assert path.suffix == ".json"
        assert "transcript" in path.name
        assert path.exists()

    def test_valid_json(self, tmp_path: Path) -> None:
        d = self._make_diarized(tmp_path)
        path = TranscriptJsonExporter().export(d, tmp_path / "out")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "turns" in data
        assert "speakers" in data
        assert "metadata" in data
        assert data["schema_version"] == "1.0"

    def test_turn_count_matches(self, tmp_path: Path) -> None:
        d = self._make_diarized(tmp_path)
        path = TranscriptJsonExporter().export(d, tmp_path / "out")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert len(data["turns"]) == 2

    def test_spanish_chars_preserved(self, tmp_path: Path) -> None:
        d = self._make_diarized(tmp_path)
        path = TranscriptJsonExporter().export(d, tmp_path / "out")
        content = path.read_text(encoding="utf-8")
        assert "¿Cómo se llama?" in content
        assert "García" in content

    def test_turn_has_required_fields(self, tmp_path: Path) -> None:
        d = self._make_diarized(tmp_path)
        path = TranscriptJsonExporter().export(d, tmp_path / "out")
        data = json.loads(path.read_text(encoding="utf-8"))
        for turn in data["turns"]:
            assert {"speaker", "start", "end", "text", "segments", "turn_index"} <= turn.keys()

    def test_speaker_stats_present(self, tmp_path: Path) -> None:
        d = self._make_diarized(tmp_path)
        path = TranscriptJsonExporter().export(d, tmp_path / "out")
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "INTERVIEWER" in data["speakers"]
        assert "INTERVIEWEE" in data["speakers"]
        assert "total_speech_s" in data["speakers"]["INTERVIEWER"]


# ─────────────────────────────────────────────
#  SpeakerTextExporter
# ─────────────────────────────────────────────


class TestSpeakerTextExporter:
    def _make_diarized(self, tmp_path: Path) -> DiarizedTranscript:
        turns = [
            _turn("INTERVIEWER", 0.0, 5.0, "¿Cómo se llama?"),
            _turn("INTERVIEWEE", 5.0, 30.0, "Me llamo Ana García."),
        ]
        transcript = TranscriptionResult(
            segments=[_seg(0, 30, "test")], language="es", duration=30.0
        )
        stats = _compute_stats(turns)
        return DiarizedTranscript(turns=turns, speakers=stats, source=transcript)

    def test_creates_file(self, tmp_path: Path) -> None:
        d = self._make_diarized(tmp_path)
        path = SpeakerTextExporter().export(d, tmp_path / "out")
        assert "transcript" in path.name
        assert path.suffix == ".txt"
        assert path.exists()

    def test_contains_speaker_labels(self, tmp_path: Path) -> None:
        d = self._make_diarized(tmp_path)
        path = SpeakerTextExporter().export(d, tmp_path / "out")
        content = path.read_text(encoding="utf-8")
        assert "[INTERVIEWER]" in content
        assert "[INTERVIEWEE]" in content

    def test_contains_timestamps(self, tmp_path: Path) -> None:
        d = self._make_diarized(tmp_path)
        path = SpeakerTextExporter().export(d, tmp_path / "out")
        content = path.read_text(encoding="utf-8")
        assert "0:00:00" in content  # turn at t=0

    def test_contains_text(self, tmp_path: Path) -> None:
        d = self._make_diarized(tmp_path)
        path = SpeakerTextExporter().export(d, tmp_path / "out")
        content = path.read_text(encoding="utf-8")
        assert "Ana García" in content

    def test_header_present(self, tmp_path: Path) -> None:
        d = self._make_diarized(tmp_path)
        path = SpeakerTextExporter().export(d, tmp_path / "out")
        content = path.read_text(encoding="utf-8")
        assert "Interview transcript" in content
        assert "Duration" in content
