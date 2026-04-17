"""
test_exporters.py — Tests for all ResultExporter implementations
================================================================
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from transcriber.output.exporters import (
    JsonExporter,
    PlainTextExporter,
    SrtExporter,
    _seconds_to_srt_time,
    exporter_for,
)
from transcriber.protocols import TranscriptionResult, TranscriptionSegment


# ─────────────────────────────────────────────
#  Helper tests
# ─────────────────────────────────────────────


class TestSecondsToSrtTime:
    def test_zero(self) -> None:
        assert _seconds_to_srt_time(0.0) == "00:00:00,000"

    def test_minutes_and_seconds(self) -> None:
        assert _seconds_to_srt_time(83.456) == "00:01:23,456"

    def test_hours(self) -> None:
        assert _seconds_to_srt_time(3661.0) == "01:01:01,000"

    def test_milliseconds_precision(self) -> None:
        assert _seconds_to_srt_time(0.001) == "00:00:00,001"


# ─────────────────────────────────────────────
#  PlainTextExporter
# ─────────────────────────────────────────────


class TestPlainTextExporter:
    def test_creates_txt_file(
        self, sample_result: TranscriptionResult, tmp_path: Path
    ) -> None:
        dest = tmp_path / "out"
        path = PlainTextExporter().export(sample_result, dest)

        assert path.suffix == ".txt"
        assert path.exists()

    def test_content_matches_full_text(
        self, sample_result: TranscriptionResult, tmp_path: Path
    ) -> None:
        dest = tmp_path / "out"
        path = PlainTextExporter().export(sample_result, dest)

        assert path.read_text(encoding="utf-8") == sample_result.full_text

    def test_forces_txt_suffix(
        self, sample_result: TranscriptionResult, tmp_path: Path
    ) -> None:
        dest = tmp_path / "out.srt"  # wrong extension
        path = PlainTextExporter().export(sample_result, dest)
        assert path.suffix == ".txt"

    def test_creates_parent_dirs(
        self, sample_result: TranscriptionResult, tmp_path: Path
    ) -> None:
        dest = tmp_path / "nested" / "deep" / "out"
        PlainTextExporter().export(sample_result, dest)
        assert (tmp_path / "nested" / "deep").is_dir()


# ─────────────────────────────────────────────
#  JsonExporter
# ─────────────────────────────────────────────


class TestJsonExporter:
    def test_creates_json_file(
        self, sample_result: TranscriptionResult, tmp_path: Path
    ) -> None:
        dest = tmp_path / "out"
        path = JsonExporter().export(sample_result, dest)

        assert path.suffix == ".json"
        assert path.exists()

    def test_valid_json_structure(
        self, sample_result: TranscriptionResult, tmp_path: Path
    ) -> None:
        dest = tmp_path / "out"
        path = JsonExporter().export(sample_result, dest)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["language"] == "es"
        assert isinstance(data["segments"], list)
        assert len(data["segments"]) == len(sample_result.segments)

    def test_segments_have_required_keys(
        self, sample_result: TranscriptionResult, tmp_path: Path
    ) -> None:
        dest = tmp_path / "out"
        path = JsonExporter().export(sample_result, dest)

        data = json.loads(path.read_text(encoding="utf-8"))
        for seg in data["segments"]:
            assert {"start", "end", "text", "confidence"} <= seg.keys()

    def test_average_confidence_present(
        self, sample_result: TranscriptionResult, tmp_path: Path
    ) -> None:
        dest = tmp_path / "out"
        path = JsonExporter().export(sample_result, dest)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert "average_confidence" in data
        assert 0.0 <= data["average_confidence"] <= 1.0

    def test_utf8_content_preserved(self, tmp_path: Path) -> None:
        """Spanish characters must survive JSON round-trip."""
        seg = TranscriptionSegment(
            start=0.0, end=1.0, text="¿Cómo estás, señor?", confidence=0.9
        )
        result = TranscriptionResult(segments=[seg], language="es", duration=1.0)
        dest = tmp_path / "out"
        path = JsonExporter().export(result, dest)

        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["segments"][0]["text"] == "¿Cómo estás, señor?"


# ─────────────────────────────────────────────
#  SrtExporter
# ─────────────────────────────────────────────


class TestSrtExporter:
    def test_creates_srt_file(
        self, sample_result: TranscriptionResult, tmp_path: Path
    ) -> None:
        dest = tmp_path / "out"
        path = SrtExporter().export(sample_result, dest)

        assert path.suffix == ".srt"
        assert path.exists()

    def test_srt_has_correct_entry_count(
        self, sample_result: TranscriptionResult, tmp_path: Path
    ) -> None:
        dest = tmp_path / "out"
        path = SrtExporter().export(sample_result, dest)

        content = path.read_text(encoding="utf-8")
        # Each entry starts with a sequence number on its own line.
        entries = [line for line in content.splitlines() if line.strip().isdigit()]
        assert len(entries) == len(sample_result.segments)

    def test_srt_contains_arrow_separator(
        self, sample_result: TranscriptionResult, tmp_path: Path
    ) -> None:
        dest = tmp_path / "out"
        path = SrtExporter().export(sample_result, dest)
        assert "-->" in path.read_text(encoding="utf-8")

    def test_empty_result_produces_empty_file(self, tmp_path: Path) -> None:
        result = TranscriptionResult(segments=[], language="es", duration=0.0)
        dest = tmp_path / "out"
        path = SrtExporter().export(result, dest)
        assert path.read_text(encoding="utf-8").strip() == ""


# ─────────────────────────────────────────────
#  Factory
# ─────────────────────────────────────────────


class TestExporterFor:
    def test_returns_correct_types(self) -> None:
        assert isinstance(exporter_for(".txt"), PlainTextExporter)
        assert isinstance(exporter_for(".json"), JsonExporter)
        assert isinstance(exporter_for(".srt"), SrtExporter)

    def test_case_insensitive(self) -> None:
        assert isinstance(exporter_for(".TXT"), PlainTextExporter)
        assert isinstance(exporter_for(".JSON"), JsonExporter)

    def test_unknown_format_raises(self) -> None:
        with pytest.raises(ValueError, match="No exporter"):
            exporter_for(".docx")
