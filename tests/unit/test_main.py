from __future__ import annotations

from pathlib import Path

from transcriber.main import _collect_audio_files, _resolve_audio_path


def test_resolve_audio_path_accepts_project_relative_fallback(tmp_path: Path, monkeypatch) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    real_file = audio_dir / "clip.m4a"
    real_file.write_bytes(b"\x00")

    monkeypatch.chdir(tmp_path)

    resolved = _resolve_audio_path(Path("/audio/clip.m4a"))

    assert resolved == real_file


def test_resolve_audio_path_matches_normalized_filename(tmp_path: Path, monkeypatch) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    real_file = audio_dir / "Entrevista_Natalia _peliroja_.m4a"
    real_file.write_bytes(b"\x00")

    monkeypatch.chdir(tmp_path)

    resolved = _resolve_audio_path(Path("/audio/Entrevista_Natalia_peliroja_.m4a"))

    assert resolved == real_file


def test_collect_audio_files_returns_resolved_match(tmp_path: Path, monkeypatch) -> None:
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    real_file = audio_dir / "Entrevista_Natalia _peliroja_.m4a"
    real_file.write_bytes(b"\x00")

    monkeypatch.chdir(tmp_path)

    collected = _collect_audio_files(Path("/audio/Entrevista_Natalia_peliroja_.m4a"), batch=False)

    assert collected == [real_file]
