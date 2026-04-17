"""
test_service.py — Tests for TranscriptionService
=================================================

All tests use mock objects — no real model, ffmpeg, or pydub needed.

Key fixture: ``mock_normalizer``
    Injected into every ``TranscriptionService`` constructor so that
    ``_transcribe_direct`` never tries to decode audio.  This preserves
    the invariant that ``transcribe(audio_file)`` is called with the
    exact path the test supplied, making ``result is sample_result``
    identity assertions safe.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from transcriber.protocols import (
    HaltException,
    TranscriptionResult,
    TranscriptionSegment,
)
from transcriber.transcription.service import TranscriptionService


# ─────────────────────────────────────────────
#  transcribe_file
# ─────────────────────────────────────────────


class TestTranscribeFile:
    def test_returns_result_from_engine(
        self,
        mock_transcriber: MagicMock,
        mock_normalizer: MagicMock,
        sample_result: TranscriptionResult,
        audio_file: Path,
    ) -> None:
        service = TranscriptionService(mock_transcriber, normalizer=mock_normalizer)
        result = service.transcribe_file(audio_file)

        assert result is sample_result
        mock_transcriber.transcribe.assert_called_once_with(audio_file)

    def test_propagates_file_not_found(
        self,
        mock_transcriber: MagicMock,
        mock_normalizer: MagicMock,
        tmp_path: Path,
    ) -> None:
        mock_transcriber.transcribe.side_effect = FileNotFoundError("missing")
        service = TranscriptionService(mock_transcriber, normalizer=mock_normalizer)

        with pytest.raises(FileNotFoundError):
            service.transcribe_file(tmp_path / "ghost.wav")

    def test_export_called_when_exporter_and_dest_given(
        self,
        mock_transcriber: MagicMock,
        mock_normalizer: MagicMock,
        sample_result: TranscriptionResult,
        audio_file: Path,
        tmp_path: Path,
    ) -> None:
        mock_exporter = MagicMock()
        mock_exporter.export.return_value = tmp_path / "out.txt"

        service = TranscriptionService(
            mock_transcriber, exporter=mock_exporter, normalizer=mock_normalizer
        )
        dest = tmp_path / "transcript"
        service.transcribe_file(audio_file, export_to=dest)

        mock_exporter.export.assert_called_once_with(sample_result, dest)

    def test_export_not_called_without_dest(
        self,
        mock_transcriber: MagicMock,
        mock_normalizer: MagicMock,
        audio_file: Path,
        tmp_path: Path,
    ) -> None:
        mock_exporter = MagicMock()
        service = TranscriptionService(
            mock_transcriber, exporter=mock_exporter, normalizer=mock_normalizer
        )
        service.transcribe_file(audio_file)  # no export_to

        mock_exporter.export.assert_not_called()

    def test_raises_if_export_to_given_but_no_exporter(
        self,
        mock_transcriber: MagicMock,
        mock_normalizer: MagicMock,
        audio_file: Path,
        tmp_path: Path,
    ) -> None:
        service = TranscriptionService(mock_transcriber, normalizer=mock_normalizer)

        with pytest.raises(ValueError, match="exporter"):
            service.transcribe_file(audio_file, export_to=tmp_path / "out")

    def test_normalizes_non_wav_before_direct_transcription(
        self,
        mock_transcriber: MagicMock,
        audio_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        When the source is not WAV, the normalizer produces a temp WAV path
        and the engine receives *that path*, not the original.
        """
        prepared_path = audio_file.with_suffix(".wav")
        prepare_mock = MagicMock(
            return_value=MagicMock(path=prepared_path, cleanup_required=True)
        )
        cleanup_mock = MagicMock()
        normalizer_mock = MagicMock(prepare=prepare_mock, cleanup=cleanup_mock)

        service = TranscriptionService(mock_transcriber, normalizer=normalizer_mock)
        service.transcribe_file(audio_file)

        prepare_mock.assert_called_once_with(audio_file)
        mock_transcriber.transcribe.assert_called_once_with(prepared_path)
        cleanup_mock.assert_called_once()

    def test_preserves_original_source_path_after_normalization(
        self,
        mock_transcriber: MagicMock,
        audio_file: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """
        Even when the normalizer creates a temp WAV, the returned result
        must carry source_path pointing to the *original* file.
        """
        prepared_path = audio_file.with_suffix(".converted.wav")
        # Return a result that has the temp path as source.
        mock_transcriber.transcribe.return_value = TranscriptionResult(
            segments=[],
            language="es",
            duration=1.0,
            source_path=prepared_path,
        )
        normalizer_mock = MagicMock(
            prepare=MagicMock(
                return_value=MagicMock(path=prepared_path, cleanup_required=True)
            ),
            cleanup=MagicMock(),
        )

        service = TranscriptionService(mock_transcriber, normalizer=normalizer_mock)
        result = service.transcribe_file(audio_file)

        assert result.source_path == audio_file.resolve()


# ─────────────────────────────────────────────
#  transcribe_to_text
# ─────────────────────────────────────────────


class TestTranscribeToText:
    def test_returns_full_text_string(
        self,
        mock_transcriber: MagicMock,
        mock_normalizer: MagicMock,
        sample_result: TranscriptionResult,
        audio_file: Path,
    ) -> None:
        service = TranscriptionService(mock_transcriber, normalizer=mock_normalizer)
        text = service.transcribe_to_text(audio_file)

        assert isinstance(text, str)
        assert text == sample_result.full_text


# ─────────────────────────────────────────────
#  transcribe_batch
# ─────────────────────────────────────────────


class TestTranscribeBatch:
    def test_processes_all_files(
        self,
        mock_transcriber: MagicMock,
        mock_normalizer: MagicMock,
        sample_result: TranscriptionResult,
        tmp_path: Path,
    ) -> None:
        files = [tmp_path / f"audio_{i}.wav" for i in range(3)]
        for f in files:
            f.write_bytes(b"\x00")

        service = TranscriptionService(mock_transcriber, normalizer=mock_normalizer)
        results = service.transcribe_batch(files)

        assert len(results) == 3
        assert mock_transcriber.transcribe.call_count == 3

    def test_failed_file_is_skipped(
        self,
        mock_transcriber: MagicMock,
        mock_normalizer: MagicMock,
        sample_result: TranscriptionResult,
        tmp_path: Path,
    ) -> None:
        files = [tmp_path / f"audio_{i}.wav" for i in range(3)]
        for f in files:
            f.write_bytes(b"\x00")

        # Second call fails with a plain OSError — should be skipped.
        mock_transcriber.transcribe.side_effect = [
            sample_result,
            OSError("disk read error"),
            sample_result,
        ]

        service = TranscriptionService(mock_transcriber, normalizer=mock_normalizer)
        results = service.transcribe_batch(files)

        assert len(results) == 2  # only successful ones returned

    def test_halt_exception_stops_batch(
        self,
        mock_transcriber: MagicMock,
        mock_normalizer: MagicMock,
        sample_result: TranscriptionResult,
        tmp_path: Path,
    ) -> None:
        """HaltException aborts the whole batch, unlike a plain engine error."""
        files = [tmp_path / f"audio_{i}.wav" for i in range(3)]
        for f in files:
            f.write_bytes(b"\x00")

        mock_transcriber.transcribe.side_effect = [
            sample_result,
            HaltException("Ctrl+C"),  # second file halted
            sample_result,
        ]

        service = TranscriptionService(mock_transcriber, normalizer=mock_normalizer)
        results = service.transcribe_batch(files)

        # Only the first file's result is returned; batch stops on HaltException.
        assert len(results) == 1

    def test_batch_with_output_dir_calls_export(
        self,
        mock_transcriber: MagicMock,
        mock_normalizer: MagicMock,
        sample_result: TranscriptionResult,
        tmp_path: Path,
    ) -> None:
        files = [tmp_path / f"clip_{i}.wav" for i in range(2)]
        for f in files:
            f.write_bytes(b"\x00")

        mock_exporter = MagicMock()
        mock_exporter.export.return_value = tmp_path / "x.txt"

        service = TranscriptionService(
            mock_transcriber, exporter=mock_exporter, normalizer=mock_normalizer
        )
        output_dir = tmp_path / "results"
        service.transcribe_batch(files, output_dir=output_dir)

        assert mock_exporter.export.call_count == 2

    def test_empty_list_returns_empty(
        self,
        mock_transcriber: MagicMock,
        mock_normalizer: MagicMock,
    ) -> None:
        service = TranscriptionService(mock_transcriber, normalizer=mock_normalizer)
        results = service.transcribe_batch([])
        assert results == []
        mock_transcriber.transcribe.assert_not_called()


# ─────────────────────────────────────────────
#  transcribe_and_diarize
# ─────────────────────────────────────────────


class TestTranscribeAndDiarize:
    def test_raises_without_diarizer(
        self,
        mock_transcriber: MagicMock,
        mock_normalizer: MagicMock,
        audio_file: Path,
    ) -> None:
        service = TranscriptionService(mock_transcriber, normalizer=mock_normalizer)
        with pytest.raises(ValueError, match="SpeakerDiarizer"):
            service.transcribe_and_diarize(audio_file)

    def test_calls_diarizer_and_returns_diarized_transcript(
        self,
        mock_transcriber: MagicMock,
        mock_normalizer: MagicMock,
        sample_result: TranscriptionResult,
        audio_file: Path,
        tmp_path: Path,
    ) -> None:
        from transcriber.audio.diarizer import DiarizationSegment
        from transcriber.protocols import DiarizedTranscript

        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = [
            DiarizationSegment("INTERVIEWER", 0.0, 3.5),
            DiarizationSegment("INTERVIEWEE", 3.5, 7.2),
        ]

        service = TranscriptionService(
            mock_transcriber,
            normalizer=mock_normalizer,
            diarizer=mock_diarizer,
        )
        result = service.transcribe_and_diarize(audio_file)

        assert isinstance(result, DiarizedTranscript)
        mock_diarizer.diarize.assert_called_once()
        speakers = {t.speaker for t in result.turns}
        assert "INTERVIEWER" in speakers
        assert "INTERVIEWEE" in speakers

    def test_diarized_exporter_called_when_export_to_given(
        self,
        mock_transcriber: MagicMock,
        mock_normalizer: MagicMock,
        audio_file: Path,
        tmp_path: Path,
    ) -> None:
        from transcriber.audio.diarizer import DiarizationSegment

        mock_diarizer = MagicMock()
        mock_diarizer.diarize.return_value = [
            DiarizationSegment("INTERVIEWER", 0.0, 3.5),
        ]
        mock_diarized_exporter = MagicMock()
        mock_diarized_exporter.export.return_value = tmp_path / "out.transcript.json"

        service = TranscriptionService(
            mock_transcriber,
            normalizer=mock_normalizer,
            diarizer=mock_diarizer,
            diarized_exporter=mock_diarized_exporter,
        )
        dest = tmp_path / "interview"
        service.transcribe_and_diarize(audio_file, export_to=dest)

        mock_diarized_exporter.export.assert_called_once()

    def test_validates_access_before_transcription(
        self,
        mock_transcriber: MagicMock,
        mock_normalizer: MagicMock,
        audio_file: Path,
    ) -> None:
        mock_diarizer = MagicMock()
        mock_diarizer.validate_access.side_effect = PermissionError("denied")

        service = TranscriptionService(
            mock_transcriber,
            normalizer=mock_normalizer,
            diarizer=mock_diarizer,
        )

        with pytest.raises(PermissionError, match="denied"):
            service.transcribe_and_diarize(audio_file)

        mock_transcriber.transcribe.assert_not_called()

    def test_failed_diarization_preserves_stage_checkpoint(
        self,
        mock_transcriber: MagicMock,
        mock_normalizer: MagicMock,
        audio_file: Path,
        tmp_path: Path,
    ) -> None:
        from transcriber.audio.diarizer import DiarizationSegment
        from transcriber.protocols import DiarizedTranscript

        checkpoint_path = tmp_path / f".{audio_file.stem}.diarization.checkpoint.json"
        mock_diarizer = MagicMock()
        mock_diarizer.diarize.side_effect = [
            RuntimeError("403 Forbidden"),
            [DiarizationSegment("INTERVIEWER", 0.0, 7.2)],
        ]

        service = TranscriptionService(
            mock_transcriber,
            normalizer=mock_normalizer,
            diarizer=mock_diarizer,
            checkpoint_dir=tmp_path,
        )

        with pytest.raises(RuntimeError, match="403 Forbidden"):
            service.transcribe_and_diarize(audio_file)

        assert checkpoint_path.exists()
        assert mock_transcriber.transcribe.call_count == 1

        result = service.transcribe_and_diarize(audio_file)

        assert isinstance(result, DiarizedTranscript)
        assert mock_transcriber.transcribe.call_count == 1
        assert not checkpoint_path.exists()
