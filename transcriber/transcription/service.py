"""
service.py — TranscriptionService (Single-Responsibility + Dependency-Inversion)
=================================================================================

Orchestrates the full pipeline:

  1. Normalise audio (non-WAV → 16 kHz mono WAV via AudioNormalizer).
  2. Optionally split into speech-safe chunks (AudioChunker + checkpoint/resume).
  3. Transcribe with Whisper (FasterWhisperTranscriber).
  4. Optionally run speaker diarization (SpeakerDiarizer) and align labels.
  5. Export via an injected ResultExporter.

Dependency injection
--------------------
Every collaborator — normalizer, exporter, diarizer, chunker — is injected
through the constructor so each can be replaced with a test double without
any monkey-patching.

Halt / resume
-------------
During chunked transcription, Ctrl+C or 'q' raises ``HaltException`` (not
a plain ``RuntimeError``) so ``transcribe_batch`` can distinguish a deliberate
pause from an engine crash.  Re-run the same command to resume.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from transcriber.protocols import (
    DiarizedTranscript,
    HaltException,
    ResultExporter,
    SpeechTranscriber,
    TranscriptionResult,
)

logger = logging.getLogger(__name__)


class TranscriptionService:
    """
    Orchestrates transcription with optional chunking, diarization, and export.

    Parameters
    ----------
    transcriber:
        Any object satisfying :class:`~transcriber.protocols.SpeechTranscriber`.
    exporter:
        Optional :class:`~transcriber.protocols.ResultExporter` for raw transcripts.
    diarized_exporter:
        Optional exporter for ``DiarizedTranscript`` objects
        (e.g. ``TranscriptJsonExporter`` or ``SpeakerTextExporter``).
    chunk_config:
        Optional :class:`~transcriber.audio.chunker.ChunkConfig`.
    checkpoint_dir:
        Directory for checkpoint files (defaults to audio file's parent).
    enable_halt:
        Register Ctrl+C / 'q' handlers (default True; set False in tests).
    diarizer:
        Optional :class:`~transcriber.audio.diarizer.SpeakerDiarizer`.
    normalizer:
        Optional :class:`~transcriber.audio.normalizer.AudioNormalizer`.
        When ``None`` (the default) one is created on demand.  Inject a
        stub in tests to bypass ffmpeg/pydub entirely.

    Examples
    --------
    Full pipeline — chunked, diarized, exported as structured JSON::

        from transcriber.audio.chunker import ChunkConfig
        from transcriber.audio.diarizer import SpeakerDiarizer
        from transcriber.output.exporters import TranscriptJsonExporter

        service = TranscriptionService(
            engine,
            diarizer=SpeakerDiarizer(hf_token="hf_..."),
            diarized_exporter=TranscriptJsonExporter(),
            chunk_config=ChunkConfig(target_duration_s=600),
        )
        diarized = service.transcribe_and_diarize(
            Path("audio/interview.mp3"),
            export_to=Path("output/interview"),
        )
        for turn in diarized.interviewee_turns:
            print(turn.text)
    """

    def __init__(
        self,
        transcriber: SpeechTranscriber,
        *,
        exporter: ResultExporter | None = None,
        diarized_exporter=None,
        chunk_config=None,
        checkpoint_dir: Path | None = None,
        enable_halt: bool = True,
        diarizer=None,
        normalizer=None,
    ) -> None:
        self._transcriber = transcriber
        self._exporter = exporter
        self._diarized_exporter = diarized_exporter
        self._chunk_config = chunk_config
        self._checkpoint_dir = checkpoint_dir
        self._enable_halt = enable_halt
        self._diarizer = diarizer
        self._normalizer = normalizer  # None → created lazily in _make_normalizer()

    # ── public API ────────────────────────────────────────────────────────────

    def transcribe_file(
        self,
        audio_path: Path,
        *,
        export_to: Path | None = None,
        keep_chunks: bool = False,
    ) -> TranscriptionResult:
        """
        Transcribe a single audio file.

        Normalizes non-WAV inputs, optionally chunks long files, and returns
        a ``TranscriptionResult``.  For speaker-attributed output use
        :meth:`transcribe_and_diarize` instead.

        Parameters
        ----------
        audio_path:
            Path to the audio file (any ffmpeg-supported format).
        export_to:
            Destination stem for the raw transcript export.
        keep_chunks:
            Preserve temporary chunk WAV files (useful for debugging).

        Raises
        ------
        FileNotFoundError
            If *audio_path* does not exist.
        HaltException
            If the user pauses a chunked job (checkpoint is preserved).
        ValueError
            If *export_to* is given but no exporter was injected.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        if self._chunk_config is not None:
            result = self._transcribe_chunked(audio_path, keep_chunks=keep_chunks)
        else:
            result = self._transcribe_direct(audio_path)

        self._log_summary(result)

        if export_to is not None:
            self._export(result, export_to)

        return result

    def transcribe_and_diarize(
        self,
        audio_path: Path,
        *,
        export_to: Path | None = None,
        keep_chunks: bool = False,
    ) -> DiarizedTranscript:
        """
        Transcribe and identify speakers (INTERVIEWER / INTERVIEWEE).

        Runs the full pipeline:
        transcription → diarization → alignment → optional export.

        Parameters
        ----------
        audio_path:
            Path to the audio file.
        export_to:
            Destination stem; the diarized exporter appends the extension.
        keep_chunks:
            Preserve temporary chunk WAV files.

        Returns
        -------
        DiarizedTranscript
            Speaker-attributed transcript with grouped turns.

        Raises
        ------
        ValueError
            If no diarizer was injected.
        """
        if self._diarizer is None:
            raise ValueError(
                "transcribe_and_diarize() requires a SpeakerDiarizer. "
                "Pass diarizer=SpeakerDiarizer(hf_token=...) to TranscriptionService."
            )

        from transcriber.audio.aligner import align  # noqa: PLC0415
        from transcriber.transcription.stage_checkpoint import (  # noqa: PLC0415
            DiarizationStageCheckpoint,
        )

        # Fail before the expensive Whisper pass if HuggingFace access is not ready.
        self._diarizer.validate_access()

        stage_checkpoint = DiarizationStageCheckpoint(
            audio_path,
            checkpoint_dir=self._checkpoint_dir,
        )

        # 1. Transcribe (with chunking / resume if configured), or reuse the
        #    completed transcript if diarization failed on a previous run.
        transcript = stage_checkpoint.load_transcript()
        if transcript is None:
            transcript = self.transcribe_file(audio_path, keep_chunks=keep_chunks)
            stage_checkpoint.save_transcript(transcript)
            logger.info(
                "Saved transcription stage checkpoint → '%s'",
                stage_checkpoint.checkpoint_path,
            )

        # 2. Diarize the *original* file (pyannote needs the full audio).
        normalizer = self._make_normalizer()
        prepared = normalizer.prepare(audio_path)
        try:
            diarization = self._diarizer.diarize(prepared.path)

            # 3. Align Whisper segments with diarization segments.
            logger.info("Aligning transcript with diarization …")
            diarized = align(transcript, diarization)
            logger.info("%s", diarized)

            # 4. Export.
            if export_to is not None and self._diarized_exporter is not None:
                saved = self._diarized_exporter.export(diarized, export_to)
                logger.info("Diarized transcript exported → '%s'", saved)

            stage_checkpoint.delete()
        except Exception:  # noqa: BLE001
            logger.warning(
                "Diarization did not finish. Transcription resume state preserved at '%s'.",
                stage_checkpoint.checkpoint_path,
            )
            raise
        finally:
            normalizer.cleanup(prepared)

        return diarized

    def transcribe_to_text(self, audio_path: Path) -> str:
        """Convenience wrapper — returns the plain transcript string only."""
        return self.transcribe_file(audio_path).full_text

    def transcribe_batch(
        self,
        audio_paths: list[Path],
        *,
        output_dir: Path | None = None,
        keep_chunks: bool = False,
    ) -> list[TranscriptionResult]:
        """
        Transcribe multiple audio files sequentially.

        ``HaltException`` stops the entire batch (preserving the checkpoint
        for the halted file).  All other exceptions skip the file and continue.

        Parameters
        ----------
        audio_paths:
            Ordered list of files to process.
        output_dir:
            Export destination directory (requires an exporter).
        keep_chunks:
            Preserve temporary chunk WAV files.

        Returns
        -------
        list[TranscriptionResult]
            Successful results only; halted or failed files are logged.
        """
        results: list[TranscriptionResult] = []

        for idx, path in enumerate(audio_paths, start=1):
            logger.info("[%d/%d] Processing '%s'", idx, len(audio_paths), path.name)
            try:
                export_to = (output_dir / path.stem) if output_dir else None
                result = self.transcribe_file(
                    path, export_to=export_to, keep_chunks=keep_chunks
                )
                results.append(result)
            except HaltException as exc:
                # Deliberate pause — stop the entire batch and preserve checkpoint.
                logger.warning("Batch halted at '%s': %s", path.name, exc)
                break
            except Exception:  # noqa: BLE001
                logger.exception("Failed to transcribe '%s' — skipping.", path.name)

        logger.info("Batch done — %d/%d succeeded.", len(results), len(audio_paths))
        return results

    # ── direct (non-chunked) path ─────────────────────────────────────────────

    def _transcribe_direct(self, audio_path: Path) -> TranscriptionResult:
        """
        Normalize → transcribe → ensure source_path points to the original file.

        The normalizer may produce a temporary WAV at a different path.
        We always stamp ``source_path`` with the *original* caller-supplied
        path so downstream code and exporters see the real file name.
        """
        normalizer = self._make_normalizer()
        prepared = normalizer.prepare(audio_path)
        try:
            logger.info("Transcribing '%s' …", audio_path.name)
            result = self._transcriber.transcribe(prepared.path)
        finally:
            normalizer.cleanup(prepared)

        # Only stamp source_path when the normalizer actually created a temp
        # file (cleanup_required=True).  When the file was passed through
        # unchanged, the model's result is returned as-is — preserving object
        # identity for callers that rely on it (e.g. tests using "is").
        if prepared.cleanup_required:
            result = TranscriptionResult(
                segments=result.segments,
                language=result.language,
                duration=result.duration,
                source_path=audio_path.resolve(),
            )
        return result

    # ── chunked path ──────────────────────────────────────────────────────────

    def _transcribe_chunked(
        self, audio_path: Path, *, keep_chunks: bool = False
    ) -> TranscriptionResult:
        """
        Core loop: split → checkpoint → transcribe each chunk → reassemble.

        Flow
        ----
        ::

            split audio into VAD-safe chunks
                │
            load checkpoint (skips already-completed chunks on resume)
                │
            ┌───▼──────────────────────────────────────────────┐
            │  for each chunk:                                 │
            │    halt requested? ──yes──► save checkpoint,    │
            │         │                   raise HaltException  │
            │        no                                        │
            │    already done? ──yes──► skip                  │
            │        no                                        │
            │    transcribe chunk ──► save to checkpoint       │
            └──────────────────────────────────────────────────┘
                │
            all done → reassemble → delete checkpoint
        """
        from transcriber.audio.checkpoint import CheckpointManager  # noqa: PLC0415
        from transcriber.audio.chunker import AudioChunker          # noqa: PLC0415
        from transcriber.audio.halt import HaltController           # noqa: PLC0415
        from transcriber.audio.reassembler import reassemble        # noqa: PLC0415

        # ── 1. Split ─────────────────────────────────────────────────────────
        chunker = AudioChunker(self._chunk_config)
        chunks = chunker.split(audio_path, keep_chunks=keep_chunks)

        if len(chunks) == 1:
            logger.info("File fits in one chunk — transcribing directly.")
            result = self._transcriber.transcribe(chunks[0].path)
            if not keep_chunks:
                _delete_chunk(chunks[0])
            return result

        # ── 2. Checkpoint ─────────────────────────────────────────────────────
        checkpoint = CheckpointManager(audio_path, checkpoint_dir=self._checkpoint_dir)
        resumed = checkpoint.load(total_chunks=len(chunks))
        if resumed:
            logger.info(
                "Resuming — %d/%d chunks done, %d remaining.",
                checkpoint.completed_count, len(chunks),
                len(chunks) - checkpoint.completed_count,
            )

        # ── 3. Halt controller ────────────────────────────────────────────────
        halt = HaltController(prompt_keyboard=self._enable_halt)
        if self._enable_halt:
            halt.install()

        halted = False
        t_total = time.perf_counter()

        try:
            for chunk in chunks:
                if halt.should_halt():
                    halted = True
                    break

                if checkpoint.is_done(chunk):
                    logger.info(
                        "  [%d/%d] Skipping (already done).",
                        chunk.index + 1, len(chunks),
                    )
                    continue

                logger.info(
                    "  [%d/%d] Chunk %.1f–%.1f min (%.0f s) …",
                    chunk.index + 1, len(chunks),
                    chunk.start_s / 60, chunk.end_s / 60, chunk.duration_s,
                )
                t0 = time.perf_counter()
                chunk_result = self._transcriber.transcribe(chunk.path)
                elapsed = time.perf_counter() - t0
                ratio = chunk.duration_s / elapsed if elapsed > 0 else 0

                logger.info(
                    "  [%d/%d] Done in %.0f s (%.1fx real-time, %d segment(s))",
                    chunk.index + 1, len(chunks),
                    elapsed, ratio, len(chunk_result.segments),
                )
                checkpoint.save_chunk(chunk, chunk_result)
                logger.info(
                    "  [%d/%d] %s",
                    chunk.index + 1, len(chunks), checkpoint.summary(),
                )

                if not keep_chunks:
                    _delete_chunk(chunk)

        finally:
            if self._enable_halt:
                halt.uninstall()

        # ── 4. Handle halt ────────────────────────────────────────────────────
        if halted:
            logger.warning(
                "Job paused at chunk %d/%d. Checkpoint: %s",
                checkpoint.completed_count, len(chunks),
                checkpoint.checkpoint_path,
            )
            raise HaltException(
                f"Paused after {checkpoint.completed_count}/{len(chunks)} chunks "
                f"({halt.halt_reason}). Re-run the same command to resume."
            )

        # ── 5. Reassemble ─────────────────────────────────────────────────────
        total_elapsed = time.perf_counter() - t_total
        logger.info(
            "All %d chunks done in %.1f min. Reassembling …",
            len(chunks), total_elapsed / 60,
        )

        all_results = checkpoint.all_results_in_order(chunks)
        merged = reassemble(all_results, chunks, source_path=audio_path)
        checkpoint.delete()
        return merged

    # ── private helpers ───────────────────────────────────────────────────────

    def _make_normalizer(self):  # type: ignore[return]
        """Return the injected normalizer, or create a default one."""
        if self._normalizer is not None:
            return self._normalizer
        from transcriber.audio.normalizer import AudioNormalizer  # noqa: PLC0415
        return AudioNormalizer()

    def _export(self, result: TranscriptionResult, destination: Path) -> None:
        if self._exporter is None:
            raise ValueError(
                "export_to was provided but no exporter was injected into TranscriptionService."
            )
        saved_path = self._exporter.export(result, destination)
        logger.info("Exported → '%s'", saved_path)

    @staticmethod
    def _log_summary(result: TranscriptionResult) -> None:
        logger.info(
            "Transcription complete | language=%s | duration=%.1f min | "
            "segments=%d | avg_confidence=%.0f%%",
            result.language, result.duration / 60,
            len(result.segments), result.average_confidence * 100,
        )


def _delete_chunk(chunk) -> None:  # type: ignore[no-untyped-def]
    """Silently remove a temporary chunk WAV file."""
    try:
        chunk.path.unlink(missing_ok=True)
        logger.debug("Deleted chunk: %s", chunk.path.name)
    except OSError:
        logger.warning("Could not delete chunk: %s", chunk.path)
