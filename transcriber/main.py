"""
main.py — CLI entry-point (Composition Root)
============================================

Usage
-----
Single file, transcribe only::

    python -m transcriber.main audio/interview.mp3

With chunking (recommended for files > 20 min)::

    python -m transcriber.main audio/interview.mp3 --chunk --format srt

With diarization (identify INTERVIEWER / INTERVIEWEE)::

    python -m transcriber.main audio/interview.mp3 --diarize

Full pipeline — chunked + diarized + exported as JSON turns::

    python -m transcriber.main audio/interview.mp3 \\
        --chunk --chunk-minutes 10 \\
        --diarize --diarize-format transcript.json

Batch directory::

    python -m transcriber.main audio/ --batch --chunk --format json

Override model via env var::

    TRANSCRIBER_MODEL_SIZE=medium python -m transcriber.main audio/file.mp3 --chunk
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transcriber.config import TranscriberConfig
from transcriber.logging_setup import configure_logging
from transcriber.models.faster_whisper_model import FasterWhisperTranscriber
from transcriber.output.exporters import exporter_for
from transcriber.transcription.service import TranscriptionService

AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".opus"}
)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe Spanish audio using faster-whisper.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", type=Path,
                        help="Audio file or directory (use with --batch).")
    parser.add_argument("--format", choices=["txt", "json", "srt"], default=None,
                        help="Raw transcript export format.")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for exported files.")

    # ── chunking ──────────────────────────────────────────────────────────────
    chunk_group = parser.add_argument_group("chunking (recommended for files > 20 min)")
    chunk_group.add_argument("--chunk", action="store_true",
                             help="Split audio into speech-safe chunks before transcribing.")
    chunk_group.add_argument("--chunk-minutes", type=float, default=10.0, metavar="N",
                             help="Target chunk length in minutes (default: 10).")
    chunk_group.add_argument("--silence-ms", type=int, default=800, metavar="MS",
                             help="Min silence length for a cut point in ms (default: 800).")
    chunk_group.add_argument("--keep-chunks", action="store_true",
                             help="Do not delete temporary chunk WAV files.")

    # ── diarization ───────────────────────────────────────────────────────────
    diar_group = parser.add_argument_group(
        "diarization (identify INTERVIEWER / INTERVIEWEE)"
    )
    diar_group.add_argument(
        "--diarize", action="store_true",
        help=(
            "Run speaker diarization after transcription. "
            "Requires HF_TOKEN env var and pyannote.audio installed."
        ),
    )
    diar_group.add_argument(
        "--diarize-format",
        choices=["transcript.json", "transcript.txt"],
        default="transcript.json",
        metavar="FMT",
        help=(
            "Output format for diarized transcript. "
            "'transcript.json' (default) for auto-coding pipelines; "
            "'transcript.txt' for NVivo / ATLAS.ti import."
        ),
    )
    diar_group.add_argument(
        "--flip-roles", action="store_true",
        help=(
            "Swap INTERVIEWER / INTERVIEWEE assignment. "
            "Use when the interviewee speaks less than the interviewer."
        ),
    )

    # ── misc ─────────────────────────────────────────────────────────────────
    parser.add_argument("--batch", action="store_true",
                        help="Transcribe all audio files in the INPUT directory.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already have an output file (use with --batch).")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def _collect_audio_files(path: Path, *, batch: bool) -> list[Path]:
    if batch:
        if not path.is_dir():
            print(f"ERROR: --batch requires a directory, got: {path}", file=sys.stderr)
            sys.exit(1)
        files = sorted(f for f in path.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS)
        if not files:
            print(f"No audio files found in '{path}'", file=sys.stderr)
            sys.exit(1)
        return files
    resolved = _resolve_audio_path(path)
    if resolved is None:
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(1)
    return [resolved]


def _resolve_audio_path(path: Path) -> Path | None:
    """Resolve common CLI path mistakes without forcing the user to rename files."""
    if path.is_file():
        return path

    candidates = [path]
    if path.is_absolute():
        candidates.append(Path.cwd() / str(path).lstrip("/"))

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    search_parent = _resolve_existing_parent(path)
    if search_parent is None:
        return None

    target_key = _normalize_filename_key(path.name)
    matches = [
        entry for entry in search_parent.iterdir()
        if entry.is_file()
        and entry.suffix.lower() in AUDIO_EXTENSIONS
        and _normalize_filename_key(entry.name) == target_key
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _resolve_existing_parent(path: Path) -> Path | None:
    candidates = [path.parent]
    if path.is_absolute():
        candidates.append(Path.cwd() / str(path.parent).lstrip("/"))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _normalize_filename_key(name: str) -> str:
    return "".join(ch for ch in name.casefold() if ch.isalnum())


def _print_result(result: object) -> None:
    from transcriber.protocols import DiarizedTranscript, TranscriptionResult
    if isinstance(result, DiarizedTranscript):
        _print_diarized(result)
    elif isinstance(result, TranscriptionResult):
        _print_transcript(result)


def _print_transcript(result: object) -> None:
    from transcriber.protocols import TranscriptionResult
    if not isinstance(result, TranscriptionResult):
        return
    print("\n" + "─" * 60)
    print(f"  Language   : {result.language}")
    print(f"  Duration   : {result.duration / 60:.1f} min")
    print(f"  Segments   : {len(result.segments)}")
    print(f"  Confidence : {result.average_confidence:.0%}")
    print("─" * 60)
    print(f"\n{result.full_text}\n")


def _print_diarized(result: object) -> None:
    from transcriber.protocols import DiarizedTranscript
    if not isinstance(result, DiarizedTranscript):
        return
    print("\n" + "═" * 60)
    print(f"  Language   : {result.source.language}")
    print(f"  Duration   : {result.source.duration / 60:.1f} min")
    print(f"  Confidence : {result.source.average_confidence:.0%}")
    for lbl, s in result.speakers.items():
        print(f"  {lbl:14s} : {s.turn_count} turns / {s.total_speech_s:.0f}s / ~{s.word_count} words")
    print("═" * 60)
    # Print the first 5 turns as a preview.
    for turn in result.turns[:5]:
        mins, secs = divmod(int(turn.start), 60)
        print(f"\n  [{turn.speaker}] {mins}:{secs:02d}")
        preview = turn.text[:120] + ("…" if len(turn.text) > 120 else "")
        print(f"  {preview}")
    if len(result.turns) > 5:
        print(f"\n  … {len(result.turns) - 5} more turns (see exported file)")
    print()


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv()
    
    parser = _build_arg_parser()
    args = parser.parse_args()

    configure_logging(args.log_level)

    config = TranscriberConfig()
    output_dir: Path = args.output_dir or config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Chunk config ──────────────────────────────────────────────────────────
    chunk_config = None
    if args.chunk:
        from transcriber.audio.chunker import ChunkConfig  # noqa: PLC0415
        chunk_config = ChunkConfig(
            target_duration_s=args.chunk_minutes * 60,
            min_silence_ms=args.silence_ms,
        )

    # ── Diarizer ──────────────────────────────────────────────────────────────
    diarizer = None
    diarized_exporter = None
    if args.diarize:
        from transcriber.audio.diarizer import SpeakerDiarizer  # noqa: PLC0415
        from transcriber.output.exporters import (  # noqa: PLC0415
            SpeakerTextExporter,
            TranscriptJsonExporter,
        )
        diarizer = SpeakerDiarizer(flip_roles=args.flip_roles)
        diarized_exporter = (
            SpeakerTextExporter()
            if args.diarize_format == "transcript.txt"
            else TranscriptJsonExporter()
        )

    # ── Compose object graph ──────────────────────────────────────────────────
    engine = FasterWhisperTranscriber(config)
    exporter = exporter_for(f".{args.format}") if args.format else None
    service = TranscriptionService(
        engine,
        exporter=exporter,
        diarized_exporter=diarized_exporter,
        chunk_config=chunk_config,
        diarizer=diarizer,
    )

    audio_files = _collect_audio_files(args.input, batch=args.batch)

    if len(audio_files) == 1:
        export_to = (output_dir / audio_files[0].stem) if (exporter or diarized_exporter) else None

        if args.diarize:
            result = service.transcribe_and_diarize(
                audio_files[0], export_to=export_to, keep_chunks=args.keep_chunks
            )
        else:
            result = service.transcribe_file(
                audio_files[0], export_to=export_to, keep_chunks=args.keep_chunks
            )
        _print_result(result)
    else:
        service.transcribe_batch(
            audio_files,
            output_dir=output_dir if (exporter or diarized_exporter) else None,
            keep_chunks=args.keep_chunks,
            skip_existing=args.skip_existing,
        )


if __name__ == "__main__":
    main()
