#!/usr/bin/env python3
"""
scripts/batch_transcribe.py — Batch transcription helper script
================================================================

A convenience script that transcribes every audio file in a directory
and saves each result in all three formats (.txt, .json, .srt).

Usage::

    python scripts/batch_transcribe.py ./audio_files --output ./results

Optional arguments::

    --model    tiny | base | small | medium | large-v3  (default: large-v3)
    --device   cpu | cuda                               (default: cpu)
    --format   txt | json | srt | all                   (default: all)

Examples::

    # Transcribe everything in ./recordings → ./output (all formats)
    python scripts/batch_transcribe.py recordings/

    # Use a lighter model for a quick draft
    python scripts/batch_transcribe.py recordings/ --model medium

    # Export only SRT (for video subtitling)
    python scripts/batch_transcribe.py recordings/ --format srt
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure the project root is on sys.path when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from transcriber.config import ComputeType, TranscriberConfig, WhisperModelSize
from transcriber.logging_setup import configure_logging
from transcriber.models.faster_whisper_model import FasterWhisperTranscriber
from transcriber.output.exporters import JsonExporter, PlainTextExporter, SrtExporter
from transcriber.protocols import TranscriptionResult
from transcriber.transcription.service import TranscriptionService

AUDIO_EXTENSIONS = frozenset({".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus"})


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Batch-transcribe a directory of Spanish audio files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input_dir", type=Path, help="Directory containing audio files.")
    p.add_argument(
        "--output", "-o", type=Path, default=Path("./output"),
        help="Destination directory for transcription files. (default: ./output)"
    )
    p.add_argument(
        "--model", default="large-v3",
        choices=[m.value for m in WhisperModelSize],
        help="Whisper model size. (default: large-v3)"
    )
    p.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Inference device. (default: cpu)"
    )
    p.add_argument(
        "--format", default="all", choices=["txt", "json", "srt", "all"],
        help="Output format(s). (default: all)"
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def _export_all(result: TranscriptionResult, base: Path) -> None:
    """Write result in all three formats to *base* (stem only, no suffix)."""
    PlainTextExporter().export(result, base)
    JsonExporter().export(result, base)
    SrtExporter().export(result, base)


def main() -> None:
    args = _parse_args()
    configure_logging(args.log_level)

    input_dir: Path = args.input_dir
    output_dir: Path = args.output

    if not input_dir.is_dir():
        print(f"ERROR: '{input_dir}' is not a directory.", file=sys.stderr)
        sys.exit(1)

    audio_files = sorted(
        f for f in input_dir.rglob("*") if f.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        print(f"No audio files found in '{input_dir}'.", file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n🎙  Found {len(audio_files)} audio file(s) in '{input_dir}'")
    print(f"📂  Output → '{output_dir}'")
    print(f"🤖  Model  : {args.model} on {args.device}\n")

    config = TranscriberConfig(
        model_size=WhisperModelSize(args.model),
        language="es",
        device=args.device,
        compute_type=ComputeType.INT8 if args.device == "cpu" else ComputeType.FLOAT16,
    )
    engine = FasterWhisperTranscriber(config)
    service = TranscriptionService(engine)

    total_start = time.perf_counter()
    successes = 0

    for idx, audio_path in enumerate(audio_files, start=1):
        print(f"[{idx:03d}/{len(audio_files):03d}] {audio_path.name}", end=" … ", flush=True)
        t0 = time.perf_counter()
        try:
            result = service.transcribe_file(audio_path)
            base = output_dir / audio_path.stem

            if args.format == "all":
                _export_all(result, base)
            else:
                from transcriber.output.exporters import exporter_for
                exporter_for(f".{args.format}").export(result, base)

            elapsed = time.perf_counter() - t0
            print(f"✓  {elapsed:.1f}s  ({result.average_confidence:.0%} confidence)")
            successes += 1
        except Exception as exc:  # noqa: BLE001
            print(f"✗  FAILED — {exc}")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n{'─'*50}")
    print(f"✅  {successes}/{len(audio_files)} files transcribed in {total_elapsed:.1f}s")
    print(f"📂  Results saved to '{output_dir.resolve()}'")


if __name__ == "__main__":
    main()
