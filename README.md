# audio-transcriber

Spanish audio transcription and speaker diarization — built with faster-whisper,
pyannote.audio, and a clean SOLID architecture.

---

## Features

- Whisper large-v3 accuracy, ~4× faster via CTranslate2 INT8
- Spanish-first forced decoding + VAD silence removal
- Smart audio chunking at natural silence boundaries (never mid-word)
- Checkpoint / resume — Ctrl+C saves progress, re-run to continue
- Stage resume for diarization — reuse the finished transcript if diarization fails later
- Speaker diarization — identifies INTERVIEWER and INTERVIEWEE automatically
- Two output formats for qualitative research pipelines:
  - `.transcript.json` — machine-readable turns with timestamps for auto-coding
  - `.transcript.txt` — human-readable labelled dialogue for NVivo / ATLAS.ti
- Fully unit-tested — all 146 tests run without a model, internet, or GPU

---

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | Use pyenv if your system Python is older |
| ffmpeg | 4–7 | Must be the system package, not pip |
| pydub | 0.25+ | Installed via pip, wraps ffmpeg |
| faster-whisper | 1.0+ | CPU inference via CTranslate2 |
| pyannote.audio | 3.1+ | Optional — diarization only |
| torch (CPU) | 2.x | Optional — required by pyannote only |

---

## Installation

### Step 1 — Create the virtual environment

Always work inside a `.venv` so that `pip`, `pytest`, and `python` all
refer to the same interpreter.

```bash
cd audio-transcriber

# Create the venv (do this once)
python3 -m venv .venv

# Activate it — you must do this every terminal session
source .venv/bin/activate

# Confirm you are inside the venv
which python          # should print: .../audio-transcriber/.venv/bin/python
which pip             # should print: .../audio-transcriber/.venv/bin/pip
```

### Step 2 — Install the package

```bash
# Core install (transcription only — no diarization)
pip install -e .

# With dev tools (pytest, ruff, mypy)
pip install -e ".[dev]"
```

### Step 3 — Install torch CPU-only (required for diarization)

> Skip this step if you do not need speaker diarization.

The default `pip install torch` downloads the CUDA wheel which pulls in
`libnppicc.so` and other CUDA NPP libraries. On a machine without an
NVIDIA GPU and the full CUDA toolkit these files do not exist and torch
will print a long `torchcodec` warning and may crash at runtime.

Install the CPU-only wheel explicitly:

```bash
# CPU-only torch — no CUDA libraries, no NPP warnings
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify it works:

```bash
python -c "import torch; print(torch.__version__, '— CUDA:', torch.cuda.is_available())"
# Expected: 2.x.x — CUDA: False
```

### Step 4 — Install diarization dependencies

```bash
pip install -e ".[diarization]"
```

### Step 5 — ffmpeg

ffmpeg must be installed as a system package, not via pip. The pip package
`ffmpeg-python` is a Python wrapper and does not install the binary.

```bash
# Pop!_OS / Ubuntu / Debian
sudo apt update && sudo apt install ffmpeg

# Verify
ffmpeg -version | head -1
# Expected: ffmpeg version 4.x / 5.x / 6.x
```

> Your log shows `FFmpeg version 8` was detected by torchcodec but
> `libnppicc.so.13` was missing. This is not an ffmpeg problem — it is
> caused by the CUDA torch wheel (see Step 3). Once you switch to the
> CPU torch wheel the torchcodec warnings disappear entirely.

### Step 6 — HuggingFace token (diarization only)

pyannote requires you to accept the model licence terms once via the
HuggingFace website.

1. Create a free account at https://hf.co
2. Accept the terms at https://hf.co/pyannote/speaker-diarization-3.1
3. Accept the terms at https://hf.co/pyannote/segmentation-3.0
4. Create a read token at https://hf.co/settings/tokens
5. Set it in your environment:

```bash
# Option A — add to .env (recommended, never commit this file)
echo "HF_TOKEN=hf_your_token_here" >> .env

# Option B — export for the current session only
export HF_TOKEN=hf_your_token_here
```

### Step 7 — Copy and edit the environment file

```bash
cp .env.example .env
# Edit .env with your preferred model size, device, etc.
```

---

## Quick Start

### Transcribe only (no diarization)

```bash
# Activate venv first
source .venv/bin/activate

# Single file → print to stdout
python -m transcriber.main audio/interview.m4a

# With chunking (recommended for files longer than 20 min)
python -m transcriber.main audio/interview.m4a --chunk

# Chunked + export as SRT
python -m transcriber.main audio/interview.m4a --chunk --format srt
```

### Transcribe + identify speakers

```bash
# Full pipeline — chunked, diarized, exported as structured JSON
python -m transcriber.main audio/interview.m4a \
    --chunk --chunk-minutes 10 \
    --diarize --diarize-format transcript.json

# Human-readable version for NVivo / ATLAS.ti
python -m transcriber.main audio/interview.m4a \
    --chunk --chunk-minutes 10 \
    --diarize --diarize-format transcript.txt

# If the interviewee speaks less than the interviewer, flip the assignment
python -m transcriber.main audio/interview.m4a \
    --diarize --flip-roles
```

When `--diarize` is enabled, the CLI now validates HuggingFace access before
starting the long Whisper pass. If diarization still fails later, the finished
transcript is cached in a hidden stage checkpoint and the same command resumes
from diarization instead of retranscribing the audio.

### Pause and resume a long job

```bash
# Start — press Ctrl+C at any time to pause
python -m transcriber.main audio/interview.m4a --chunk

# Re-run the exact same command — skips completed chunks automatically
python -m transcriber.main audio/interview.m4a --chunk
```

---

## CLI Reference

```
usage: transcribe [-h] [--format {txt,json,srt}] [--output-dir DIR]
                  [--chunk] [--chunk-minutes N] [--silence-ms MS]
                  [--keep-chunks] [--diarize]
                  [--diarize-format {transcript.json,transcript.txt}]
                  [--flip-roles] [--batch]
                  [--log-level {DEBUG,INFO,WARNING,ERROR}]
                  input

positional arguments:
  input                 Audio file or directory (use with --batch)

transcription:
  --format              Raw transcript format: txt | json | srt
  --output-dir          Directory for exported files

chunking (recommended for files > 20 min):
  --chunk               Split audio at silence boundaries before transcribing
  --chunk-minutes N     Target chunk length in minutes (default: 10)
  --silence-ms MS       Min silence length for a cut point in ms (default: 800)
  --keep-chunks         Keep temporary chunk WAV files after transcription

diarization (identify INTERVIEWER / INTERVIEWEE):
  --diarize             Run speaker diarization (requires HF_TOKEN)
  --diarize-format FMT  transcript.json (default) or transcript.txt
  --flip-roles          Swap INTERVIEWER / INTERVIEWEE assignment

other:
  --batch               Transcribe all audio files in INPUT directory
  --log-level           Logging verbosity (default: INFO)
```

---

## Configuration

All settings can be overridden via environment variables or `.env`.

| Variable | Default | Description |
|---|---|---|
| `TRANSCRIBER_MODEL_SIZE` | `large-v3` | `tiny`, `base`, `small`, `medium`, `large-v3` |
| `TRANSCRIBER_LANGUAGE` | `es` | ISO 639-1 language code |
| `TRANSCRIBER_DEVICE` | `cpu` | `cpu` only (no CUDA on this machine) |
| `TRANSCRIBER_COMPUTE_TYPE` | `int8` | `int8` for CPU |
| `TRANSCRIBER_BEAM_SIZE` | `5` | Beam width (1 = greedy, ~30% faster) |
| `TRANSCRIBER_VAD_FILTER` | `true` | Remove silence before transcription |
| `OUTPUT_DIR` | `./output` | Default export directory |
| `HF_TOKEN` | — | HuggingFace token for diarization |

### Model size guide for your i7-8565U + 12 GB RAM

| Model | RAM | Est. time (26 min audio) | Best for |
|---|---|---|---|
| `tiny` | ~1 GB | ~4 min | Quick drafts |
| `medium` | ~5 GB | ~40 min | Balanced — recommended |
| **`large-v3`** | ~10 GB | **~48 min** | **Best accuracy (your current setting)** |

---

## Output formats explained

| Format | Use for |
|---|---|
| `.transcript.json` | Auto-coding pipelines, thematic synthesis scripts — machine-readable turns with timestamps, speaker labels, word counts, confidence scores |
| `.transcript.txt` | NVivo / ATLAS.ti / Dedoose import, manual annotation, sharing with collaborators — speaker-labelled paragraphs with `H:MM:SS` timestamps |
| `.srt` | Video subtitle players (VLC, mpv) |
| `.json` | Raw Whisper output with segment-level data (no speaker labels) |
| `.txt` | Plain text transcript (no speakers, no timestamps) — avoid for research pipelines |

---

## Project structure

```
audio-transcriber/
├── transcriber/
│   ├── __init__.py                      # Public re-exports
│   ├── config.py                        # pydantic-settings configuration
│   ├── protocols.py                     # Domain objects + Protocol interfaces
│   │                                    # (TranscriptionSegment, TranscriptionResult,
│   │                                    #  SpeakerTurn, DiarizedTranscript,
│   │                                    #  SpeechTranscriber, ResultExporter,
│   │                                    #  HaltException)
│   ├── logging_setup.py                 # Centralised log configuration
│   ├── main.py                          # CLI entry-point + Composition Root
│   │
│   ├── audio/
│   │   ├── normalizer.py                # Non-WAV → 16 kHz mono WAV conversion
│   │   ├── chunker.py                   # VAD-aware audio splitter
│   │   ├── reassembler.py               # Merge chunk results + fix timestamps
│   │   ├── checkpoint.py                # Atomic checkpoint save/load for resume
│   │   ├── halt.py                      # Ctrl+C / SIGTERM / 'q' halt controller
│   │   ├── diarizer.py                  # pyannote.audio speaker diarization
│   │   └── aligner.py                   # Fuse Whisper segments with diarization
│   │
│   ├── models/
│   │   └── faster_whisper_model.py      # FasterWhisperTranscriber engine
│   │
│   ├── transcription/
│   │   └── service.py                   # TranscriptionService (orchestration)
│   │
│   └── output/
│       └── exporters.py                 # All exporters:
│                                        #   PlainTextExporter  → .txt
│                                        #   JsonExporter       → .json
│                                        #   SrtExporter        → .srt
│                                        #   TranscriptJsonExporter → .transcript.json
│                                        #   SpeakerTextExporter    → .transcript.txt
│
├── tests/
│   ├── conftest.py                      # Shared fixtures (mock_normalizer etc.)
│   ├── unit/
│   │   ├── test_protocols.py
│   │   ├── test_config.py
│   │   ├── test_exporters.py
│   │   ├── test_service.py
│   │   ├── test_faster_whisper_model.py
│   │   ├── test_chunker_reassembler.py
│   │   ├── test_checkpoint_halt.py
│   │   ├── test_diarization.py
│   │   └── test_main.py
│   └── integration/
│       └── test_pipeline_integration.py # Requires tiny model (~74 MB)
│
├── scripts/
│   └── batch_transcribe.py
│
├── docs/
│   └── ARCHITECTURE.md
│
├── .env.example
├── pyproject.toml
└── README.md
```

---

## Running tests

### The common problem — pytest not found

The most frequent issue is running `pytest` (or `pip`) from outside the
virtual environment. The system Python and the venv Python are completely
separate environments with separate package lists.

```bash
# Wrong — uses system Python, pytest may not be installed there
pytest tests/

# Wrong — installs into system Python, not the venv
sudo pip install pytest

# Correct — activate the venv first, then use its pytest
source .venv/bin/activate
which pytest    # must print: .../audio-transcriber/.venv/bin/pytest
pytest tests/unit/ -v
```

If `source .venv/bin/activate` does not work, the venv may not exist yet:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Running unit tests (fast, no model, no internet)

All 146 unit tests run in under 2 seconds. They use mock objects — no
Whisper model is downloaded, no real audio is decoded.

```bash
# Activate venv
source .venv/bin/activate

# All unit tests with coverage
pytest tests/unit/ -v

# Single test file
pytest tests/unit/test_service.py -v

# Single test by name
pytest tests/unit/test_service.py::TestTranscribeFile::test_returns_result_from_engine -v

# Without coverage (slightly faster)
pytest tests/unit/ -v -p no:cov

# Quiet mode — just pass/fail counts
pytest tests/unit/ -q
```

### Running integration tests (requires model download)

Integration tests download the tiny Whisper model (~74 MB) on first run
and generate a synthetic WAV file. They do not require a real interview.

```bash
# Run integration tests explicitly
pytest tests/integration/ -v -m integration

# Run everything (unit + integration)
pytest -v
```

### Running a subset by keyword

```bash
# All tests related to diarization
pytest -k diarization -v

# All tests related to chunking
pytest -k chunk -v

# Skip integration tests
pytest -m "not integration" -v
```

### Expected output

```
tests/unit/test_config.py::TestTranscriberConfig::test_defaults_are_sensible PASSED
tests/unit/test_diarization.py::TestAlign::test_basic_two_speaker_alignment PASSED
...
========================= 146 passed in 1.88s =========================
```

---

## Troubleshooting

### `pytest: command not found`

The venv is not active. Run `source .venv/bin/activate` first.

### `ModuleNotFoundError: No module named 'transcriber'`

The package is not installed in the venv. Run `pip install -e .` with the
venv active.

### `TypeError: Pipeline.from_pretrained() got an unexpected keyword argument 'use_auth_token'`

This was a bug in an earlier version of the code — fixed in v1.1.0. The
`diarizer.py` now detects the installed pyannote version and uses the
correct keyword (`token` for 3.x, `use_auth_token` for 2.x) automatically.

### `UserWarning: torchcodec is not installed correctly` / `libnppicc.so.13`

This warning comes from installing the default (CUDA) torch wheel on a
machine without the full CUDA toolkit. It does not affect Whisper
transcription (which uses CTranslate2, not torch). It only matters for
pyannote diarization.

Fix: reinstall torch with the CPU wheel (see Installation Step 3).

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### `ffmpeg not found on PATH`

```bash
sudo apt install ffmpeg
ffmpeg -version
```

### Diarization pipeline download is slow or fails

pyannote downloads ~1 GB of model weights on first use. Ensure you have:
- Accepted the terms at https://hf.co/pyannote/speaker-diarization-3.1
- Accepted the terms at https://hf.co/pyannote/segmentation-3.0
- A valid `HF_TOKEN` set in `.env` or as an environment variable

### INTERVIEWER and INTERVIEWEE are swapped

The diarizer assigns roles by total speaking time: less time = INTERVIEWER.
If your interview has an unusually talkative interviewer, use `--flip-roles`:

```bash
python -m transcriber.main audio/interview.m4a --diarize --flip-roles
```

---

## Python API

```python
from pathlib import Path
from transcriber.config import TranscriberConfig
from transcriber.models.faster_whisper_model import FasterWhisperTranscriber
from transcriber.audio.chunker import ChunkConfig
from transcriber.audio.diarizer import SpeakerDiarizer
from transcriber.output.exporters import TranscriptJsonExporter
from transcriber.transcription.service import TranscriptionService

config = TranscriberConfig(language="es", device="cpu")
engine = FasterWhisperTranscriber(config)

service = TranscriptionService(
    engine,
    diarizer=SpeakerDiarizer(),            # reads HF_TOKEN from env
    diarized_exporter=TranscriptJsonExporter(),
    chunk_config=ChunkConfig(target_duration_s=600),
)

diarized = service.transcribe_and_diarize(
    Path("audio/interview.m4a"),
    export_to=Path("output/interview"),    # writes output/interview.transcript.json
)

for turn in diarized.interviewee_turns:
    print(f"[{turn.start:.0f}s] {turn.text}")
```

---

## License

MIT — see `LICENSE` for details.
