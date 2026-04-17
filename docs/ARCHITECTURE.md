# Architecture Overview

## Design Philosophy

The project follows SOLID principles in a Pythonic way:

- `Protocol` for interfaces — structural duck-typing, no forced inheritance
- `dataclass(frozen=True)` for value objects — immutability enforced at runtime
- `pydantic-settings` for configuration — validated at startup, reads `.env` automatically
- Constructor injection throughout — every collaborator is a parameter, making tests trivial

The goal is a codebase that is easy to test, easy to extend, and hard to break accidentally.

---

## Full Layer Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  Entry Points                                                       │
│  main.py (CLI)  ·  scripts/batch_transcribe.py                      │
└───────────────────────────────┬─────────────────────────────────────┘
                                │ wires and calls
┌───────────────────────────────▼─────────────────────────────────────┐
│  Application Layer  —  transcriber/transcription/service.py         │
│  TranscriptionService                                               │
│  transcribe_file() · transcribe_and_diarize() · transcribe_batch()  │
└──┬──────────────┬──────────────┬──────────────┬─────────────────────┘
   │              │              │              │
   │ (Protocol)   │ (Protocol)   │ (injected)   │ (injected)
   ▼              ▼              ▼              ▼
Engine        Exporter      Normalizer    Diarizer
Layer         Layer         Layer         Layer
   │              │              │              │
FasterWhisper  PlainText     AudioNorm-    SpeakerDiarizer
Transcriber    JsonExporter  alizer        (pyannote.audio)
               SrtExporter               │
               Transcript-               ▼
               JsonExporter          Aligner
               SpeakerText-          (fuse Whisper
               Exporter               + diarization)
                                          │
                                          ▼
                                     DiarizedTranscript
                                     (turns by speaker)

               Audio Processing Layer  (transcriber/audio/)
               ┌──────────────────────────────────────────┐
               │  Chunker   → splits at silence boundaries│
               │  Reassembler → merges chunk results      │
               │  Checkpoint → atomic resume state        │
               │  HaltController → Ctrl+C / 'q' / SIGTERM│
               └──────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  Domain Core  —  transcriber/protocols.py  +  config.py            │
│                                                                     │
│  Value objects (frozen dataclasses — immutable):                    │
│    TranscriptionSegment  · TranscriptionResult                      │
│    SpeakerTurn · SpeakerStats · DiarizedTranscript                  │
│                                                                     │
│  Protocols (structural interfaces):                                  │
│    SpeechTranscriber   — transcribe(path) → TranscriptionResult     │
│    ResultExporter      — export(result, dest) → Path                │
│                                                                     │
│  Exceptions:                                                        │
│    HaltException — deliberate pause, distinct from engine errors    │
│                                                                     │
│  Configuration:                                                     │
│    TranscriberConfig   — pydantic-settings, reads .env              │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Complete Data Flow

### Transcription only (`transcribe_file`)

```
Audio file (Path)
    │
    ▼
AudioNormalizer.prepare()
    │   WAV? pass-through (cleanup_required=False)
    │   Other? decode → 16 kHz mono WAV temp file (cleanup_required=True)
    ▼
AudioChunker.split()          [only when chunk_config is set]
    │   1. Load audio via pydub
    │   2. Scan for silence regions (pydub.silence.detect_silence)
    │   3. Find longest silence within ±search_window of each target boundary
    │   4. Cut at silence midpoint → never mid-word
    │   5. Add overlap_ms padding to each chunk for model warm-up
    ▼
CheckpointManager.load()      [chunked path only]
    │   Restore completed chunk results from .checkpoint.json if present
    ▼
HaltController.install()      [chunked path only]
    │   Register SIGINT / SIGTERM handlers + optional stdin 'q' reader
    ▼
[for each chunk]
    │   halt.should_halt()? → raise HaltException (checkpoint preserved)
    │   checkpoint.is_done()? → skip
    │   FasterWhisperTranscriber.transcribe(chunk.wav)
    │       └── WhisperModel.transcribe() → lazy Segment iterator
    │           → materialise all segments immediately
    │           → _logprob_to_confidence() per segment
    │           → TranscriptionResult (immutable)
    │   checkpoint.save_chunk() → atomic write (tmp → rename)
    ▼
reassemble(chunk_results, chunks)
    │   1. Shift segment timestamps: chunk-relative → absolute
    │   2. Drop overlap segments (those before start_s + overlap_ms/1000)
    │   3. SequenceMatcher dedup at boundaries (similarity > 0.82)
    ▼
AudioNormalizer.cleanup()     [if temp WAV was created]
    ▼
ResultExporter.export()       [if export_to is set]
    ▼
TranscriptionResult           [returned to caller]
```

### With diarization (`transcribe_and_diarize`)

```
[same transcription flow as above]
    ▼
TranscriptionResult
    │
    ├── AudioNormalizer.prepare(original_file)
    │       [pyannote needs the full original audio, not chunks]
    ▼
SpeakerDiarizer.diarize(prepared.path)
    │   1. pyannote Pipeline.from_pretrained() [lazy, cached after first call]
    │   2. Pipeline(audio) → raw (speaker, start, end) segments
    │   3. Total speaking time per speaker label
    │   4. Sort by time: less → INTERVIEWER, more → INTERVIEWEE
    │   5. Return list[DiarizationSegment] with mapped labels
    ▼
align(transcript, diarization)
    │   For each Whisper segment [ws, we]:
    │     → find all diarization segments overlapping [ws, we]
    │     → pick speaker with most overlap (seconds of intersection)
    │     → if no overlap: carry forward last known speaker
    │   Group consecutive same-speaker segments → SpeakerTurn objects
    │   Compute SpeakerStats per speaker
    ▼
DiarizedTranscript
    │   .turns              → list[SpeakerTurn] (primary unit for coding)
    │   .speakers           → dict[str, SpeakerStats]
    │   .interviewee_turns  → filtered list
    │   .interviewer_turns  → filtered list
    │   .source             → TranscriptionResult with speaker on each segment
    ▼
TranscriptJsonExporter / SpeakerTextExporter
    ▼
.transcript.json / .transcript.txt
```

---

## SOLID Principles Applied

### Single-Responsibility Principle

Each class has exactly one reason to change:

| Class | Sole responsibility |
|---|---|
| `AudioNormalizer` | Convert non-WAV to 16 kHz mono WAV |
| `AudioChunker` | Split audio at silence boundaries |
| `CheckpointManager` | Save/load progress atomically |
| `HaltController` | Detect and signal user halt requests |
| `FasterWhisperTranscriber` | Run Whisper inference, adapt output |
| `SpeakerDiarizer` | Identify who spoke when |
| `align()` | Fuse Whisper timestamps with diarization timestamps |
| `TranscriptionService` | Orchestrate the pipeline — nothing else |
| `TranscriptJsonExporter` | Write `.transcript.json` |
| `SpeakerTextExporter` | Write `.transcript.txt` |
| `TranscriberConfig` | Hold and validate settings |

### Open/Closed Principle

Adding a new output format: add one class to `exporters.py`, register in
`_EXPORTERS`. Zero changes to existing code.

Adding a new transcription engine: add one module to `transcriber/models/`.
Zero changes to `TranscriptionService`.

### Liskov Substitution Principle

Any class with `transcribe(path) -> TranscriptionResult` satisfies
`SpeechTranscriber`. You can substitute a cloud API wrapper, a mock, or a
faster engine without touching the service.

### Interface Segregation Principle

Two narrow protocols, one method each:
- `SpeechTranscriber.transcribe(path)`
- `ResultExporter.export(result, dest)`

Neither protocol is contaminated with methods its clients don't need.

### Dependency-Inversion Principle

`TranscriptionService` depends on protocols, never on concrete classes.
`main.py` is the only place concrete classes are instantiated and wired
together (the Composition Root). Tests inject stubs through the constructor
— no monkey-patching required.

---

## Key Design Decisions

### `cleanup_required` flag on `PreparedAudio`

`AudioNormalizer.prepare()` returns a `PreparedAudio` dataclass with a
`cleanup_required: bool` field. When the input is already WAV, the file is
returned unchanged and `cleanup_required=False` — the service returns the
engine's result as-is, preserving object identity. When a temp file is
created, `cleanup_required=True` and the service stamps `source_path` with
the original caller-supplied path. This design avoids an equality comparison
against a resolved path (which would rebuild the frozen dataclass every time)
and keeps unit test `result is sample_result` assertions valid.

### Why `HaltException` instead of `RuntimeError`?

`TranscriptionService.transcribe_batch()` needs to distinguish two cases:

1. Engine error on file N → skip that file, continue to file N+1.
2. User pressed Ctrl+C on file N → stop the entire batch, preserve checkpoint.

Using a plain `RuntimeError` for both makes case 2 indistinguishable from
case 1 at the batch level. `HaltException(RuntimeError)` is a named subclass
that the batch loop catches specifically, stopping immediately while letting
`except Exception` swallow ordinary errors.

### Why materialise the Whisper segment iterator?

`WhisperModel.transcribe()` returns a lazy generator. Errors inside it
(e.g. CTranslate2 allocation failure on a large chunk) would only surface
when the caller iterates — which might be deep inside the reassembler. We
call `list(segments_iter)` immediately inside `_to_segment()` so errors
are raised inside `transcribe()` itself, making stack traces actionable.

### Why temporal overlap for alignment?

A simpler approach (assign the diarization speaker whose segment contains
the Whisper segment's midpoint) fails at speaker boundaries: if a Whisper
segment spans a turn change, its midpoint might be attributed to the wrong
speaker. Temporal overlap is correct: we compute intersection length for
every diarization segment that overlaps the Whisper window and pick the
dominant one — the speaker who "owned" more of that time window.

### Why atomic checkpoint writes?

The checkpoint file is written with write-then-rename (`os.replace`). On
POSIX systems `rename` is atomic: readers either see the old complete file
or the new complete file, never a half-written one. This means Ctrl+C, a
power cut, or OOM during a write cannot corrupt the checkpoint.

---

## Adding a New Engine

1. Create `transcriber/models/my_engine.py`.
2. Implement `transcribe(self, audio_path: Path) -> TranscriptionResult`.
3. Wire it in `main.py` — the only change needed.

```python
# main.py
engine = MyEngineTranscriber(config)   # replaces FasterWhisperTranscriber
service = TranscriptionService(engine, ...)
```

## Adding a New Export Format

1. Add a class to `exporters.py`:

```python
class CsvExporter:
    def export(self, result: TranscriptionResult, destination: Path) -> Path:
        path = destination.with_suffix(".csv")
        ...
        return path
```

2. Register it: `_EXPORTERS[".csv"] = CsvExporter`
3. Add `"csv"` to the `--format` choices in `main.py`.

For diarized formats (operating on `DiarizedTranscript`), follow the same
pattern but accept `DiarizedTranscript` and inject via `diarized_exporter=`.

---

## Test Architecture

```
tests/
├── conftest.py              # Shared fixtures
│   ├── mock_normalizer      # Stub that bypasses ffmpeg/pydub
│   │                        # Returns audio_file unchanged, cleanup_required=False
│   ├── mock_transcriber     # Always returns sample_result
│   ├── sample_result        # Fixed TranscriptionResult for assertions
│   └── audio_file           # .wav file — passes through normalizer unchanged
│
├── unit/                    # 146 tests, ~2 s, no network, no model, no GPU
│   ├── test_protocols.py    # Value object behaviour
│   ├── test_config.py       # Pydantic validation
│   ├── test_exporters.py    # All five exporters
│   ├── test_service.py      # Full service with injected stubs
│   ├── test_faster_whisper_model.py  # _logprob_to_confidence
│   ├── test_chunker_reassembler.py   # Splitting + timestamp correction
│   ├── test_checkpoint_halt.py       # Resume + signal handling
│   ├── test_diarization.py           # Diarizer, aligner, diarized exporters
│   └── test_main.py                  # Path resolution helpers
│
└── integration/             # Skipped by default; require tiny model (~74 MB)
    └── test_pipeline_integration.py
```

The key testing principle: `TranscriptionService` receives all its
collaborators through its constructor. Unit tests inject `MagicMock` objects
via fixture. No `monkeypatch` is used in the core service tests — the design
makes it unnecessary.
