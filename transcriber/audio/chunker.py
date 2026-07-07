"""
chunker.py — VAD-aware audio chunker (speech-safe splitting)
============================================================

The core problem with naive splitting (fixed bytes / fixed seconds) is that
a cut can land in the middle of a word or sentence.  Whisper then receives
an audio chunk that starts or ends mid-utterance and either:

  - hallucinates a word to complete the broken fragment, or
  - silently drops the tail of a sentence.

Strategy used here
------------------
1. **Silence scan** — walk the audio looking for stretches of silence that
   are long enough to be natural pauses (configurable, default 800 ms).
2. **Chunk at the deepest silence inside each target window** — if the
   target chunk size is 10 minutes, we scan [9 min → 11 min] for the
   longest silence and cut there.  The chunk boundary is always inside a
   pause, never inside speech.
3. **Overlap padding** — each chunk includes a small configurable overlap
   (default 200 ms) at both ends so Whisper's attention context has a
   warm-up and cool-down.  When re-assembling, overlapping transcript text
   is de-duplicated by comparing the tail of the previous segment with the
   head of the next (fuzzy match on the last N words).

This module is a pure audio-processing utility — it has no dependency on
the transcription model and can be tested in isolation.

Dependencies
------------
- pydub    : audio decoding + silence detection (wraps ffmpeg)
- ffmpeg   : must be installed on the system (``apt install ffmpeg``)

Install
-------
    pip install pydub
    sudo apt install ffmpeg       # Ubuntu / Pop!_OS
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────


@dataclass(frozen=True)
class ChunkConfig:
    """
    Parameters controlling how audio is split.

    Attributes
    ----------
    target_duration_s:
        Desired chunk length in seconds.  The actual cut will be at
        the nearest silence boundary, so chunks may be slightly shorter
        or longer.  Default: 600 s (10 minutes).
    max_duration_s:
        Hard upper limit.  If no silence is found within the search
        window, the chunk is cut here even if it falls mid-speech.
        Default: 720 s (12 minutes).
    min_silence_ms:
        Minimum silence length (ms) that qualifies as a safe cut point.
        Default: 800 ms — long enough to be a sentence boundary, short
        enough to find candidates in dense speech.
    silence_thresh_dbfs:
        Audio level (dBFS) below which a segment is considered silence.
        -40 dBFS works for clean recordings; use -35 for noisy environments.
    overlap_ms:
        Milliseconds of audio added to the start and end of each chunk
        as context padding for the model.  Overlapping text is removed
        during transcript reassembly.
    search_window_s:
        How far before/after the target cut point to search for silence,
        in seconds.  Default: 60 s (search ±1 minute).
    tmp_dir:
        Directory for temporary chunk files.  If None, the system temp
        directory is used.  Chunks are deleted after transcription unless
        ``keep_chunks=True`` is passed to :func:`split_audio`.
    """

    target_duration_s: float = 600.0       # 10 minutes
    max_duration_s: float = 720.0          # 12 minutes hard cap
    min_silence_ms: int = 800              # 0.8 s silence = safe cut
    silence_thresh_dbfs: float = -40.0    # dBFS silence threshold
    overlap_ms: int = 200                  # 200 ms warm-up / cool-down
    search_window_s: float = 60.0         # ±1 min around target
    tmp_dir: Path | None = None


@dataclass
class AudioChunk:
    """
    One piece of a larger audio file, ready for transcription.

    Attributes
    ----------
    index:
        Zero-based position in the chunk sequence.
    path:
        Filesystem path to the WAV file for this chunk.
    start_s:
        Start time of this chunk within the *original* audio (seconds).
    end_s:
        End time of this chunk within the original audio (seconds).
    overlap_start_ms:
        How many milliseconds at the *start* of this chunk are overlap
        from the previous chunk.  Used during reassembly to trim
        duplicate transcript text.
    overlap_end_ms:
        How many milliseconds at the *end* of this chunk are overlap
        into the next chunk.
    """

    index: int
    path: Path
    start_s: float
    end_s: float
    overlap_start_ms: int = 0
    overlap_end_ms: int = 0

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def __str__(self) -> str:
        return (
            f"Chunk {self.index:03d} [{self.start_s/60:.1f} min"
            f" → {self.end_s/60:.1f} min] ({self.duration_s:.0f}s)"
        )


# ─────────────────────────────────────────────
#  Core logic
# ─────────────────────────────────────────────


class AudioChunker:
    """
    Splits a long audio file into VAD-safe chunks.

    Parameters
    ----------
    config:
        Splitting configuration.

    Example
    -------
    >>> chunker = AudioChunker(ChunkConfig(target_duration_s=600))
    >>> chunks = chunker.split(Path("interview.mp3"))
    >>> for chunk in chunks:
    ...     print(chunk)
    Chunk 000 [0.0 min → 10.2 min] (612s)
    Chunk 001 [10.1 min → 20.3 min] (613s)
    """

    def __init__(self, config: ChunkConfig | None = None) -> None:
        self._config = config or ChunkConfig()
        self._check_ffmpeg()

    # ── public ───────────────────────────────────────────────────────────────

    def split(self, audio_path: Path, *, keep_chunks: bool = False) -> list[AudioChunk]:
        """
        Split *audio_path* into speech-safe chunks.

        Parameters
        ----------
        audio_path:
            Input audio file (any format ffmpeg supports: mp3, wav, m4a, ogg …).
        keep_chunks:
            If True, chunk files are NOT deleted after creation.
            Useful for debugging or re-running transcription without re-splitting.

        Returns
        -------
        list[AudioChunk]
            Ordered list of chunks.  Each chunk's ``.path`` points to a
            temporary WAV file at 16 kHz mono (Whisper's native format).

        Raises
        ------
        FileNotFoundError
            If *audio_path* does not exist.
        RuntimeError
            If ffmpeg is not available or audio loading fails.
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Loading audio: %s", audio_path.name)
        audio = self._load_audio(audio_path)
        total_ms = len(audio)
        total_s = total_ms / 1000
        logger.info("Audio duration: %.1f min (%.0f s)", total_s / 60, total_s)

        # Find all silence regions in one pass — reused for every split decision.
        logger.info("Scanning for silence boundaries (≥%d ms, ≤%.0f dBFS) …",
                    self._config.min_silence_ms, self._config.silence_thresh_dbfs)
        silences = self._detect_silences(audio)
        logger.info("Found %d silence regions.", len(silences))

        cut_points_ms = self._compute_cut_points(total_ms, silences)
        logger.info("Will produce %d chunk(s).", len(cut_points_ms) + 1)

        chunks = self._export_chunks(audio, cut_points_ms, audio_path, total_ms)
        return chunks

    # ── private ──────────────────────────────────────────────────────────────

    @staticmethod
    def _check_ffmpeg() -> None:
        """Raise RuntimeError early if ffmpeg is not on PATH."""
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "ffmpeg not found on PATH.\n"
                "Install it with:  sudo apt install ffmpeg"
            )

    def _load_audio(self, path: Path):  # type: ignore[return]
        """Load audio via pydub and normalise to 16 kHz mono WAV."""
        try:
            from pydub import AudioSegment  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "pydub is required for chunking.  Install: pip install pydub"
            ) from exc

        audio = AudioSegment.from_file(str(path))
        # Normalise to Whisper's expected format: 16 kHz mono
        return audio.set_frame_rate(16_000).set_channels(1)

    def _detect_silences(self, audio) -> list[tuple[int, int]]:  # type: ignore[return]
        """
        Return list of (start_ms, end_ms) silence intervals.

        Uses pydub's ``detect_silence`` which does a simple RMS-based scan.
        """
        from pydub.silence import detect_silence  # noqa: PLC0415

        return detect_silence(
            audio,
            min_silence_len=self._config.min_silence_ms,
            silence_thresh=self._config.silence_thresh_dbfs,
        )

    def _compute_cut_points(
        self, total_ms: int, silences: list[tuple[int, int]]
    ) -> list[int]:
        """
        Determine where to cut the audio (in ms from start).

        Algorithm
        ---------
        Starting from position 0, we advance by ``target_duration_s`` and
        then search for the longest silence within the search window centred
        on the target.  We cut at the midpoint of that silence.

        If no silence is found, we fall back to the ``max_duration_s`` boundary
        and log a warning.
        """
        target_ms = int(self._config.target_duration_s * 1000)
        max_ms = int(self._config.max_duration_s * 1000)
        window_ms = int(self._config.search_window_s * 1000)

        cut_points: list[int] = []
        cursor = 0

        while cursor + target_ms < total_ms:
            search_start = cursor + target_ms - window_ms
            search_end = cursor + target_ms + window_ms

            best = self._best_silence_in_window(silences, search_start, search_end)

            if best is not None:
                # Cut at midpoint of the silence — deepest quiet point.
                cut_ms = (best[0] + best[1]) // 2
                logger.debug(
                    "Cut at %.1f min (silence %.1f–%.1f s)",
                    cut_ms / 60_000,
                    best[0] / 1000,
                    best[1] / 1000,
                )
            else:
                # No silence found — hard cut at max duration.
                cut_ms = cursor + max_ms
                logger.warning(
                    "No silence found near %.1f min — hard cut at %.1f min. "
                    "Reduce silence_thresh_dbfs if this happens often.",
                    (cursor + target_ms) / 60_000,
                    cut_ms / 60_000,
                )

            cut_points.append(cut_ms)
            cursor = cut_ms

        return cut_points

    @staticmethod
    def _best_silence_in_window(
        silences: list[tuple[int, int]], window_start: int, window_end: int
    ) -> tuple[int, int] | None:
        """
        Return the longest silence that overlaps [window_start, window_end].

        Longer silences = safer, more natural cut points.
        """
        candidates = [
            (s, e) for (s, e) in silences
            if s < window_end and e > window_start
        ]
        if not candidates:
            return None
        # Prefer longest silence — most natural pause.
        return max(candidates, key=lambda se: se[1] - se[0])

    def _export_chunks(
        self,
        audio,  # pydub.AudioSegment
        cut_points_ms: list[int],
        source_path: Path,
        total_ms: int,
    ) -> list[AudioChunk]:
        """
        Slice the audio at *cut_points_ms* and write each slice to a temp WAV.

        Each slice includes ``overlap_ms`` of audio before and after the
        actual boundary so Whisper has warm-up context.
        """
        overlap = self._config.overlap_ms
        tmp_dir = self._config.tmp_dir or Path(tempfile.gettempdir()) / "transcriber_chunks"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        # Build boundary list: [(start_ms, end_ms), ...]
        boundaries = [0] + cut_points_ms + [total_ms]
        chunks: list[AudioChunk] = []

        for idx in range(len(boundaries) - 1):
            seg_start = boundaries[idx]
            seg_end = boundaries[idx + 1]

            # Add overlap padding (clamped to file boundaries).
            padded_start = max(0, seg_start - overlap)
            padded_end = min(total_ms, seg_end + overlap)

            overlap_start_ms = seg_start - padded_start
            overlap_end_ms = padded_end - seg_end

            segment = audio[padded_start:padded_end]

            # Export as WAV — Whisper reads WAV natively without ffmpeg overhead.
            chunk_path = tmp_dir / f"{source_path.stem}_chunk{idx:03d}.wav"
            segment.export(str(chunk_path), format="wav")

            chunk = AudioChunk(
                index=idx,
                path=chunk_path,
                start_s=seg_start / 1000,
                end_s=seg_end / 1000,
                overlap_start_ms=overlap_start_ms,
                overlap_end_ms=overlap_end_ms,
            )
            chunks.append(chunk)
            logger.info(str(chunk))

        return chunks
