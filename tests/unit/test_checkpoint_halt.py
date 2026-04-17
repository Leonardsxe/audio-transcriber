"""
test_checkpoint_halt.py — Tests for CheckpointManager and HaltController
=========================================================================
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from transcriber.audio.checkpoint import (
    CheckpointManager,
    _dict_to_result,
    _result_to_dict,
)
from transcriber.audio.chunker import AudioChunk
from transcriber.audio.halt import HaltController
from transcriber.protocols import TranscriptionResult, TranscriptionSegment


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────


def _seg(start: float, end: float, text: str) -> TranscriptionSegment:
    return TranscriptionSegment(start=start, end=end, text=text, confidence=0.9)


def _result(*segs: TranscriptionSegment) -> TranscriptionResult:
    return TranscriptionResult(
        segments=list(segs),
        language="es",
        duration=segs[-1].end if segs else 0.0,
    )


def _chunk(idx: int, start: float = 0.0, end: float = 60.0) -> AudioChunk:
    return AudioChunk(
        index=idx,
        path=Path(f"/tmp/chunk_{idx:03d}.wav"),
        start_s=start,
        end_s=end,
    )


def _fake_source(tmp_path: Path, content: bytes = b"\x00" * 128) -> Path:
    src = tmp_path / "interview.mp3"
    src.write_bytes(content)
    return src


# ─────────────────────────────────────────────
#  Serialisation round-trip
# ─────────────────────────────────────────────


class TestSerialisationRoundTrip:
    def test_result_round_trip(self) -> None:
        original = _result(_seg(0.0, 2.5, "hola mundo"), _seg(3.0, 5.0, "adiós"))
        restored = _dict_to_result(_result_to_dict(original))

        assert len(restored.segments) == 2
        assert restored.segments[0].text == "hola mundo"
        assert restored.segments[1].start == pytest.approx(3.0)
        assert restored.language == "es"

    def test_empty_segments(self) -> None:
        original = TranscriptionResult(segments=[], language="es", duration=0.0)
        restored = _dict_to_result(_result_to_dict(original))
        assert restored.segments == []

    def test_spanish_characters_preserved(self) -> None:
        original = _result(_seg(0, 1, "¿Cómo estás, señorita?"))
        restored = _dict_to_result(_result_to_dict(original))
        assert restored.segments[0].text == "¿Cómo estás, señorita?"


# ─────────────────────────────────────────────
#  CheckpointManager — basic lifecycle
# ─────────────────────────────────────────────


class TestCheckpointManagerLifecycle:
    def test_is_fresh_when_no_file(self, tmp_path: Path) -> None:
        src = _fake_source(tmp_path)
        mgr = CheckpointManager(src, checkpoint_dir=tmp_path)
        assert mgr.is_fresh is True

    def test_load_returns_false_when_no_file(self, tmp_path: Path) -> None:
        src = _fake_source(tmp_path)
        mgr = CheckpointManager(src, checkpoint_dir=tmp_path)
        assert mgr.load(total_chunks=3) is False

    def test_save_creates_file(self, tmp_path: Path) -> None:
        src = _fake_source(tmp_path)
        mgr = CheckpointManager(src, checkpoint_dir=tmp_path)
        mgr.load(total_chunks=3)
        mgr.save_chunk(_chunk(0), _result(_seg(0, 5, "hola")))
        assert mgr.checkpoint_path.exists()

    def test_delete_removes_file(self, tmp_path: Path) -> None:
        src = _fake_source(tmp_path)
        mgr = CheckpointManager(src, checkpoint_dir=tmp_path)
        mgr.load(total_chunks=1)
        mgr.save_chunk(_chunk(0), _result(_seg(0, 5, "test")))
        mgr.delete()
        assert not mgr.checkpoint_path.exists()

    def test_completed_count_increments(self, tmp_path: Path) -> None:
        src = _fake_source(tmp_path)
        mgr = CheckpointManager(src, checkpoint_dir=tmp_path)
        mgr.load(total_chunks=3)
        assert mgr.completed_count == 0
        mgr.save_chunk(_chunk(0), _result(_seg(0, 5, "a")))
        assert mgr.completed_count == 1
        mgr.save_chunk(_chunk(1), _result(_seg(5, 10, "b")))
        assert mgr.completed_count == 2


# ─────────────────────────────────────────────
#  CheckpointManager — is_done / resume
# ─────────────────────────────────────────────


class TestCheckpointManagerResume:
    def _write_and_reload(
        self, tmp_path: Path, src: Path, chunks_done: list[int], total: int
    ) -> CheckpointManager:
        """Helper: save N chunks then create a fresh manager and load."""
        mgr = CheckpointManager(src, checkpoint_dir=tmp_path)
        mgr.load(total_chunks=total)
        for idx in chunks_done:
            mgr.save_chunk(_chunk(idx), _result(_seg(0, 5, f"chunk {idx}")))

        # New manager — simulates a fresh process
        mgr2 = CheckpointManager(src, checkpoint_dir=tmp_path)
        mgr2.load(total_chunks=total)
        return mgr2

    def test_is_done_returns_true_for_saved_chunks(self, tmp_path: Path) -> None:
        src = _fake_source(tmp_path)
        mgr2 = self._write_and_reload(tmp_path, src, [0, 1], total=3)
        assert mgr2.is_done(_chunk(0)) is True
        assert mgr2.is_done(_chunk(1)) is True
        assert mgr2.is_done(_chunk(2)) is False

    def test_results_restored_after_reload(self, tmp_path: Path) -> None:
        src = _fake_source(tmp_path)
        mgr2 = self._write_and_reload(tmp_path, src, [0], total=2)
        result = mgr2.get_result(_chunk(0))
        assert result is not None
        assert result.segments[0].text == "chunk 0"

    def test_load_returns_true_on_valid_checkpoint(self, tmp_path: Path) -> None:
        src = _fake_source(tmp_path)
        mgr = CheckpointManager(src, checkpoint_dir=tmp_path)
        mgr.load(total_chunks=2)
        mgr.save_chunk(_chunk(0), _result(_seg(0, 5, "a")))

        mgr2 = CheckpointManager(src, checkpoint_dir=tmp_path)
        assert mgr2.load(total_chunks=2) is True

    def test_size_mismatch_starts_fresh(self, tmp_path: Path) -> None:
        src = _fake_source(tmp_path, content=b"\x00" * 100)
        mgr = CheckpointManager(src, checkpoint_dir=tmp_path)
        mgr.load(total_chunks=2)
        mgr.save_chunk(_chunk(0), _result(_seg(0, 5, "a")))

        # Change file size
        src.write_bytes(b"\x00" * 200)

        mgr2 = CheckpointManager(src, checkpoint_dir=tmp_path)
        assert mgr2.load(total_chunks=2) is False

    def test_chunk_count_mismatch_starts_fresh(self, tmp_path: Path) -> None:
        src = _fake_source(tmp_path)
        mgr = CheckpointManager(src, checkpoint_dir=tmp_path)
        mgr.load(total_chunks=3)
        mgr.save_chunk(_chunk(0), _result(_seg(0, 5, "a")))

        # Reload with different chunk count (file was re-split)
        mgr2 = CheckpointManager(src, checkpoint_dir=tmp_path)
        assert mgr2.load(total_chunks=5) is False  # different total → fresh

    def test_corrupted_json_starts_fresh(self, tmp_path: Path) -> None:
        src = _fake_source(tmp_path)
        mgr = CheckpointManager(src, checkpoint_dir=tmp_path)
        mgr.load(total_chunks=2)
        mgr.save_chunk(_chunk(0), _result(_seg(0, 5, "a")))

        # Corrupt the checkpoint
        mgr.checkpoint_path.write_text("NOT JSON", encoding="utf-8")

        mgr2 = CheckpointManager(src, checkpoint_dir=tmp_path)
        assert mgr2.load(total_chunks=2) is False

    def test_all_results_in_order(self, tmp_path: Path) -> None:
        src = _fake_source(tmp_path)
        mgr = CheckpointManager(src, checkpoint_dir=tmp_path)
        mgr.load(total_chunks=3)
        chunks = [_chunk(i, i * 60, (i + 1) * 60) for i in range(3)]
        results = [_result(_seg(0, 5, f"text {i}")) for i in range(3)]
        for c, r in zip(chunks, results):
            mgr.save_chunk(c, r)

        ordered = mgr.all_results_in_order(chunks)
        assert [r.segments[0].text for r in ordered] == ["text 0", "text 1", "text 2"]

    def test_all_results_raises_if_missing(self, tmp_path: Path) -> None:
        src = _fake_source(tmp_path)
        mgr = CheckpointManager(src, checkpoint_dir=tmp_path)
        mgr.load(total_chunks=2)
        mgr.save_chunk(_chunk(0), _result(_seg(0, 5, "a")))
        # chunk 1 never saved
        with pytest.raises(RuntimeError, match="no cached result"):
            mgr.all_results_in_order([_chunk(0), _chunk(1)])

    def test_atomic_write_checkpoint_valid_json(self, tmp_path: Path) -> None:
        """Checkpoint file must always be valid JSON even under rapid writes."""
        src = _fake_source(tmp_path)
        mgr = CheckpointManager(src, checkpoint_dir=tmp_path)
        mgr.load(total_chunks=5)
        for i in range(5):
            mgr.save_chunk(_chunk(i), _result(_seg(i * 10, i * 10 + 5, f"seg {i}")))
            data = json.loads(mgr.checkpoint_path.read_text(encoding="utf-8"))
            assert isinstance(data, dict)


# ─────────────────────────────────────────────
#  HaltController
# ─────────────────────────────────────────────


class TestHaltController:
    def test_not_halted_initially(self) -> None:
        ctrl = HaltController(prompt_keyboard=False)
        assert ctrl.should_halt() is False
        assert ctrl.was_halted() is False

    def test_trigger_sets_flag(self) -> None:
        ctrl = HaltController(prompt_keyboard=False)
        ctrl.trigger("test")
        assert ctrl.should_halt() is True
        assert ctrl.halt_reason == "test"

    def test_install_uninstall_restores_signals(self) -> None:
        import signal as _signal
        original = _signal.getsignal(_signal.SIGINT)

        ctrl = HaltController(prompt_keyboard=False)
        ctrl.install()
        assert _signal.getsignal(_signal.SIGINT) != original
        ctrl.uninstall()
        assert _signal.getsignal(_signal.SIGINT) == original

    def test_signal_triggers_halt(self) -> None:
        import signal as _signal
        ctrl = HaltController(prompt_keyboard=False)
        ctrl.install()
        # Simulate SIGINT programmatically
        _signal.raise_signal(_signal.SIGINT)
        ctrl.uninstall()
        assert ctrl.was_halted() is True
        assert "Ctrl+C" in ctrl.halt_reason

    def test_summary_output(self, tmp_path: Path) -> None:
        src = _fake_source(tmp_path)
        mgr = CheckpointManager(src, checkpoint_dir=tmp_path)
        mgr.load(total_chunks=10)
        for i in range(4):
            mgr.save_chunk(_chunk(i), _result(_seg(0, 1, "x")))
        summary = mgr.summary()
        assert "4/10" in summary
        assert "40%" in summary
