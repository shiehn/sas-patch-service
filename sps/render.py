"""Headless probe rendering through surgepy (pinned Surge @ release_xt_1.3.4)."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from . import SURGE_SRC
from .probes import SAMPLE_RATE, Probe

_SURGEPY_DIR = SURGE_SRC / "ignore" / "bpy" / "src" / "surge-python"

_surgepy = None


def get_surgepy():
    global _surgepy
    if _surgepy is None:
        if str(_SURGEPY_DIR) not in sys.path:
            sys.path.insert(0, str(_SURGEPY_DIR))
        import surgepy  # type: ignore
        _surgepy = surgepy
    return _surgepy


def render_probe(
    fxp_path: str,
    probe: Probe,
    param_delta: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """Render one probe for one patch with a fresh synth (max isolation).
    `param_delta` (evolution children) is applied on top of the loaded patch —
    values only; the parent's modulation routing / wavetables are preserved.
    Returns float32 stereo array shape (2, n_samples)."""
    surgepy = get_surgepy()
    synth = surgepy.createSurge(SAMPLE_RATE)
    try:
        synth.loadPatch(fxp_path)
        if param_delta:
            from .params import SurgeParams

            SurgeParams(synth).apply(param_delta)
        block_size = synth.getBlockSize()
        n_blocks = int(math.ceil(probe.total_sec * SAMPLE_RATE / block_size))
        buf = synth.createMultiBlock(n_blocks)

        # events sorted by time, converted to block indices
        events = sorted(probe.events, key=lambda e: (e.t, e.kind == "on"))
        cursor = 0  # block index

        def process_until(block_idx: int) -> None:
            nonlocal cursor
            count = min(block_idx, n_blocks) - cursor
            if count > 0:
                synth.processMultiBlock(buf, cursor, count)
                cursor += count

        for ev in events:
            process_until(int(ev.t * SAMPLE_RATE / block_size))
            if ev.kind == "on":
                synth.playNote(0, ev.note, ev.vel, 0)
            else:
                synth.releaseNote(0, ev.note, 0)
        process_until(n_blocks)
        return np.asarray(buf, dtype=np.float32)
    finally:
        del synth


def probe_stats(audio: np.ndarray) -> Dict[str, float]:
    mono = audio.mean(axis=0)
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size else 0.0
    # activity: fraction of 50ms windows above a low threshold
    win = SAMPLE_RATE // 20
    n_win = max(1, mono.size // win)
    frames = mono[: n_win * win].reshape(n_win, win)
    frame_rms = np.sqrt(np.mean(np.square(frames), axis=1))
    activity = float(np.mean(frame_rms > 1e-4))
    bad = float(np.mean(~np.isfinite(mono)))
    return {"peak": peak, "rms": rms, "activity": activity, "nonfinite_frac": bad}


def to_mono_16bit_ok(audio: np.ndarray) -> Optional[np.ndarray]:
    """Mono mix, sanitized for writing; returns None if numerically broken."""
    mono = audio.mean(axis=0)
    if not np.all(np.isfinite(mono)):
        return None
    peak = np.max(np.abs(mono))
    if peak > 1.0:  # normalize only if clipping; keep natural level otherwise
        mono = mono / peak * 0.98
    return mono.astype(np.float32)
