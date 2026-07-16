"""Probe suite v1 — deterministic MIDI programs rendered per patch.

A patch is a conditional sound-producing system; each probe observes it played a
different way (proposal §6.3). Times in seconds, 120 BPM feel, 48 kHz renders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

SAMPLE_RATE = 48000


@dataclass(frozen=True)
class NoteEvent:
    t: float          # seconds
    kind: str         # 'on' | 'off'
    note: int         # midi
    vel: int = 100


@dataclass(frozen=True)
class Probe:
    id: str
    events: Tuple[NoteEvent, ...]
    total_sec: float  # render length incl. tail


def _riff(notes_vels: List[Tuple[int, int]], start: float, step: float, gate: float) -> List[NoteEvent]:
    out: List[NoteEvent] = []
    t = start
    for note, vel in notes_vels:
        out.append(NoteEvent(t, "on", note, vel))
        out.append(NoteEvent(t + gate, "off", note))
        t += step
    return out


def _held(chord: List[int], on: float, off: float, vel: int = 95) -> List[NoteEvent]:
    return [NoteEvent(on, "on", n, vel) for n in chord] + [NoteEvent(off, "off", n) for n in chord]


# 120 BPM: 8th = 0.25 s. Two bars of a syncopated low riff (C1 root).
BASS_RIFF = Probe(
    id="bass-riff-v1",
    events=tuple(
        _riff([(24, 112), (24, 88), (31, 104), (24, 88), (27, 108), (24, 88), (34, 100), (31, 92)],
              start=0.0, step=0.25, gate=0.18)
        + _riff([(24, 112), (24, 88), (31, 104), (24, 88), (27, 108), (29, 96), (27, 100), (24, 112)],
                start=2.0, step=0.25, gate=0.18)
    ),
    total_sec=6.0,
)

LOW_SUSTAIN = Probe(
    id="low-sustain-v1",
    events=tuple(_held([36], on=0.0, off=4.0, vel=100)),
    total_sec=6.0,
)

MID_PHRASE = Probe(
    id="mid-phrase-v1",
    events=tuple(
        _riff([(60, 96), (62, 90), (64, 100), (67, 106)], start=0.0, step=0.5, gate=0.42)
        + _riff([(64, 92), (62, 88)], start=2.0, step=0.5, gate=0.42)
        + _held([60], on=3.0, off=4.25, vel=98)
    ),
    total_sec=6.0,
)

PAD_CHORD = Probe(
    id="pad-chord-v1",
    events=tuple(_held([48, 52, 55, 59], on=0.0, off=5.0, vel=90)),
    total_sec=8.0,
)

# v2 (2026-07-16): transient character across three registers — 90 ms gates expose
# attack/click/decay precision that the phrase probes average away. Added for the
# GATE-2 loss-family iteration (perc one-shots, keys). Older corpus entries simply
# lack this observation; per-patch obs maps make mixed probe sets safe.
STACCATO = Probe(
    id="staccato-v1",
    events=tuple(
        _riff([(36, 118), (36, 74), (48, 118), (48, 74), (60, 112), (72, 104)],
              start=0.0, step=0.5, gate=0.09)
    ),
    total_sec=4.5,
)

PROBES: Tuple[Probe, ...] = (BASS_RIFF, LOW_SUSTAIN, MID_PHRASE, PAD_CHORD, STACCATO)
