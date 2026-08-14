"""Acid-program tests: cg_GLOBAL allowlist, mod-matrix delta genes, LFO
temposync XML traits, and the acid probe.

Everything that needs the native surgepy module is marked and skipped cleanly
when it can't load (CI without the built .so); the delta-routing unit test
runs against a fake synth so the MOD/ contract is pinned even without surge.

Run: ./tests/run_tests.sh   (or .venv/bin/python -m pytest tests/ -q)
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "surge" / "ignore" / "bpy" / "src" / "surge-python"))

from sps import fxp as fxp_mod  # noqa: E402
from sps.motion_traits import set_lfo_temposync_stream, set_lfo_temposync_xml  # noqa: E402
from sps.params import (  # noqa: E402
    ACID_MOD_ROUTES,
    MUTABLE_GLOBAL_NAMES,
    MutationConfig,
    SurgeParams,
    mod_key,
    parse_mod_key,
)
from sps.probes import ACID_RIFF, PROBES  # noqa: E402
from sps.wrapper import parse_patch_stream  # noqa: E402

INIT_SAW = ROOT / "third_party/surge/resources/data/patches_factory/Templates/Init Saw.fxp"


def surgepy_or_none():
    try:
        import surgepy  # type: ignore

        return surgepy
    except Exception:
        return None


needs_surgepy = pytest.mark.skipif(surgepy_or_none() is None, reason="surgepy .so not available")

# The .fxp fixture comes from the third_party/surge clone (README setup step
# 2), which is NOT part of this repo — skip cleanly instead of erroring on a
# fresh checkout.
needs_init_saw = pytest.mark.skipif(
    not INIT_SAW.exists(),
    reason="third_party/surge not cloned (README setup step 2) — Init Saw.fxp missing",
)


# ---------------------------------------------------------------------------
# MOD/ key contract (no surgepy needed — fake synth pins the wiring)
# ---------------------------------------------------------------------------

def test_parse_mod_key_roundtrip():
    key = mod_key("velocity", "FILTER/0/2")
    assert key == "MOD/velocity:FILTER/0/2"
    assert parse_mod_key(key) == ("velocity", "FILTER/0/2")
    assert parse_mod_key("FILTER/0/2") is None
    assert parse_mod_key("MOD/broken") is None


# ---------------------------------------------------------------------------
# cg_GLOBAL allowlist + mod genes (real surgepy)
# ---------------------------------------------------------------------------

@needs_surgepy
def test_global_allowlist_exposes_expression_params_only():
    surgepy = surgepy_or_none()
    synth = surgepy.createSurge(48000)
    params = SurgeParams(synth)

    global_specs = [s for s in params.specs if s.group == "GLOBAL"]
    names = {s.name for s in global_specs}
    assert names == set(MUTABLE_GLOBAL_NAMES)

    # Logistics params must NOT leak in.
    all_names = {s.name for s in params.specs}
    for forbidden in ("Global Volume", "Scene Mode", "Split Point", "Polyphony Limit", "Active Scene"):
        assert forbidden not in all_names

    # Latch (5) excluded from Play Mode's mutable range.
    play_modes = [s for s in global_specs if s.name == "Play Mode"]
    assert play_modes and all(s.vmax == 4.0 for s in play_modes)

    # Round-trip: portamento + mono play mode are settable through apply().
    porta = next(s for s in global_specs if s.name == "Portamento")
    mode = next(s for s in global_specs if s.name == "Play Mode")
    params.apply({porta.key: -2.0, mode.key: 1.0})  # mono + audible glide
    assert synth.getParamVal(porta.handle) == pytest.approx(-2.0)
    assert synth.getParamVal(mode.handle) == pytest.approx(1.0)


@needs_surgepy
def test_mod_route_applies_velocity_to_cutoff_depth():
    surgepy = surgepy_or_none()
    synth = surgepy.createSurge(48000)
    synth.loadPatch(str(INIT_SAW))
    params = SurgeParams(synth)

    cutoff = params.find_first_by_name("Filter 1 Cutoff")
    assert cutoff is not None and cutoff.key.startswith("FILTER/0/")

    from surgepy import constants as sc  # type: ignore

    src = synth.getModSource(sc.ms_velocity)
    assert synth.getModDepth01(cutoff.handle, src) == pytest.approx(0.0)

    params.apply({mod_key("velocity", cutoff.key): 0.42})
    assert synth.getModDepth01(cutoff.handle, src) == pytest.approx(0.42, abs=1e-3)

    # Unknown source / target must be a silent no-op, never a crash.
    params.apply({mod_key("nosuchsource", cutoff.key): 0.3, mod_key("velocity", "NOPE/0/0"): 0.3})


@needs_surgepy
def test_mutation_can_emit_mod_route_gene():
    import random

    surgepy = surgepy_or_none()
    synth = surgepy.createSurge(48000)
    synth.loadPatch(str(INIT_SAW))
    params = SurgeParams(synth)

    from sps.params import mutate_values

    config = MutationConfig(mod_route_prob=1.0)
    parent = params.snapshot()
    rng = random.Random(1234)
    for _ in range(8):
        delta = mutate_values(params, parent, rng, config)
        mod_keys = [k for k in delta if k.startswith("MOD/")]
        if mod_keys:
            route_names = {r.target_name for r in ACID_MOD_ROUTES}
            assert route_names == {"Filter 1 Cutoff"}
            for k in mod_keys:
                source, param_key = parse_mod_key(k)
                assert source in {r.source for r in ACID_MOD_ROUTES}
                spec = params.by_key[param_key]
                assert spec.name == "Filter 1 Cutoff"
                lo = min(r.lo for r in ACID_MOD_ROUTES)
                hi = max(r.hi for r in ACID_MOD_ROUTES)
                assert lo <= delta[k] <= hi
            break
    else:
        pytest.fail("mod_route_prob=1.0 never produced a MOD/ gene in 8 children")


# ---------------------------------------------------------------------------
# temposync XML trait
# ---------------------------------------------------------------------------

@needs_init_saw
def test_temposync_xml_sets_and_clears_attribute():
    f = fxp_mod.read_file(str(INIT_SAW))
    parsed = parse_patch_stream(f.chunk)

    synced = set_lfo_temposync_xml(parsed.xml, lfo=1, enabled=True)
    assert b'<a_lfo0_rate' in synced
    node = synced[synced.find(b"<a_lfo0_rate"):]
    node = node[: node.find(b"/>") + 2]
    assert b'temposync="1"' in node
    # Idempotent + other LFOs untouched.
    assert synced.count(b'temposync="1"') == 1
    resynced = set_lfo_temposync_xml(synced, lfo=1, enabled=True)
    assert resynced.count(b'temposync="1"') == 1

    cleared = set_lfo_temposync_xml(synced, lfo=1, enabled=False)
    assert b"temposync" not in cleared

    with pytest.raises(ValueError):
        set_lfo_temposync_xml(parsed.xml, lfo=7)
    with pytest.raises(ValueError):
        set_lfo_temposync_xml(b"<not-a-patch/>", lfo=1)


@needs_init_saw
def test_temposync_stream_recomputes_header_and_preserves_tail():
    f = fxp_mod.read_file(str(INIT_SAW))
    out = set_lfo_temposync_stream(f.chunk, lfos=(1,))

    reparsed = parse_patch_stream(out)
    assert b'temposync="1"' in reparsed.xml
    assert int.from_bytes(out[4:8], "little", signed=True) == len(reparsed.xml)
    # wtsize header fields verbatim
    assert out[8:32] == f.chunk[8:32]


@needs_surgepy
def test_surge_loads_temposynced_patch_and_rate_reads_as_note_value(tmp_path):
    from sps.motion_traits import apply_to_fxp_file

    surgepy = surgepy_or_none()
    out_fxp = tmp_path / "init-saw-synced.fxp"
    apply_to_fxp_file(str(INIT_SAW), str(out_fxp), lfos=(1,))

    synth = surgepy.createSurge(48000)
    synth.loadPatch(str(out_fxp))
    params = SurgeParams(synth)
    rate = params.find_first_by_name("LFO 1 Rate")
    assert rate is not None
    display = str(synth.getParamDisplay(rate.handle))
    # A temposynced rate displays as a note division ("1/4", "1/8 D", ...),
    # never a Hz figure.
    assert "/" in display and "Hz" not in display


# ---------------------------------------------------------------------------
# acid probe
# ---------------------------------------------------------------------------

def test_acid_probe_shape():
    assert ACID_RIFF in PROBES
    assert ACID_RIFF.id == "acid-riff-v1"

    ons = sorted((e.t, e.note, e.vel) for e in ACID_RIFF.events if e.kind == "on")
    offs = {(e.note, e.t) for e in ACID_RIFF.events if e.kind == "off"}
    assert offs and ons

    # Bar 1 must contain overlapped DIFFERENT-pitch pairs (the slide gesture):
    # some note's off lands after the next note's on, and adjacent pitches differ.
    overlaps = 0
    for i in range(len(ons) - 1):
        t_on, note, _ = ons[i]
        t_next, next_note, _ = ons[i + 1]
        # earliest off for THIS voicing of the note (offs is unordered)
        off_t = min((t for (n, t) in offs if n == note and t > t_on), default=None)
        if off_t is not None and off_t > t_next:
            assert note != next_note, "same-pitch overlap is retrigger noise, not a slide"
            overlaps += 1
    assert overlaps >= 4

    # Accent contrast for velocity-routing fitness.
    vels = [v for (_, _, v) in ons]
    assert max(vels) - min(vels) >= 48


@needs_surgepy
def test_acid_probe_renders_audio():
    from sps.render import probe_stats, render_probe

    audio = render_probe(str(INIT_SAW), ACID_RIFF)
    stats = probe_stats(audio)
    assert stats["nonfinite_frac"] == 0.0
    assert stats["activity"] > 0.3
    assert stats["peak"] > 1e-3
