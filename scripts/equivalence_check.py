"""De-risk item #1 (first half): Tracktion <PLUGIN> wrapper ⇄ .fxp equivalence.

Checks, without needing surgepy:
  A. Every sampled bundled preset's decoded state is a valid Surge 'sub3' stream
     with a parseable <patch> XML document (i.e. wrapper state == fxp chunk format).
  B. juce_b64 round-trip: decode(encode(state)) == state for real states.
  C. wrapper round-trip: encode_wrapper(chunk) → decode_wrapper → identical bytes,
     for real factory .fxp chunks (i.e. we can mint client-ready wrappers from fxp).

The second half (surgepy loads a converted bundled fxp and renders non-silence)
runs inside scripts/render_probes.py once surgepy is built.
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import SAS_APP, SURGE_SRC
from sps import fxp as fxp_mod
from sps import juce_b64
from sps import wrapper as wrapper_mod

PRESET_DIR = SAS_APP / "resources" / "tracktion-presets" / "SurgeXT"
FACTORY = SURGE_SRC / "resources" / "data" / "patches_factory"
N_SAMPLES = 25


def check_a_b() -> int:
    fails = 0
    files = [p for p in sorted(PRESET_DIR.glob("*.json")) if p.name != "manifest.json"]
    rng = random.Random(42)
    sampled = []
    for jp in files:
        presets = list(json.loads(jp.read_text()).items())
        sampled.extend((jp.stem, k, v) for k, v in rng.sample(presets, min(2, len(presets))))
    rng.shuffle(sampled)
    sampled = sampled[:N_SAMPLES]

    for category, name, b64v in sampled:
        tag = f"{category}/{name}"
        try:
            w = wrapper_mod.decode_wrapper_b64(b64v)
            assert w.chunk[:4] == b"sub3", f"chunk does not start with sub3: {w.chunk[:8]!r}"
            stream = wrapper_mod.parse_patch_stream(w.chunk)
            assert stream.xml.lstrip().startswith(b"<?xml"), "no xml decl"
            assert stream.revision is not None, "no <patch revision=>"
            rt = juce_b64.decode(juce_b64.encode(w.raw_state))
            assert rt == w.raw_state, "juce_b64 round-trip mismatch"
            # D: our VC2! container round-trips the chunk
            assert wrapper_mod.unwrap_vst3_state(wrapper_mod.wrap_vst3_state(w.chunk)) == w.chunk, \
                "VC2 container round-trip mismatch"
            print(f"  OK  A+B+D {tag}  rev={stream.revision} bytes={len(w.chunk)} wt={stream.has_wavetables}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  FAIL A+B+D {tag}: {type(e).__name__}: {e}")
    return fails


def check_c() -> int:
    fails = 0
    all_fxp = sorted(FACTORY.rglob("*.fxp"))
    for path in random.Random(7).sample(all_fxp, min(N_SAMPLES, len(all_fxp))):
        tag = str(path.relative_to(FACTORY))
        try:
            f = fxp_mod.read_file(path)
            wrapped_b64 = wrapper_mod.encode_wrapper(f.chunk)
            w = wrapper_mod.decode_wrapper_b64(wrapped_b64)
            assert w.chunk == f.chunk, "wrapper round-trip: chunk != original"
            assert w.attrs.get("uniqueId") == "190e4fbd", f"attrs lost: {w.attrs}"
            print(f"  OK  C   {tag}  chunk={len(f.chunk)}")
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"  FAIL C   {tag}: {type(e).__name__}: {e}")
    return fails


def main() -> None:
    print(f"A+B: bundled wrapper decode + juce_b64 round-trip ({N_SAMPLES} samples)")
    fails = check_a_b()
    print(f"C: factory fxp → client wrapper → bytes round-trip ({N_SAMPLES} samples)")
    fails += check_c()
    print(f"\n{'ALL CHECKS PASSED' if fails == 0 else f'{fails} FAILURES'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
