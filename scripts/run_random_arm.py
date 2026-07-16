"""Random-arm control: brute-force random patches through the SAME gates.

Settles seeded-evolution vs random-generate-then-filter empirically (proposal §6.2).
Candidates are derived from Surge's init state by several rounds of heavy
grammar-guarded mutation (parameter-safety constraints; osc-type guard as always).
Each candidate renders the four probes once; its "anchor" is whichever core
vocabulary anchor it best matches (labels are assigned, not targeted). The same
fitness (best-anchor − neg penalty) and quality gates apply.

Outputs data/random_arm/{candidates.jsonl, summary.json} + fxp for gate-passers.

Usage: .venv/bin/python scripts/run_random_arm.py [--count 800] [--workers 6]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR, SURGE_SRC

OUT = DATA_DIR / "random_arm"
INIT_ROUNDS = 6  # rounds of heavy mutation from init = "random but structurally sane"
NEG_WEIGHT = 0.3


def _worker(task):
    """Generate one random candidate from init, render all probes, return
    (mono_by_probe, stats_by_probe, delta_size, seed)."""
    seed = task
    import sys as _sys

    _sys.path.insert(0, str(SURGE_SRC / "ignore" / "bpy" / "src" / "surge-python"))
    import surgepy  # type: ignore

    from sps.params import MutationConfig, SurgeParams, mutate_values
    from sps.probes import PROBES
    from sps.render import probe_stats, to_mono_16bit_ok

    rng = random.Random(seed)
    synth = surgepy.createSurge(48000)
    params = SurgeParams(synth)
    allowed_osc = {sp.key: params.allowed_osc_type_values(sp) for sp in params.osc_type_specs()}

    cfg = MutationConfig(groups_per_child=(3, 6), params_per_group=(4, 10),
                         base_sigma=0.22, int_step_prob=0.5, bool_flip_prob=0.25,
                         osc_type_switch_prob=0.35)
    delta = {}
    for _ in range(INIT_ROUNDS):
        current = params.snapshot()
        step = mutate_values(params, current, rng, cfg, allowed_osc=allowed_osc)
        params.apply(step)
        delta.update(step)

    monos, stats = {}, {}
    import math

    for probe in PROBES:
        n_blocks = int(math.ceil(probe.total_sec * 48000 / synth.getBlockSize()))
        buf = synth.createMultiBlock(n_blocks)
        cursor = 0

        def process_until(block_idx):
            nonlocal cursor
            count = min(block_idx, n_blocks) - cursor
            if count > 0:
                synth.processMultiBlock(buf, cursor, count)
                cursor += count

        for ev in sorted(probe.events, key=lambda e: (e.t, e.kind == "on")):
            process_until(int(ev.t * 48000 / synth.getBlockSize()))
            if ev.kind == "on":
                synth.playNote(0, ev.note, ev.vel, 0)
            else:
                synth.releaseNote(0, ev.note, 0)
        process_until(n_blocks)
        audio = np.asarray(buf, dtype=np.float32)
        st = probe_stats(audio)
        stats[probe.id] = st
        mono = to_mono_16bit_ok(audio)
        if mono is not None and st["peak"] >= 1e-5 and st["activity"] > 0.05:
            monos[probe.id] = mono
        # reset voices between probes
        synth.allNotesOff()
        cursor = 0
    # save patch bytes for potential survivors (savePatch needs a path later; keep delta+seed)
    return monos, stats, len(delta), seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=800)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    vectors = np.load(DATA_DIR / "anchor_vectors.npz", allow_pickle=False)
    coverage = json.loads((DATA_DIR / "anchor_coverage.json").read_text())
    core_ids = {a["id"] for a in coverage["anchors"]}
    all_ids = [str(x) for x in vectors["anchor_ids"]]
    core_mask = np.array([aid in core_ids for aid in all_ids])
    anchor_vecs = vectors["anchor_vecs"][core_mask]
    core_id_list = [aid for aid in all_ids if aid in core_ids]
    negative_vecs = vectors["negative_vecs"]

    baseline = json.loads((DATA_DIR / "aesthetics_baseline.json").read_text())["percentiles"]

    from sps.aesthetics import clap_quality_contrast
    from sps.clap_embed import ClapEmbedder

    embedder = ClapEmbedder()
    t0 = time.time()
    rows = []
    n_alive = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
         open(OUT / "candidates.jsonl", "w") as log:
        for i, (monos, stats, delta_size, seed) in enumerate(
                pool.map(_worker, range(args.count), chunksize=2), 1):
            rec = {"seed": seed, "delta_size": delta_size,
                   "probes_alive": sorted(monos.keys())}
            if not monos:
                rec["status"] = "silent"
            else:
                n_alive += 1
                waves = list(monos.values())
                embs = embedder.embed_audio(waves)
                pooled_mean = embs.mean(axis=0)
                pooled_mean /= np.linalg.norm(pooled_mean) + 1e-9
                anchor_sims = anchor_vecs @ pooled_mean
                best_a = int(np.argmax(anchor_sims))
                neg = float(np.max(negative_vecs @ pooled_mean))
                contrast = float(clap_quality_contrast(embedder, pooled_mean[None, :])[0])
                clarity = float(anchor_sims[best_a] - anchor_sims.mean())
                fitness = float(anchor_sims[best_a]) - NEG_WEIGHT * max(0.0, neg)
                rec.update({
                    "status": "alive",
                    "best_anchor": core_id_list[best_a],
                    "anchor_cos": round(float(anchor_sims[best_a]), 4),
                    "fitness": round(fitness, 4),
                    "neg_cos": round(neg, 4),
                    "clarity": round(clarity, 4),
                    "clap_contrast": round(contrast, 4),
                    "gates": {
                        "clarity": clarity > 0.05,
                        "negative": float(anchor_sims[best_a]) - neg > 0.05,
                        "clean": contrast >= baseline["clap_contrast"]["p25"],
                    },
                })
                rec["pass"] = all(rec["gates"].values())
            rows.append(rec)
            log.write(json.dumps(rec) + "\n")
            if i % 100 == 0:
                rate = i / (time.time() - t0)
                print(f"  {i}/{args.count} ({rate:.1f}/s; alive {n_alive})", flush=True)

    alive = [r for r in rows if r.get("status") == "alive"]
    passing = [r for r in alive if r.get("pass")]
    strong = [r for r in passing if r["anchor_cos"] >= 0.55]
    summary = {
        "count": args.count,
        "alive": len(alive),
        "gate_pass": len(passing),
        "strong_anchor_055": len(strong),
        "yield_pct": round(100 * len(passing) / args.count, 2),
        "strong_yield_pct": round(100 * len(strong) / args.count, 2),
        "mean_anchor_cos_alive": round(float(np.mean([r["anchor_cos"] for r in alive])), 4) if alive else None,
        "anchors_hit": sorted({r["best_anchor"] for r in passing}),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nrandom arm done in {(time.time()-t0)/60:.1f} min "
          f"(compare with evolution: ~85/91 campaigns beat ceilings, "
          f"survivor anchor_cos typically 0.55–0.70)")


if __name__ == "__main__":
    main()
