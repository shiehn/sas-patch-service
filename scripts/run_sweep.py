"""Sweep generation campaigns across every under-covered anchor.

Runs an evolution campaign for each anchor classified sparse/empty in
data/anchor_coverage.json, then verifies survivors (full probes + v1 gates).
Resumable: anchors with an existing campaign.json are skipped.

Usage:
  .venv/bin/python scripts/run_sweep.py [--pop 32] [--gens 10] [--classes sparse,empty]

Writes data/sweep_summary.json when done.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling script imports
from verify_survivors import verify  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=32)
    ap.add_argument("--gens", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--classes", default="sparse,empty")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--anchors", default="", help="comma list overrides class selection")
    ap.add_argument("--tag", default="", help="campaign dir suffix (retry rounds)")
    ap.add_argument("--profile", default="default", choices=["default", "transient"],
                    help="transient: fitness = dominant+staccato probes, FX-weighted mutation")
    args = ap.parse_args()

    coverage = json.loads((DATA_DIR / "anchor_coverage.json").read_text())
    if args.anchors:
        wanted = set(args.anchors.split(","))
        targets = [a for a in coverage["anchors"] if a["id"] in wanted]
    else:
        classes = set(args.classes.split(","))
        targets = [a for a in coverage["anchors"] if a["class"] in classes]
    if args.limit:
        targets = targets[: args.limit]

    vectors = np.load(DATA_DIR / "anchor_vectors.npz", allow_pickle=False)
    anchor_ids = [str(x) for x in vectors["anchor_ids"]]
    negative_vecs = vectors["negative_vecs"]

    corpus = {}
    for l in (DATA_DIR / "corpus.jsonl").read_text().splitlines():
        r = json.loads(l)
        corpus[r["id"]] = r
    import os

    index_dir = DATA_DIR / os.environ.get("SPS_INDEX_DIR", "index")
    pooled_rows = [json.loads(l) for l in (index_dir / "pooled.jsonl").read_text().splitlines()]
    pooled = np.load(index_dir / "pooled.npy")

    from sps.clap_embed import ClapEmbedder
    from sps.evolve import Campaign, CampaignConfig

    embedder = ClapEmbedder()  # loaded ONCE for the whole sweep
    summary = []
    t0 = time.time()
    for n, entry in enumerate(targets, 1):
        anchor_id = entry["id"]
        dir_name = f"{anchor_id}-{args.tag}" if args.tag else anchor_id
        camp_dir = DATA_DIR / "campaigns" / dir_name
        if (camp_dir / "campaign.json").exists():
            print(f"[{n}/{len(targets)}] {anchor_id}: already done, skipping", flush=True)
            continue
        try:
            anchor_vec = vectors["anchor_vecs"][anchor_ids.index(anchor_id)]
            sims = pooled @ anchor_vec
            seeds = []
            for idx in np.argsort(-sims):
                row = corpus.get(pooled_rows[idx]["id"])
                if not row:
                    continue
                fxp = row["fxp"] if row["fxp"].startswith("/") else str(DATA_DIR / row["fxp"])
                if Path(fxp).is_file():
                    seeds.append(fxp)
                if len(seeds) >= args.seeds:
                    break

            print(f"[{n}/{len(targets)}] {anchor_id} ({entry['class']}, ceiling {entry['best_score']})",
                  flush=True)
            fitness_probes = None
            mutation = None
            if args.profile == "transient":
                from sps.params import MutationConfig

                fitness_probes = [entry["dominant_probe"], "staccato-v1"]
                mutation = MutationConfig(group_weights={"FX": 3.0})
            campaign = Campaign(
                CampaignConfig(
                    anchor_id=anchor_id,
                    anchor_text=entry["text"],
                    seeds=seeds,
                    probe_id=entry["dominant_probe"],
                    population=args.pop,
                    generations=args.gens,
                    fitness_probes=fitness_probes,
                    mutation=mutation,
                ),
                anchor_vec=anchor_vec,
                negative_vecs=negative_vecs,
                embedder=embedder,
                out_dir=camp_dir,
                workers=args.workers,
            )
            manifest = campaign.run()
            results = verify(dir_name, embedder, quiet=True)
            passing = [r for r in results if r["verdict"] == "PASS"]
            best_anchor = max((s["anchor_cos"] for s in manifest["survivors"]), default=-1)
            beat = best_anchor > entry["best_score"]
            summary.append({
                "anchor_id": anchor_id, "class": entry["class"],
                "ceiling": entry["best_score"], "evolved_best": round(best_anchor, 4),
                "beat_corpus": beat, "survivors": len(results), "gate_pass": len(passing),
            })
            print(f"    → evolved {best_anchor:.3f} vs ceiling {entry['best_score']:.3f} "
                  f"({'BEAT' if beat else 'below'}); gates {len(passing)}/{len(results)} "
                  f"[{time.time()-t0:.0f}s total]", flush=True)
        except Exception as e:  # noqa: BLE001 — one bad anchor must not kill the sweep
            print(f"    → ERROR {type(e).__name__}: {e}", flush=True)
            traceback.print_exc(limit=2)
            summary.append({"anchor_id": anchor_id, "error": str(e)})

    (DATA_DIR / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    ok = [s for s in summary if "error" not in s]
    beat = [s for s in ok if s.get("beat_corpus")]
    total_pass = sum(s.get("gate_pass", 0) for s in ok)
    print(f"\nSWEEP DONE in {(time.time()-t0)/60:.1f} min: {len(ok)} campaigns, "
          f"{len(beat)} beat their corpus ceiling, {total_pass} gate-passing patches")
    print(f"summary: {DATA_DIR / 'sweep_summary.json'}")


if __name__ == "__main__":
    main()
