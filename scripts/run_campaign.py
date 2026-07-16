"""Run one anchor-conditioned generation campaign.

Usage:
  .venv/bin/python scripts/run_campaign.py brass-trumpet-muted [--pop 24] [--gens 8] [--seeds 6]

Reads the anchor + its coverage entry (dominant probe, top seed patches) from
data/anchor_coverage.json (run scripts/anchor_coverage.py first). Survivors land
in data/campaigns/<anchor-id>/ as .fxp + campaign.json manifest, then get a
verification render through ALL probes with anchor-vocab clarity metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR

INDEX = DATA_DIR / os.environ.get("SPS_INDEX_DIR", "index")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("anchor_id")
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gens", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    coverage = json.loads((DATA_DIR / "anchor_coverage.json").read_text())
    entry = next((a for a in coverage["anchors"] if a["id"] == args.anchor_id), None)
    if not entry:
        sys.exit(f"anchor {args.anchor_id!r} not in data/anchor_coverage.json")

    vectors = np.load(DATA_DIR / "anchor_vectors.npz", allow_pickle=False)
    anchor_ids = [str(x) for x in vectors["anchor_ids"]]
    anchor_vec = vectors["anchor_vecs"][anchor_ids.index(args.anchor_id)]
    negative_vecs = vectors["negative_vecs"]

    # seeds: the anchor's nearest existing patches, resolved to fxp paths
    corpus = {}
    for l in (DATA_DIR / "corpus.jsonl").read_text().splitlines():
        r = json.loads(l)
        corpus[r["id"]] = r
    pooled_rows = [json.loads(l) for l in (INDEX / "pooled.jsonl").read_text().splitlines()]
    pooled = np.load(INDEX / "pooled.npy")
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

    from sps.clap_embed import ClapEmbedder
    from sps.evolve import Campaign, CampaignConfig

    print(f'campaign: "{entry["text"]}" ({args.anchor_id})')
    print(f"  class={entry['class']} best-existing={entry['best_score']} probe={entry['dominant_probe']}")
    print(f"  seeds: {[Path(s).stem for s in seeds]}")

    embedder = ClapEmbedder()  # MPS if available — batch embedding per generation
    campaign = Campaign(
        CampaignConfig(
            anchor_id=args.anchor_id,
            anchor_text=entry["text"],
            seeds=seeds,
            probe_id=entry["dominant_probe"],
            population=args.pop,
            generations=args.gens,
        ),
        anchor_vec=anchor_vec,
        negative_vecs=negative_vecs,
        embedder=embedder,
        out_dir=DATA_DIR / "campaigns" / args.anchor_id,
        workers=args.workers,
    )
    manifest = campaign.run()

    best_anchor = max((s["anchor_cos"] for s in manifest["survivors"]), default=-1)
    print(f"\nbest evolved anchor-similarity: {best_anchor}  vs existing-corpus ceiling: {entry['best_score']}"
          f"  → {'BEAT the corpus' if best_anchor > entry['best_score'] else 'below corpus ceiling'}")
    print(f"best-ever penalized fitness: {manifest['best_ever']} (anchor − {campaign.config.neg_weight}·neg)")
    print("survivors:")
    for s in manifest["survivors"]:
        print(f"  #{s['rank']} {s['id']:<34} fitness={s['fitness']} anchor={s['anchor_cos']} "
              f"neg={s['neg_cos']} Δ{s['delta_size']} ({s['op']})")
    print(f"\nmanifest: {DATA_DIR / 'campaigns' / args.anchor_id / 'campaign.json'}")


if __name__ == "__main__":
    main()
