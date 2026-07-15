"""Query the local index: text → top-k patches.

Usage:
  .venv/bin/python scripts/search_cli.py "hollow standup bass" [-k 8] [--category Bass]
  ... --probe bass-riff-v1     weight retrieval toward one probe
  ... --play                   afplay the #1 result's best probe render

First-stage: cosine over per-patch pooled vectors. Then per-observation rerank:
each candidate is scored by its best-matching probe (or a chosen probe).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR

INDEX = DATA_DIR / os.environ.get("SPS_INDEX_DIR", "index")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("-k", type=int, default=8)
    ap.add_argument("--category", default="", help="substring filter on category")
    ap.add_argument("--source", default="", help="bundled|factory|third_party")
    ap.add_argument("--probe", default="", help="rerank by this probe only")
    ap.add_argument("--play", action="store_true")
    ap.add_argument("--template", default="this is the sound of {q}",
                    help="zero-shot text template (golden-set winner); '{q}' for raw")
    args = ap.parse_args()

    from sps.clap_embed import ClapEmbedder

    pooled = np.load(INDEX / "pooled.npy")
    rows = [json.loads(l) for l in (INDEX / "pooled.jsonl").read_text().splitlines()]
    obs = np.load(INDEX / "obs.npy")
    obs_rows = [json.loads(l) for l in (INDEX / "obs.jsonl").read_text().splitlines()]

    mask = np.ones(len(rows), dtype=bool)
    if args.category:
        mask &= np.array([args.category.lower() in r["category"].lower() for r in rows])
    if args.source:
        mask &= np.array([r["source"] == args.source for r in rows])

    embedder = ClapEmbedder(device="cpu")  # query path mirrors the future service: CPU
    q = embedder.embed_text([args.template.format(q=args.query)])[0]

    sims = pooled @ q
    sims[~mask] = -1.0
    top = np.argsort(-sims)[: max(args.k * 5, 40)]  # candidates for rerank

    scored = []
    for p in top:
        if sims[p] <= -1.0:
            continue
        r = rows[p]
        obs_scores = {obs_rows[i]["probe"]: float(obs[i] @ q) for i in r["obs_idx"]}
        rerank = obs_scores.get(args.probe) if args.probe else max(obs_scores.values())
        if rerank is None:
            continue
        best_probe = args.probe or max(obs_scores, key=obs_scores.get)
        scored.append((0.5 * float(sims[p]) + 0.5 * rerank, r, obs_scores, best_probe))
    scored.sort(key=lambda t: -t[0])
    scored = scored[: args.k]

    print(f'\nquery: "{args.query}"' + (f"  [category~{args.category}]" if args.category else ""))
    print(f"{'#':>2} {'score':>6}  {'name':<34} {'category':<22} {'source':<11} best-probe")
    for rank, (score, r, obs_scores, best_probe) in enumerate(scored, 1):
        print(f"{rank:>2} {score:6.3f}  {r['name']:<34.34} {r['category']:<22.22} "
              f"{r['source']:<11} {best_probe}  "
              f"({' '.join(f'{p.split('-')[0]}:{s:.2f}' for p, s in sorted(obs_scores.items()))})")

    if args.play and scored:
        _, r, _, best_probe = scored[0]
        flac = DATA_DIR / "renders" / r["id"].replace("/", "__").replace(" ", "_") / f"{best_probe}.flac"
        print(f"\nplaying {flac} ...")
        subprocess.run(["afplay", str(flac)], check=False)


if __name__ == "__main__":
    main()
