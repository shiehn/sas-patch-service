"""Anchor coverage scan — where is the corpus strong, sparse, or empty?

Embeds every anchor (prompt-ensembled: template variants averaged), scores it
against the existing patch index, and classifies coverage. The weakest anchors
are the generation campaign targets (anchor-conditioned seeding); the strongest
need no generation at all.

Outputs data/anchor_coverage.json and prints the table weakest-first.

Usage: .venv/bin/python scripts/anchor_coverage.py [--top 3]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR, REPO_ROOT

INDEX = DATA_DIR / os.environ.get("SPS_INDEX_DIR", "index")

COVERED = 0.60   # best existing patch this close → no campaign needed
SPARSE = 0.48    # between: adjacent material exists, campaign worthwhile


def classify(score: float) -> str:
    if score >= COVERED:
        return "covered"
    if score >= SPARSE:
        return "sparse"
    return "empty"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--vocab", default="anchors_v2.json")
    ap.add_argument("--tier", default="core", help="core | research | all")
    args = ap.parse_args()

    from sps.clap_embed import ClapEmbedder

    vocab = json.loads((REPO_ROOT / "eval" / args.vocab).read_text())
    all_anchors = vocab["anchors"]
    # report/campaign-target list respects --tier; the vector store always carries
    # EVERY anchor so verification of any campaign keeps working across tiers
    anchors = [a for a in all_anchors
               if args.tier == "all" or a.get("tier", "core") == args.tier]
    templates = vocab["templates"]

    pooled = np.load(INDEX / "pooled.npy")
    rows = [json.loads(l) for l in (INDEX / "pooled.jsonl").read_text().splitlines()]
    obs = np.load(INDEX / "obs.npy")
    obs_rows = [json.loads(l) for l in (INDEX / "obs.jsonl").read_text().splitlines()]

    embedder = ClapEmbedder(device="cpu")

    # prompt-ensembled anchor vectors: embed every (template × anchor), average per anchor
    texts = [t.format(a=a["text"]) for a in all_anchors for t in templates]
    flat = np.zeros((len(texts), 512), dtype=np.float32)
    batch = 64
    for i in range(0, len(texts), batch):
        flat[i:i + batch] = embedder.embed_text(texts[i:i + batch])
    all_vecs = flat.reshape(len(all_anchors), len(templates), 512).mean(axis=1)
    all_vecs /= np.linalg.norm(all_vecs, axis=1, keepdims=True) + 1e-9
    vec_by_id = {a["id"]: v for a, v in zip(all_anchors, all_vecs)}
    per_anchor = np.stack([vec_by_id[a["id"]] for a in anchors]) if anchors else all_vecs[:0]

    # negatives (for later gates; embedded and stored alongside)
    neg_texts = [t.format(a=n) for n in vocab["negatives"] for t in templates]
    neg_flat = embedder.embed_text(neg_texts)
    negatives = np.asarray(neg_flat).reshape(len(vocab["negatives"]), len(templates), 512).mean(axis=1)
    negatives /= np.linalg.norm(negatives, axis=1, keepdims=True) + 1e-9

    report = []
    for a, vec in zip(anchors, per_anchor):
        sims = pooled @ vec
        order = np.argsort(-sims)[: max(args.top, 5)]
        top = []
        for idx in order[: args.top]:
            r = rows[idx]
            probe_scores = {obs_rows[i]["probe"]: float(obs[i] @ vec) for i in r["obs_idx"]}
            top.append({
                "name": r["name"], "category": r["category"], "id": r["id"],
                "score": round(float(sims[idx]), 4),
                "best_probe": max(probe_scores, key=probe_scores.get),
            })
        best = top[0]["score"]
        probe_votes = [t["best_probe"] for t in top]
        dominant_probe = max(set(probe_votes), key=probe_votes.count)
        report.append({
            "id": a["id"], "text": a["text"], "role": a["role"],
            "best_score": best,
            "class": classify(best),
            "dominant_probe": dominant_probe,
            "top": top,
        })

    report.sort(key=lambda r: r["best_score"])
    out = {
        "vocab_version": vocab["version"],
        "templates": templates,
        "thresholds": {"covered": COVERED, "sparse": SPARSE},
        "anchor_vectors": str(DATA_DIR / "anchor_vectors.npz"),
        "anchors": report,
    }
    (DATA_DIR / "anchor_coverage.json").write_text(json.dumps(out, indent=2))
    np.savez(
        DATA_DIR / "anchor_vectors.npz",
        anchor_ids=np.array([a["id"] for a in all_anchors]),
        anchor_vecs=all_vecs.astype(np.float32),
        negative_texts=np.array(vocab["negatives"]),
        negative_vecs=negatives.astype(np.float32),
    )

    counts = {}
    for r in report:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    print(f"{'anchor':<26} {'class':<8} {'best':>6}  dominant-probe    top match")
    for r in report:
        print(f"{r['id']:<26} {r['class']:<8} {r['best_score']:>6.3f}  {r['dominant_probe']:<16} "
              f"{r['top'][0]['name'][:30]} ({r['top'][0]['category']})")
    print(f"\ncoverage: {counts} of {len(report)} anchors  → campaign targets = empty + sparse")
    print(f"saved: data/anchor_coverage.json + data/anchor_vectors.npz")


if __name__ == "__main__":
    main()
