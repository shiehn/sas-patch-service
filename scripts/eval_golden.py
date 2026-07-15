"""Run the golden query set against the index → category-precision@k + report.

Only queries with `expect_category` count toward precision (objective subset);
all queries get their top-k dumped to data/eval_report.json for listening review.

Usage: .venv/bin/python scripts/eval_golden.py [-k 5]
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


def category_matches(expected: str, category: str, name: str) -> bool:
    hay = f"{category} {name}".lower()
    return expected.lower() in hay


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--template", default="{q}",
                    help="text template, e.g. 'a recording of {q}' or '{q}, a {role} synthesizer sound'")
    args = ap.parse_args()

    from sps.clap_embed import ClapEmbedder

    golden = json.loads((REPO_ROOT / "eval" / "golden_queries.json").read_text())
    pooled = np.load(INDEX / "pooled.npy")
    rows = [json.loads(l) for l in (INDEX / "pooled.jsonl").read_text().splitlines()]
    obs = np.load(INDEX / "obs.npy")
    obs_rows = [json.loads(l) for l in (INDEX / "obs.jsonl").read_text().splitlines()]

    embedder = ClapEmbedder(device="cpu")
    queries = golden["queries"]
    texts = [args.template.format(q=q["text"], role=q.get("role", "synth")) for q in queries]
    q_vecs = embedder.embed_text(texts)

    report = []
    precisions = []
    for q, qv in zip(queries, q_vecs):
        sims = pooled @ qv
        cand = np.argsort(-sims)[: args.k * 5]
        scored = []
        for p in cand:
            r = rows[p]
            obs_scores = {obs_rows[i]["probe"]: float(obs[i] @ qv) for i in r["obs_idx"]}
            rerank = max(obs_scores.values())
            scored.append((0.5 * float(sims[p]) + 0.5 * rerank, r))
        scored.sort(key=lambda t: -t[0])
        top = scored[: args.k]

        hits = None
        if q.get("expect_category"):
            hits = sum(1 for _, r in top if category_matches(q["expect_category"], r["category"], r["name"]))
            precisions.append(hits / args.k)
        report.append({
            "id": q["id"], "text": q["text"], "expect": q.get("expect_category"),
            "hits_at_k": hits,
            "top": [{"score": round(s, 3), "name": r["name"], "category": r["category"],
                     "source": r["source"], "id": r["id"]} for s, r in top],
        })
        tag = f"{hits}/{args.k}" if hits is not None else "  — "
        print(f"  [{tag}] {q['id']:<12} {q['text'][:58]:<58} → {report[-1]['top'][0]['name'][:30]} ({report[-1]['top'][0]['category']})")

    out = DATA_DIR / "eval_report.json"
    out.write_text(json.dumps(report, indent=2))
    objective = [q for q in queries if q.get("expect_category")]
    print(f"\ncategory-precision@{args.k}: {np.mean(precisions):.3f} over {len(objective)} objective queries")
    print(f"full report: {out}")


if __name__ == "__main__":
    main()
