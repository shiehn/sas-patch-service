"""Calibrate quality gates against the FACTORY corpus distribution.

Samples factory/third_party renders, scores AudioBox axes + CLAP quality-contrast,
and writes percentile tables to data/aesthetics_baseline.json. Gates then say
"at or above the factory 25th percentile", never absolute numbers.

Usage: .venv/bin/python scripts/aesthetics_baseline.py [--sample 240]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=240)
    args = ap.parse_args()

    import soundfile as sf

    from sps.aesthetics import audiobox_scores, clap_quality_contrast
    from sps.clap_embed import ClapEmbedder

    renders = [json.loads(l) for l in (DATA_DIR / "renders.jsonl").read_text().splitlines()]
    human = [r for r in renders
             if r.get("status") == "ok" and r["id"].split("/")[0] in ("factory", "third_party")
             and r.get("activity", 0) > 0.05]
    rng = random.Random(42)
    sample = rng.sample(human, min(args.sample, len(human)))
    paths = [str(DATA_DIR / r["flac"]) for r in sample]

    t0 = time.time()
    print(f"scoring {len(paths)} factory/third-party renders with AudioBox…")
    ab = audiobox_scores(paths)

    print("computing CLAP quality-contrast…")
    embedder = ClapEmbedder()
    contrasts = []
    for i in range(0, len(paths), 12):
        waves = [sf.read(p, dtype="float32")[0] for p in paths[i:i + 12]]
        embs = embedder.embed_audio(waves)
        contrasts.extend(clap_quality_contrast(embedder, embs).tolist())

    pcts = [5, 10, 25, 50, 75, 90]
    baseline = {"sample_size": len(paths), "percentiles": {}}
    for axis in ("PQ", "CE", "CU", "PC"):
        vals = np.array([row[axis] for row in ab])
        baseline["percentiles"][axis] = {f"p{p}": round(float(np.percentile(vals, p)), 3) for p in pcts}
    cvals = np.array(contrasts)
    baseline["percentiles"]["clap_contrast"] = {f"p{p}": round(float(np.percentile(cvals, p)), 4) for p in pcts}

    (DATA_DIR / "aesthetics_baseline.json").write_text(json.dumps(baseline, indent=2))
    print(json.dumps(baseline["percentiles"], indent=2))
    print(f"done in {time.time()-t0:.0f}s → data/aesthetics_baseline.json")


if __name__ == "__main__":
    main()
