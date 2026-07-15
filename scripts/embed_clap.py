"""Embed every rendered probe with LAION-CLAP music → local index files.

Outputs under data/index/:
  obs.npy        (N, 512) float32 L2-normed per-observation audio embeddings
  obs.jsonl      N rows {id, probe, flac}
  pooled.npy     (P, 512) per-patch mean-pooled (renormed)
  pooled.jsonl   P rows {id, name, category, source, probes: [...]}
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR

INDEX = DATA_DIR / os.environ.get("SPS_INDEX_DIR", "index")
BATCH = 12


def main() -> None:
    import soundfile as sf

    from sps.clap_embed import ClapEmbedder

    INDEX.mkdir(parents=True, exist_ok=True)
    renders = [json.loads(l) for l in (DATA_DIR / "renders.jsonl").read_text().splitlines()]
    ok = [r for r in renders if r.get("status") == "ok" and r.get("activity", 0) > 0.05]
    # de-dupe (id, probe) keeping last
    by_key = {(r["id"], r["probe"]): r for r in ok}
    obs = list(by_key.values())
    print(f"render rows: {len(renders)}; embeddable observations: {len(obs)}")

    embedder = ClapEmbedder()
    print(f"device: {embedder.device}")

    vecs = np.zeros((len(obs), 512), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(obs), BATCH):
        batch = obs[i:i + BATCH]
        waves = []
        for r in batch:
            wave, sr = sf.read(DATA_DIR / r["flac"], dtype="float32")
            assert sr == 48000, f"unexpected sr {sr}"
            waves.append(wave)
        vecs[i:i + len(batch)] = embedder.embed_audio(waves)
        if (i // BATCH) % 20 == 0:
            done = i + len(batch)
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(obs)}  ({rate:.1f} obs/s, eta {(len(obs)-done)/max(rate,0.1):.0f}s)", flush=True)

    np.save(INDEX / "obs.npy", vecs)
    with open(INDEX / "obs.jsonl", "w") as f:
        for r in obs:
            f.write(json.dumps({"id": r["id"], "probe": r["probe"], "flac": r["flac"]}) + "\n")

    # pool per patch
    corpus = {}
    for l in (DATA_DIR / "corpus.jsonl").read_text().splitlines():
        row = json.loads(l)
        corpus[row["id"]] = row
    groups = defaultdict(list)
    for idx, r in enumerate(obs):
        groups[r["id"]].append(idx)

    pooled_rows = []
    pooled = np.zeros((len(groups), 512), dtype=np.float32)
    for p, (pid, idxs) in enumerate(sorted(groups.items())):
        v = vecs[idxs].mean(axis=0)
        pooled[p] = v / (np.linalg.norm(v) + 1e-9)
        c = corpus.get(pid, {})
        pooled_rows.append({
            "id": pid,
            "name": c.get("surge_meta_name") or c.get("name", ""),
            "category": c.get("category", ""),
            "source": c.get("source", ""),
            "probes": [obs[i]["probe"] for i in idxs],
            "obs_idx": idxs,
        })
    np.save(INDEX / "pooled.npy", pooled)
    with open(INDEX / "pooled.jsonl", "w") as f:
        for r in pooled_rows:
            f.write(json.dumps(r) + "\n")
    print(f"indexed {len(pooled_rows)} patches / {len(obs)} observations in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
