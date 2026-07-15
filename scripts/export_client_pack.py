"""Export the index as a client-consumable pack (TS-friendly, no numpy needed).

The Signals & Sorcery client intersects this pack against the .fxp files actually
present on the user's machine (Surge XT's installed library) by CONTENT HASH, then
ranks locally with plain Float32Array math.

Output (data/client-pack/):
  manifest.json     schema/model/dims/counts + probe order
  patches.jsonl     one row per patch: idx, fxp sha256, name, category, source,
                    per-probe observation row indices
  pooled.f32        float32 LE, N × dims (row i ↔ patches.jsonl line i)
  obs.f32           float32 LE, M × dims (per-observation vectors for probe rerank)

Usage: .venv/bin/python scripts/export_client_pack.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR
from sps.clap_embed import MODEL_ID

INDEX = DATA_DIR / os.environ.get("SPS_INDEX_DIR", "index")
OUT = DATA_DIR / "client-pack"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pooled = np.load(INDEX / "pooled.npy").astype("<f4")
    rows = [json.loads(l) for l in (INDEX / "pooled.jsonl").read_text().splitlines()]
    obs = np.load(INDEX / "obs.npy").astype("<f4")
    obs_rows = [json.loads(l) for l in (INDEX / "obs.jsonl").read_text().splitlines()]

    corpus = {}
    for l in (DATA_DIR / "corpus.jsonl").read_text().splitlines():
        r = json.loads(l)
        corpus[r["id"]] = r

    t0 = time.time()
    out_rows = []
    skipped = 0
    for i, r in enumerate(rows):
        c = corpus.get(r["id"])
        fxp = c["fxp"] if c["fxp"].startswith("/") else str(DATA_DIR / c["fxp"])
        try:
            digest = hashlib.sha256(Path(fxp).read_bytes()).hexdigest()
        except OSError:
            skipped += 1
            digest = None
        out_rows.append({
            "idx": i,
            "sha256": digest,
            "id": r["id"],
            "name": r["name"],
            "category": r["category"],
            "source": r["source"],
            "author": (c or {}).get("surge_meta_author", ""),
            "obs": {obs_rows[j]["probe"]: j for j in r["obs_idx"]},
        })

    pooled.tofile(OUT / "pooled.f32")
    obs.tofile(OUT / "obs.f32")
    with open(OUT / "patches.jsonl", "w") as f:
        for r in out_rows:
            f.write(json.dumps(r) + "\n")
    manifest = {
        "schema": "sas.patch-index-pack/v1",
        "model": MODEL_ID,
        "dims": int(pooled.shape[1]),
        "patchCount": int(pooled.shape[0]),
        "obsCount": int(obs.shape[0]),
        "probes": sorted({o["probe"] for o in obs_rows}),
        "vectorDtype": "float32-le",
        "hashAlgo": "sha256(fxp file bytes)",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))

    size = sum((OUT / n).stat().st_size for n in os.listdir(OUT))
    print(json.dumps(manifest, indent=2))
    print(f"hashed {len(out_rows)-skipped}/{len(out_rows)} fxp files "
          f"(skipped {skipped}); pack size {size/1e6:.1f} MB; {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
