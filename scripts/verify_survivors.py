"""Verify a campaign's survivors: full probe renders + v1 gate metrics + listen list.

For each survivor fxp:
  - render ALL probes → FLAC under data/campaigns/<anchor>/renders/<id>/
  - objective stats (peak/rms/activity/nonfinite)
  - anchor-similarity across probes (max + which probe)
  - clarity margin: best-anchor − mean over the whole anchor vocabulary
  - negative contrast: anchor cos − max negative cos
  - novelty vs existing index: max pooled-cosine (near-1.0 = duplicate of corpus)

Usage: .venv/bin/python scripts/verify_survivors.py brass-trumpet-muted [--play]
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
    ap.add_argument("anchor_id")
    ap.add_argument("--play", action="store_true", help="afplay the #1 survivor's best probe")
    args = ap.parse_args()

    import soundfile as sf

    from sps.clap_embed import ClapEmbedder
    from sps.probes import PROBES, SAMPLE_RATE
    from sps.render import probe_stats, render_probe, to_mono_16bit_ok

    camp_dir = DATA_DIR / "campaigns" / args.anchor_id
    manifest = json.loads((camp_dir / "campaign.json").read_text())

    vectors = np.load(DATA_DIR / "anchor_vectors.npz", allow_pickle=False)
    anchor_ids = [str(x) for x in vectors["anchor_ids"]]
    anchor_vecs = vectors["anchor_vecs"]
    anchor_vec = anchor_vecs[anchor_ids.index(args.anchor_id)]
    negative_vecs = vectors["negative_vecs"]
    pooled = np.load(INDEX / "pooled.npy")

    embedder = ClapEmbedder()
    print(f'verifying {len(manifest["survivors"])} survivors of "{manifest["anchor_text"]}"\n')
    print(f"{'#':>2} {'id':<34} {'anchor':>7} {'probe':<15} {'clarity':>8} {'negΔ':>6} {'novelty':>8} verdict")

    results = []
    for s in manifest["survivors"]:
        rid = s["id"]
        render_dir = camp_dir / "renders" / rid
        render_dir.mkdir(parents=True, exist_ok=True)
        waves, probe_ids, all_stats = [], [], {}
        for probe in PROBES:
            audio = render_probe(s["fxp"], probe)
            st = probe_stats(audio)
            all_stats[probe.id] = st
            mono = to_mono_16bit_ok(audio)
            if mono is not None and st["peak"] >= 1e-5:
                sf.write(render_dir / f"{probe.id}.flac", mono, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
                waves.append(mono)
                probe_ids.append(probe.id)
        if not waves:
            print(f"   {rid:<34} ALL PROBES SILENT — reject")
            continue
        embs = embedder.embed_audio(waves)
        anchor_sims = embs @ anchor_vec
        best_i = int(np.argmax(anchor_sims))
        best_emb = embs[best_i]
        pooled_mean = embs.mean(axis=0)
        pooled_mean /= np.linalg.norm(pooled_mean) + 1e-9

        vocab_sims = anchor_vecs @ best_emb
        clarity = float(anchor_sims[best_i] - vocab_sims.mean())
        neg_delta = float(anchor_sims[best_i] - np.max(negative_vecs @ best_emb))
        novelty = float(np.max(pooled @ pooled_mean))  # vs existing corpus

        gates = {
            "objective": all(v.get("nonfinite_frac", 0) == 0 for v in all_stats.values()),
            "clarity": clarity > 0.05,
            "negative": neg_delta > 0.05,
            "novel": novelty < 0.985,
        }
        verdict = "PASS" if all(gates.values()) else "fail:" + ",".join(k for k, v in gates.items() if not v)
        results.append({**s, "anchor_best": round(float(anchor_sims[best_i]), 4),
                        "best_probe": probe_ids[best_i], "clarity": round(clarity, 4),
                        "neg_delta": round(neg_delta, 4), "novelty_max_cos": round(novelty, 4),
                        "gates": gates, "verdict": verdict,
                        "listen": str(render_dir / f"{probe_ids[best_i]}.flac")})
        print(f"{s['rank']:>2} {rid:<34} {anchor_sims[best_i]:>7.3f} {probe_ids[best_i]:<15} "
              f"{clarity:>8.3f} {neg_delta:>6.3f} {novelty:>8.3f} {verdict}")

    (camp_dir / "verification.json").write_text(json.dumps(results, indent=2))
    passing = [r for r in results if r["verdict"] == "PASS"]
    print(f"\n{len(passing)}/{len(results)} survivors pass v1 gates")
    if results:
        print(f"listen:  afplay '{results[0]['listen']}'")
    if args.play and results:
        subprocess.run(["afplay", results[0]["listen"]], check=False)


if __name__ == "__main__":
    main()
