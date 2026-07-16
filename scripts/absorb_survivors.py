"""Absorb gate-passing campaign survivors into the corpus.

- scans data/campaigns/*/verification.json for verdict == PASS
- intra-campaign dedup: greedy by fitness, drops survivors whose best-probe
  embedding is ≥ 0.97 cosine to an already-kept sibling (the novelty gate only
  compared against the EXISTING corpus, not against each other)
- copies fxp → data/fxp/generated/<anchor>/, verification renders → data/renders/,
  appends corpus.jsonl + renders.jsonl rows (source='generated', display name from
  the anchor text — the caption channel can improve names later)

Then rebuild the index + pack:
  scripts/embed_clap.py && scripts/export_client_pack.py && scripts/build_index_pack.py --version 2

Usage: .venv/bin/python scripts/absorb_survivors.py [--per-anchor 3] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR


def safe_id(pid: str) -> str:
    return pid.replace("/", "__").replace(" ", "_")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-anchor", type=int, default=3)
    ap.add_argument("--dirs", default="", help="comma list of campaign dir names (default: all)")
    ap.add_argument("--dedup-cos", type=float, default=0.97)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import soundfile as sf

    from sps.clap_embed import ClapEmbedder
    from sps.render import probe_stats

    coverage = json.loads((DATA_DIR / "anchor_coverage.json").read_text())
    anchor_meta = {a["id"]: a for a in coverage["anchors"]}

    existing_ids = set()
    for l in (DATA_DIR / "corpus.jsonl").read_text().splitlines():
        existing_ids.add(json.loads(l)["id"])

    embedder = ClapEmbedder()
    absorbed, skipped_dup, skipped_have = [], 0, 0

    only_dirs = set(args.dirs.split(",")) if args.dirs else None
    for ver_path in sorted((DATA_DIR / "campaigns").glob("*/verification.json")):
        if only_dirs is not None and ver_path.parent.name not in only_dirs:
            continue
        dir_name = ver_path.parent.name
        results = [r for r in json.loads(ver_path.read_text()) if r["verdict"] == "PASS"]
        results.sort(key=lambda r: r.get("fitness", -1), reverse=True)
        anchor_id = results[0].get("anchor_id", dir_name) if results else dir_name
        meta = anchor_meta.get(anchor_id, {})

        kept_embs: list = []
        kept_count = 0
        for r in results:
            if kept_count >= args.per_anchor:
                break
            cid = f"generated/{anchor_id}/{r['id']}"
            if cid in existing_ids:
                skipped_have += 1
                continue
            wave, sr = sf.read(r["listen"], dtype="float32")
            emb = embedder.embed_audio([wave])[0]
            if any(float(emb @ k) >= args.dedup_cos for k in kept_embs):
                skipped_dup += 1
                continue
            kept_embs.append(emb)
            kept_count += 1

            variant = r["id"].rsplit("-", 1)[-1]
            display = f"{meta.get('text', anchor_id).title()} · {variant}"
            role = meta.get("role", "")

            if not args.dry_run:
                fxp_dst = DATA_DIR / "fxp" / "generated" / anchor_id / f"{r['id']}.fxp"
                fxp_dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(r["fxp"], fxp_dst)

                render_src = Path(r["listen"]).parent
                render_dst = DATA_DIR / "renders" / safe_id(cid)
                render_dst.mkdir(parents=True, exist_ok=True)
                render_rows = []
                for flac in sorted(render_src.glob("*.flac")):
                    shutil.copy2(flac, render_dst / flac.name)
                    mono, _ = sf.read(render_dst / flac.name, dtype="float32")
                    stats = probe_stats(np.stack([mono, mono]))
                    render_rows.append({
                        "id": cid, "probe": flac.stem, **stats, "status": "ok",
                        "flac": str((render_dst / flac.name).relative_to(DATA_DIR)),
                    })
                with open(DATA_DIR / "renders.jsonl", "a") as f:
                    for row in render_rows:
                        f.write(json.dumps(row) + "\n")
                with open(DATA_DIR / "corpus.jsonl", "a") as f:
                    f.write(json.dumps({
                        "id": cid,
                        "source": "generated",
                        "fxp": str(fxp_dst.relative_to(DATA_DIR)),
                        "name": r["id"],
                        "surge_meta_name": display,
                        "surge_meta_category": f"Generated/{role}",
                        "surge_meta_author": "S&S corpus factory",
                        "category": f"Gen-{role}" if role else "Generated",
                        "revision": 24,
                        "has_wavetables": False,
                        "state_bytes": Path(fxp_dst).stat().st_size,
                        "campaign": {
                            "anchor_id": anchor_id,
                            "anchor_text": meta.get("text", ""),
                            "fitness": r.get("fitness"),
                            "anchor_best": r.get("anchor_best"),
                            "parent": r.get("parent"),
                        },
                    }) + "\n")
            absorbed.append({"id": cid, "display": display, "fitness": r.get("fitness")})

    print(f"absorbed {len(absorbed)} generated patches "
          f"({skipped_dup} intra-campaign dups dropped, {skipped_have} already absorbed)"
          f"{' [DRY RUN]' if args.dry_run else ''}")
    for a in absorbed[:10]:
        print(f"  {a['display']:<52} fitness={a['fitness']}")
    if len(absorbed) > 10:
        print(f"  … and {len(absorbed) - 10} more")
    if not args.dry_run and absorbed:
        print("\nnext: .venv/bin/python scripts/embed_clap.py && "
              ".venv/bin/python scripts/export_client_pack.py && "
              ".venv/bin/python scripts/build_index_pack.py --version 2")


if __name__ == "__main__":
    main()
