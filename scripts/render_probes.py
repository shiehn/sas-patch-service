"""Render the probe suite for every corpus patch → FLAC + render log.

Usage:
  .venv/bin/python scripts/render_probes.py [--limit N] [--workers W] [--sources bundled,factory,third_party]

Outputs:
  data/renders/<patch-id>/<probe-id>.flac   (mono 48 kHz)
  data/renders.jsonl                        one row per (patch, probe) with stats
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR

RENDERS = DATA_DIR / "renders"
LOG = DATA_DIR / "renders.jsonl"


def _safe_id(pid: str) -> str:
    return pid.replace("/", "__").replace(" ", "_")


def render_one(row: Dict) -> List[Dict]:
    """Worker: render all probes for one patch. Imports stay inside the worker."""
    import numpy as np  # noqa: F401
    import soundfile as sf

    from sps.probes import PROBES, SAMPLE_RATE
    from sps.render import probe_stats, render_probe, to_mono_16bit_ok

    out_rows: List[Dict] = []
    fxp_path = row["fxp"] if row["fxp"].startswith("/") else str(DATA_DIR / row["fxp"])
    out_dir = RENDERS / _safe_id(row["id"])
    out_dir.mkdir(parents=True, exist_ok=True)
    for probe in PROBES:
        t0 = time.time()
        rec: Dict = {"id": row["id"], "probe": probe.id}
        try:
            audio = render_probe(fxp_path, probe)
            stats = probe_stats(audio)
            rec.update(stats)
            mono = to_mono_16bit_ok(audio)
            if mono is None:
                rec["status"] = "nonfinite"
            elif stats["peak"] < 1e-5:
                rec["status"] = "silent"
            else:
                flac = out_dir / f"{probe.id}.flac"
                sf.write(flac, mono, SAMPLE_RATE, format="FLAC", subtype="PCM_16")
                rec["status"] = "ok"
                rec["flac"] = str(flac.relative_to(DATA_DIR))
        except Exception as e:  # noqa: BLE001 — survey render, never kill the run
            rec["status"] = "error"
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["trace"] = traceback.format_exc(limit=2)
        rec["render_sec"] = round(time.time() - t0, 3)
        out_rows.append(rec)
    return out_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sources", default="bundled,factory,third_party")
    args = ap.parse_args()

    sources = set(args.sources.split(","))
    rows = [json.loads(l) for l in (DATA_DIR / "corpus.jsonl").read_text().splitlines()]
    rows = [r for r in rows if r["source"] in sources]
    # de-dupe corpus rows by id (re-runs of ingest append)
    seen = set()
    rows = [r for r in rows if not (r["id"] in seen or seen.add(r["id"]))]
    if args.limit:
        rows = rows[: args.limit]

    done_ids = set()
    if LOG.exists():
        for l in LOG.read_text().splitlines():
            try:
                done_ids.add(json.loads(l)["id"])
            except Exception:  # noqa: BLE001
                pass
    todo = [r for r in rows if r["id"] not in done_ids]
    print(f"corpus rows: {len(rows)}; already rendered: {len(rows) - len(todo)}; to render: {len(todo)}")

    t0 = time.time()
    n_ok = n_bad = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(LOG, "a") as log:
        for i, recs in enumerate(ex.map(render_one, todo, chunksize=4)):
            for rec in recs:
                log.write(json.dumps(rec) + "\n")
                if rec["status"] == "ok":
                    n_ok += 1
                else:
                    n_bad += 1
            if (i + 1) % 100 == 0:
                rate = (i + 1) / (time.time() - t0)
                print(f"  {i+1}/{len(todo)} patches  ({rate:.1f} patches/s, ok={n_ok} bad={n_bad})", flush=True)
            log.flush()
    print(f"done: {n_ok} probe renders ok, {n_bad} bad, {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
