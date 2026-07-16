"""Taste-vote collector — turns Steve's ears into calibrated thresholds.

Plays campaign survivors (best-probe render), collects keep/reject votes, and
stores them WITH every gate metric so thresholds can be fit to actual taste.

Vote session:   .venv/bin/python scripts/listen_survivors.py [--anchor ID] [--limit 30]
    keys:  y = keep   n = reject   r = replay   s = skip   q = quit
Calibration:    .venv/bin/python scripts/listen_survivors.py --calibrate

Votes append to data/taste_votes.jsonl (re-voting a patch overrides on calibrate).
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR

VOTES = DATA_DIR / "taste_votes.jsonl"
METRICS = ("fitness", "anchor_best", "clarity", "neg_delta")


def collect(args) -> None:
    entries = []
    for ver in sorted((DATA_DIR / "campaigns").glob("*/verification.json")):
        if args.anchor and ver.parent.name != args.anchor:
            continue
        entries.extend(json.loads(ver.read_text()))
    voted = set()
    if VOTES.exists():
        voted = {json.loads(l)["id"] for l in VOTES.read_text().splitlines()}
    pool = [e for e in entries if e["id"] not in voted and Path(e.get("listen", "")).is_file()]
    random.Random(args.seed).shuffle(pool)
    pool = pool[: args.limit]
    if not pool:
        print("nothing unvoted to play (try --anchor or delete data/taste_votes.jsonl)")
        return

    print(f"{len(pool)} clips — y keep / n reject / r replay / s skip / q quit\n")
    with open(VOTES, "a") as out:
        for i, e in enumerate(pool, 1):
            camp = json.loads((DATA_DIR / "campaigns" / e["anchor_id"] / "campaign.json").read_text())
            print(f"[{i}/{len(pool)}] \"{camp['anchor_text']}\"  ({e['id']})")
            while True:
                subprocess.run(["afplay", e["listen"]], check=False)
                ans = input("  keep? [y/n/r/s/q] ").strip().lower()
                if ans == "r":
                    continue
                break
            if ans == "q":
                break
            if ans in ("y", "n"):
                out.write(json.dumps({
                    "id": e["id"], "anchor_id": e["anchor_id"], "vote": ans == "y",
                    **{m: e.get(m) for m in METRICS},
                }) + "\n")
                out.flush()
    print(f"\nvotes stored in {VOTES}")


def calibrate() -> None:
    if not VOTES.exists():
        sys.exit("no votes yet — run a listening session first")
    rows = {}
    for l in VOTES.read_text().splitlines():
        r = json.loads(l)
        rows[r["id"]] = r  # last vote wins
    votes = list(rows.values())
    keeps = [v for v in votes if v["vote"]]
    rejects = [v for v in votes if not v["vote"]]
    print(f"{len(votes)} voted patches: {len(keeps)} keep / {len(rejects)} reject\n")
    if not keeps or not rejects:
        print("need both keeps and rejects to calibrate thresholds")
        return

    print(f"{'metric':<14} {'keep-mean':>10} {'reject-mean':>12} {'suggested-floor':>16} {'accuracy':>9}")
    for m in METRICS:
        kv = np.array([v[m] for v in keeps if v.get(m) is not None])
        rv = np.array([v[m] for v in rejects if v.get(m) is not None])
        if kv.size == 0 or rv.size == 0:
            continue
        candidates = np.unique(np.concatenate([kv, rv]))
        best_t, best_acc = None, -1.0
        for t in candidates:
            acc = (np.mean(kv >= t) * len(kv) + np.mean(rv < t) * len(rv)) / (len(kv) + len(rv))
            if acc > best_acc:
                best_t, best_acc = float(t), float(acc)
        print(f"{m:<14} {kv.mean():>10.3f} {rv.mean():>12.3f} {best_t:>16.3f} {best_acc:>8.0%}")
    print("\napply floors in verify_survivors gates / absorb_survivors selection")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", default="")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()
    if args.calibrate:
        calibrate()
    else:
        collect(args)


if __name__ == "__main__":
    main()
