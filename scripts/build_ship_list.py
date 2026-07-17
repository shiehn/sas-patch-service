"""Build data/curated_ship_list.json from the blind-A/B vote record.

Ship rule: a generated patch ships iff its anchor WON or TIED a gate2 blind A/B
(last vote per anchor wins) and the individual patch wasn't taste-rejected.
Explicit ear-approvals can be pinned via EXTRA_APPROVED_ANCHORS.

Usage: .venv/bin/python scripts/build_ship_list.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR

# anchors approved outside the A/B flow (direct listening verdicts)
EXTRA_APPROVED_ANCHORS = {"brass-trumpet-muted"}


def main() -> None:
    latest: dict = {}
    votes_path = DATA_DIR / "ab_votes.jsonl"
    if votes_path.exists():
        for line in votes_path.read_text().splitlines():
            row = json.loads(line)
            if row.get("mode") == "gate2":
                latest[row["query_id"]] = row["winner"]  # last vote wins

    ship_anchors = {a for a, v in latest.items() if v in ("semantic", "tie")}
    ship_anchors |= EXTRA_APPROVED_ANCHORS

    taste_rejects = set()
    taste_path = DATA_DIR / "taste_votes.jsonl"
    if taste_path.exists():
        for line in taste_path.read_text().splitlines():
            row = json.loads(line)
            if not row["vote"]:
                taste_rejects.add(row["id"])

    ship, dropped = [], 0
    for line in (DATA_DIR / "corpus.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("source") != "generated":
            continue
        if row.get("campaign", {}).get("anchor_id") in ship_anchors:
            if row["name"] in taste_rejects:
                dropped += 1
                continue
            ship.append(row["id"])

    out = {
        "ship_anchors": sorted(ship_anchors),
        "generated_patch_ids": sorted(ship),
        "taste_rejects_excluded": dropped,
    }
    (DATA_DIR / "curated_ship_list.json").write_text(json.dumps(out, indent=2))
    print(f"ship list: {len(ship)} generated patches across {len(ship_anchors)} anchors "
          f"({dropped} taste-rejects excluded) → data/curated_ship_list.json")


if __name__ == "__main__":
    main()
