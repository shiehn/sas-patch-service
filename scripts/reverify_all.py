"""Re-verify every campaign under the CURRENT gate stack (incl. quality judges).

Used after gates change (e.g. aesthetics floors added) to refresh every
verification.json — the input to absorption/re-curation decisions.

Usage: .venv/bin/python scripts/reverify_all.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sps import DATA_DIR
from verify_survivors import verify  # noqa: E402


def main() -> None:
    from sps.clap_embed import ClapEmbedder

    embedder = ClapEmbedder()
    campaigns = sorted(p.parent.name for p in (DATA_DIR / "campaigns").glob("*/campaign.json"))
    t0 = time.time()
    rows = []
    for i, anchor_id in enumerate(campaigns, 1):
        try:
            results = verify(anchor_id, embedder, quiet=True)
            passing = sum(1 for r in results if r["verdict"] == "PASS")
            fails = {}
            for r in results:
                if r["verdict"] != "PASS":
                    for g in r["verdict"].removeprefix("fail:").split(","):
                        fails[g] = fails.get(g, 0) + 1
            rows.append({"anchor_id": anchor_id, "survivors": len(results),
                         "pass": passing, "fail_gates": fails})
            print(f"[{i}/{len(campaigns)}] {anchor_id}: {passing}/{len(results)} PASS"
                  f"{'  ' + str(fails) if fails else ''}  [{time.time()-t0:.0f}s]", flush=True)
        except Exception as e:  # noqa: BLE001
            rows.append({"anchor_id": anchor_id, "error": str(e)})
            print(f"[{i}/{len(campaigns)}] {anchor_id}: ERROR {e}", flush=True)

    (DATA_DIR / "reverify_summary.json").write_text(json.dumps(rows, indent=2))
    ok = [r for r in rows if "error" not in r]
    total = sum(r["survivors"] for r in ok)
    passing = sum(r["pass"] for r in ok)
    gate_fail_totals: dict = {}
    for r in ok:
        for g, n in r.get("fail_gates", {}).items():
            gate_fail_totals[g] = gate_fail_totals.get(g, 0) + n
    print(f"\nRE-VERIFY DONE in {(time.time()-t0)/60:.1f} min: {passing}/{total} survivors PASS "
          f"under current gates; failures by gate: {gate_fail_totals}")
    print(f"summary: {DATA_DIR / 'reverify_summary.json'}")


if __name__ == "__main__":
    main()
