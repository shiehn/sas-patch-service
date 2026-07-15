"""Ingest the Surge in-box patch library (factory + 3rd-party) into the spike corpus.

References the .fxp files inside third_party/surge/resources/data/ in place
(no copying) and appends rows to data/corpus.jsonl.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR, SURGE_SRC
from sps import fxp as fxp_mod
from sps import wrapper as wrapper_mod

CORPUS = DATA_DIR / "corpus.jsonl"
SETS = {
    "factory": SURGE_SRC / "resources" / "data" / "patches_factory",
    "third_party": SURGE_SRC / "resources" / "data" / "patches_3rdparty",
}


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    for source, root in SETS.items():
        if not root.is_dir():
            sys.exit(f"missing {root} — clone third_party/surge first")
        for path in sorted(root.rglob("*.fxp")):
            rel = path.relative_to(root)
            pid = f"{source}/{rel}"
            try:
                f = fxp_mod.read_file(path)
                stream = wrapper_mod.parse_patch_stream(f.chunk)
                meta = stream.meta
                # patches_3rdparty/<AuthorPack>/<Category>/name.fxp ; factory/<Category>/name.fxp
                parts = rel.parts
                category = parts[-2] if len(parts) >= 2 else ""
                pack = parts[0] if source == "third_party" and len(parts) >= 3 else ""
                rows.append({
                    "id": pid,
                    "source": source,
                    "fxp": str(path),
                    "name": path.stem,
                    "surge_meta_name": meta.get("name", ""),
                    "surge_meta_category": meta.get("category", ""),
                    "surge_meta_author": meta.get("author", ""),
                    "category": category,
                    "pack": pack,
                    "revision": stream.revision,
                    "has_wavetables": stream.has_wavetables,
                    "state_bytes": len(f.chunk),
                })
            except Exception as e:  # noqa: BLE001
                failures.append({"id": pid, "error": f"{type(e).__name__}: {e}"})

    with open(CORPUS, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    by_source = {}
    for r in rows:
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    wt = sum(1 for r in rows if r["has_wavetables"])
    revs = {}
    for r in rows:
        revs[r["revision"]] = revs.get(r["revision"], 0) + 1
    print(f"ingested {by_source} → {CORPUS}; failures: {len(failures)}")
    for fl in failures[:10]:
        print(f"  {fl['id']}: {fl['error']}")
    print(f"embedded wavetables: {wt}/{len(rows)}; revision histogram: {dict(sorted(revs.items(), key=lambda kv: (kv[0] is None, kv[0])))}")


if __name__ == "__main__":
    main()
