"""Ingest the S&S bundled Surge presets into the spike corpus.

Reads sas-app/resources/tracktion-presets/SurgeXT/<Category>.json files
({"Preset_N": "<std b64 of <PLUGIN> XML>"}), decodes each preset's raw Surge
patch stream, wraps it into a standard .fxp (so surgepy can load it), and
appends rows to data/corpus.jsonl.

Run: .venv/bin/python scripts/ingest_bundled.py   (stdlib-only; any py>=3.9)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR, SAS_APP
from sps import fxp as fxp_mod
from sps import wrapper as wrapper_mod

PRESET_DIR = SAS_APP / "resources" / "tracktion-presets" / "SurgeXT"
OUT_FXP = DATA_DIR / "fxp" / "bundled"
CORPUS = DATA_DIR / "corpus.jsonl"


def main() -> None:
    if not PRESET_DIR.is_dir():
        sys.exit(f"preset dir not found: {PRESET_DIR}")
    OUT_FXP.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    for json_path in sorted(PRESET_DIR.glob("*.json")):
        if json_path.name == "manifest.json":
            continue
        category = json_path.stem  # e.g. "Basses-Low"
        presets = json.loads(json_path.read_text())
        for preset_name, b64_value in presets.items():
            pid = f"bundled/{category}/{preset_name}"
            try:
                w = wrapper_mod.decode_wrapper_b64(b64_value)
                stream = wrapper_mod.parse_patch_stream(w.chunk)
                meta = stream.meta
                out = OUT_FXP / category / f"{preset_name}.fxp"
                out.parent.mkdir(parents=True, exist_ok=True)
                fxp_mod.write_file(out, fxp_mod.Fxp(chunk=w.chunk, name=meta.get("name", preset_name)))
                rows.append({
                    "id": pid,
                    "source": "bundled",
                    "fxp": str(out.relative_to(DATA_DIR)),
                    "name": preset_name,
                    "surge_meta_name": meta.get("name", ""),
                    "surge_meta_category": meta.get("category", ""),
                    "surge_meta_author": meta.get("author", ""),
                    "category": category,
                    "revision": stream.revision,
                    "has_wavetables": stream.has_wavetables,
                    "state_bytes": len(w.chunk),
                })
            except Exception as e:  # noqa: BLE001 — survey run, collect all failures
                failures.append({"id": pid, "error": f"{type(e).__name__}: {e}"})

    with open(CORPUS, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"ingested {len(rows)} bundled presets → {CORPUS}")
    print(f"failures: {len(failures)}")
    for fl in failures[:10]:
        print(f"  {fl['id']}: {fl['error']}")

    named = sum(1 for r in rows if r["surge_meta_name"] and not r["surge_meta_name"].startswith("Preset_"))
    wt = sum(1 for r in rows if r["has_wavetables"])
    revs = sorted({r["revision"] for r in rows})
    print(f"surge-meta real names: {named}/{len(rows)}; embedded wavetables: {wt}; revisions: {revs}")
    for r in rows[:5]:
        print(f"  sample: {r['id']}  meta_name={r['surge_meta_name']!r} "
              f"meta_cat={r['surge_meta_category']!r} rev={r['revision']} wt={r['has_wavetables']}")


if __name__ == "__main__":
    main()
