"""Build the distributable patch-index pack zip for PackDownloadService.

Zips data/client-pack/ (manifest.json, patches.jsonl, pooled.f32, obs.f32) plus a
root _pack-version.json marker, deterministically (sorted entries, fixed mtime,
no extra attrs) so the sha256 is stable for a given input set.

Usage:
  .venv/bin/python scripts/build_index_pack.py --version 1
  gsutil cp dist/sas-patch-index-pack-v1.zip gs://docs-assets/
  # then update PATCH_INDEX_PACK in sas-app/src/shared/constants/sample-packs.ts
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR, REPO_ROOT

PACK_ID = "sas-patch-index-pack"
SRC = DATA_DIR / "client-pack"
DIST = REPO_ROOT / "dist"
FIXED_DATE = (2026, 1, 1, 0, 0, 0)  # deterministic zip entry timestamps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    args = ap.parse_args()

    files = sorted(p for p in SRC.iterdir() if p.is_file() and p.name != "fxp-hash-cache.json")
    if not any(p.name == "manifest.json" for p in files):
        sys.exit(f"no client pack at {SRC} — run scripts/export_client_pack.py first")

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        commit = "unknown"

    marker = {
        "packId": PACK_ID,
        "version": args.version,
        "schemaVersion": 1,
        "sourceCommit": commit,
        "sizeBytesUncompressed": sum(p.stat().st_size for p in files),
        "fileCount": len(files) + 1,
    }

    DIST.mkdir(exist_ok=True)
    out = DIST / f"{PACK_ID}-v{args.version}.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        info = zipfile.ZipInfo("_pack-version.json", date_time=FIXED_DATE)
        info.external_attr = 0o644 << 16
        zf.writestr(info, json.dumps(marker, indent=2, sort_keys=True))
        for p in files:
            info = zipfile.ZipInfo(p.name, date_time=FIXED_DATE)
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, p.read_bytes())

    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(json.dumps({
        "zip": str(out),
        "sizeBytes": out.stat().st_size,
        "sha256": digest,
        "marker": marker,
    }, indent=2))
    print(f"\nupload:  gsutil cp {out} gs://docs-assets/")
    print("then set sizeBytes + sha256 in PATCH_INDEX_PACK (sas-app sample-packs.ts)")


if __name__ == "__main__":
    main()
