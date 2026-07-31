#!/usr/bin/env python3
"""Apply XML-level motion traits (LFO temposync) to .fxp patches.

Seed prep for wobble campaigns — run this on the seed set BEFORE evolving so
fitness renders hear the synced LFO (see sps/motion_traits.py docstring).

Usage:
  python scripts/set_motion_traits.py --out-dir data/seeds-synced \
      --lfo 1 patches/seed-a.fxp patches/seed-b.fxp
  python scripts/set_motion_traits.py --in-place --lfo 1 --lfo 2 child.fxp
  python scripts/set_motion_traits.py --disable --in-place --lfo 1 child.fxp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps.motion_traits import apply_to_fxp_file  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fxp", nargs="+", help=".fxp files to process")
    ap.add_argument("--lfo", type=int, action="append", default=None,
                    help="voice LFO number 1..6 (repeatable; default: 1)")
    ap.add_argument("--scene", choices=("a", "b"), default="a")
    ap.add_argument("--disable", action="store_true", help="clear temposync instead of setting it")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--out-dir", help="write <out-dir>/<name>.fxp copies")
    group.add_argument("--in-place", action="store_true")
    args = ap.parse_args()

    lfos = args.lfo or [1]
    out_dir = Path(args.out_dir) if args.out_dir else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    for src in args.fxp:
        src_path = Path(src)
        dst_path = src_path if args.in_place else (out_dir / src_path.name)  # type: ignore[operator]
        apply_to_fxp_file(str(src_path), str(dst_path), lfos=lfos,
                          enabled=not args.disable, scene=args.scene)
        state = "cleared" if args.disable else "set"
        print(f"[motion-traits] temposync {state} on LFO {lfos} ({args.scene}): {dst_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
