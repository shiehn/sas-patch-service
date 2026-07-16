"""Caption channel: Gemini listens to every patch and writes producer-language
descriptions + a display name. The captions become (a) meaningful names in the
app, (b) a second retrieval channel (caption-text embeddings) that understands
compound producer-speak better than CLAP's 77-token contrastive tower.

Key-optional: requires GEMINI_API_KEY (skips gracefully; resumable — captions
land incrementally in data/captions.jsonl keyed by patch id).

Usage: GEMINI_API_KEY=... .venv/bin/python scripts/caption_corpus.py [--limit 100] [--source generated]
Cost:  ~$0.60 per 1,000 patches on gemini-2.5-flash.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR

MODEL = os.environ.get("SPS_CAPTION_MODEL", "gemini-2.5-flash")
CAPTIONS = DATA_DIR / "captions.jsonl"

PROMPT = """You are naming and describing synthesizer patches for an electronic
music production library. Listen to this clip of one synth patch playing a test
phrase. Describe the PATCH (the instrument's timbre/character), not the phrase.

Respond as JSON:
{"name": "<evocative 2-4 word patch name, like a hardware preset — no generic
words like Patch/Sound/Preset>",
 "caption": "<one dense sentence a producer would search by: timbre, character,
movement, texture, suggested musical role/genres>",
 "tags": ["<3-6 short lowercase tags>"]}"""


def caption_clip(flac_path: str, key: str, timeout: float = 60.0) -> dict | None:
    try:
        with open(flac_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")
        body = {
            "contents": [{"parts": [
                {"text": PROMPT},
                {"inline_data": {"mime_type": "audio/flac", "data": audio_b64}},
            ]}],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.4},
        }
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        out = json.loads(payload["candidates"][0]["content"]["parts"][0]["text"])
        return {"name": str(out.get("name", ""))[:60],
                "caption": str(out.get("caption", ""))[:400],
                "tags": [str(t)[:24] for t in out.get("tags", [])][:6]}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 = everything uncaptioned")
    ap.add_argument("--source", default="", help="filter: bundled|factory|third_party|generated")
    ap.add_argument("--sleep", type=float, default=0.2, help="between requests (rate kindness)")
    args = ap.parse_args()

    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        sys.exit("GEMINI_API_KEY not set — caption channel is dormant until it is. "
                 "Nothing else in the pipeline depends on this; run it any time.")

    renders = [json.loads(l) for l in (DATA_DIR / "renders.jsonl").read_text().splitlines()]
    best_render: dict = {}
    for r in renders:
        if r.get("status") == "ok" and r.get("activity", 0) > 0.05:
            cur = best_render.get(r["id"])
            if cur is None or r.get("rms", 0) > cur.get("rms", 0):
                best_render[r["id"]] = r

    done = set()
    if CAPTIONS.exists():
        for l in CAPTIONS.read_text().splitlines():
            row = json.loads(l)
            if "error" not in row:
                done.add(row["id"])

    todo = [(pid, r) for pid, r in sorted(best_render.items())
            if pid not in done and (not args.source or pid.startswith(args.source))]
    if args.limit:
        todo = todo[: args.limit]
    print(f"captioning {len(todo)} patches with {MODEL} "
          f"(~${len(todo) * 0.0006:.2f}); {len(done)} already done")

    t0 = time.time()
    with open(CAPTIONS, "a") as out:
        for i, (pid, r) in enumerate(todo, 1):
            result = caption_clip(str(DATA_DIR / r["flac"]), key)
            out.write(json.dumps({"id": pid, "probe": r["probe"], **(result or {})}) + "\n")
            out.flush()
            if i % 25 == 0:
                rate = i / (time.time() - t0)
                print(f"  {i}/{len(todo)} ({rate:.1f}/s, eta {(len(todo)-i)/max(rate,0.01)/60:.0f}m)",
                      flush=True)
            time.sleep(args.sleep)
    print(f"done in {(time.time()-t0)/60:.1f} min → {CAPTIONS}")


if __name__ == "__main__":
    main()
