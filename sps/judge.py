"""LLM listening judge (Gemini audio-in) — key-optional taste proxy.

Sends a rendered clip to Gemini with a producer-context rubric and gets back
{keep, score, reason}. This is the "larger models" lever used the right way
per depth-over-breadth: the big multimodal model as a JUDGE, calibrated against
Steve's votes — not as the embedder.

Inert without GEMINI_API_KEY (returns None; callers skip the gate). Uses plain
REST so the pipeline gains no SDK dependency. ~$0.001/clip on flash models.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.request
from typing import Dict, Optional

JUDGE_MODEL = os.environ.get("SPS_JUDGE_MODEL", "gemini-2.5-flash")

RUBRIC = """You are an experienced electronic music producer auditioning synthesizer
patches for a professional sound library. Listen to this clip of a synth patch
playing a test phrase. It was generated to match the description: "{anchor}".

Judge PRODUCTION-READINESS, not the test phrase's musicality. Consider: is the
timbre pleasant/intentional (not accidental-sounding)? free of harsh artifacts,
mud, or ear-fatiguing resonance? would a producer plausibly keep this sound in a
track? does it evoke the description?

Respond as JSON: {{"keep": true|false, "score": 1-10, "reason": "<one short sentence>"}}"""


def available() -> bool:
    return bool(os.environ.get("GEMINI_API_KEY"))


def judge_clip(flac_path: str, anchor_text: str, timeout: float = 60.0) -> Optional[Dict]:
    """Return {'keep': bool, 'score': float, 'reason': str} or None when no key /
    on any failure (the judge is best-effort by contract)."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None
    try:
        with open(flac_path, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode("ascii")
        body = {
            "contents": [{
                "parts": [
                    {"text": RUBRIC.format(anchor=anchor_text)},
                    {"inline_data": {"mime_type": "audio/flac", "data": audio_b64}},
                ],
            }],
            "generationConfig": {"response_mime_type": "application/json", "temperature": 0.1},
        }
        req = urllib.request.Request(
            f"https://generativelanguage.googleapis.com/v1beta/models/{JUDGE_MODEL}:generateContent",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read())
        text = payload["candidates"][0]["content"]["parts"][0]["text"]
        verdict = json.loads(text)
        return {
            "keep": bool(verdict.get("keep")),
            "score": float(verdict.get("score", 0)),
            "reason": str(verdict.get("reason", ""))[:200],
        }
    except Exception:  # noqa: BLE001
        return None
