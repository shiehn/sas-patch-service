"""Quality judges: AudioBox-Aesthetics (CC-BY-4.0) + CLAP quality-contrast.

AudioBox scores four axes 1–10 (PQ production quality, CE content enjoyment,
CU usefulness, PC complexity). Per the proposal it's a COARSE junk filter, not a
fine ranker — gates are calibrated as percentiles of the FACTORY corpus
distribution (scripts/aesthetics_baseline.py), never absolute numbers.

CLAP quality-contrast is PAM-style with our own embedder: cosine against
"clean/well-produced" prompt ensembles minus cosine against "broken/noisy" ones,
computed directly on audio embeddings we already have. Zero extra model cost.

Note: audiobox's read_wav uses torchaudio.load (torchcodec on modern torchaudio);
we shim it to soundfile, which reads our FLACs everywhere.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

QUALITY_POSITIVE = [
    "a clean well-produced instrument sound",
    "a clear polished studio recording",
    "a professional high quality synthesizer sound",
]
QUALITY_NEGATIVE = [
    "a noisy broken distorted sound",
    "a harsh unpleasant grating noise",
    "a low quality muffled recording",
]

_predictor = None
_contrast_vecs: Optional[tuple] = None


def _shim_read_wav() -> None:
    import audiobox_aesthetics.infer as abi

    def _read_wav_sf(meta):  # type: ignore[no-untyped-def]
        import soundfile as sf
        import torch

        wav, sr = sf.read(meta["path"], dtype="float32", always_2d=True)
        t = torch.from_numpy(wav.T)
        if t.shape[0] > 1:
            t = t.mean(0, keepdim=True)
        return t, sr

    abi.read_wav = _read_wav_sf


def get_audiobox():
    global _predictor
    if _predictor is None:
        _shim_read_wav()
        from audiobox_aesthetics.infer import initialize_predictor

        _predictor = initialize_predictor()
    return _predictor


def audiobox_scores(flac_paths: List[str], batch: int = 16) -> List[Dict[str, float]]:
    """Score files → [{'CE':…, 'CU':…, 'PC':…, 'PQ':…}, …] (1–10 scales)."""
    predictor = get_audiobox()
    out: List[Dict[str, float]] = []
    for i in range(0, len(flac_paths), batch):
        chunk = [{"path": p} for p in flac_paths[i:i + batch]]
        out.extend({k: float(v) for k, v in row.items()} for row in predictor.forward(chunk))
    return out


def clap_quality_contrast(embedder, audio_embs: np.ndarray) -> np.ndarray:
    """PAM-style contrast on ALREADY-COMPUTED audio embeddings: mean cosine to
    positive quality prompts minus mean cosine to negative ones. Higher = cleaner."""
    global _contrast_vecs
    if _contrast_vecs is None:
        pos = embedder.embed_text(QUALITY_POSITIVE).mean(axis=0)
        neg = embedder.embed_text(QUALITY_NEGATIVE).mean(axis=0)
        pos /= np.linalg.norm(pos) + 1e-9
        neg /= np.linalg.norm(neg) + 1e-9
        _contrast_vecs = (pos.astype(np.float32), neg.astype(np.float32))
    pos, neg = _contrast_vecs
    return audio_embs @ pos - audio_embs @ neg
