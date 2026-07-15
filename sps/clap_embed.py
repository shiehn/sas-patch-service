"""LAION-CLAP embeddings (Apache-2.0 HF ports).

512-d joint text/audio space; 48 kHz feature extractor (matches our render rate).
Audio is RMS-normalized before embedding so loudness doesn't dominate similarity.

⚠️ Empirical finding (2026-07-15): the hub port `laion/larger_clap_music` is BROKEN —
its audio tower collapses every input (even sine vs white noise) to ~0.98 cosine, on
both transformers 4.57 and 5.13, both .bin and safetensors. Its siblings
`larger_clap_music_and_speech` and `larger_clap_general` discriminate correctly
(sine-noise ~0.42), as does `clap-htsat-unfused`. Default = the music-tuned working
sibling; override with SPS_CLAP_MODEL for bake-offs.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

MODEL_ID = os.environ.get("SPS_CLAP_MODEL", "laion/larger_clap_music_and_speech")
TARGET_RMS_DB = -20.0


def pick_device() -> str:
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class ClapEmbedder:
    def __init__(self, device: str | None = None, model_id: str | None = None) -> None:
        import torch
        from transformers import ClapModel, ClapProcessor

        self.model_id = model_id or MODEL_ID
        self.device = device or pick_device()
        self.model = ClapModel.from_pretrained(self.model_id).to(self.device).eval()
        self.processor = ClapProcessor.from_pretrained(self.model_id)
        self.torch = torch

    @staticmethod
    def normalize_loudness(mono: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(np.square(mono)))) or 1e-9
        target = 10 ** (TARGET_RMS_DB / 20)
        out = mono * (target / rms)
        peak = float(np.max(np.abs(out))) if out.size else 0.0
        if peak > 0.99:
            out = out / peak * 0.99
        return out.astype(np.float32)

    def _to_tensor(self, out):
        # transformers <5 returns the projected tensor directly; v5 wraps it in a
        # BaseModelOutputWithPooling whose pooler_output IS the projected feature.
        if self.torch.is_tensor(out):
            return out
        return out.pooler_output

    def embed_audio(self, waves_48k: List[np.ndarray]) -> np.ndarray:
        waves = [self.normalize_loudness(w) for w in waves_48k]
        inputs = self.processor(
            audio=waves, sampling_rate=48000, return_tensors="pt", padding=True
        ).to(self.device)
        with self.torch.no_grad():
            emb = self._to_tensor(self.model.get_audio_features(**inputs))
        emb = emb.cpu().float().numpy()
        return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)

    def embed_text(self, texts: List[str]) -> np.ndarray:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            emb = self._to_tensor(self.model.get_text_features(**inputs))
        emb = emb.cpu().float().numpy()
        return emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
