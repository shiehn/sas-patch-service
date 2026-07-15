"""Export the CLAP TEXT tower to ONNX (+ int8-quantized variant) and verify parity.

The exported graph maps (input_ids, attention_mask) → L2-normalized 512-d query
embedding — the exact vector the index was built against. This one artifact serves
either query-embedding deployment:
  (a) a tiny gateway endpoint: python `tokenizers` + `onnxruntime` + this file
  (b) in-app via onnxruntime-node + a JS tokenizer

Usage: .venv/bin/python scripts/export_text_onnx.py
Outputs: data/onnx/clap-text-{fp32,int8}.onnx (+ tokenizer.json) and a parity report.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sps import DATA_DIR
from sps.clap_embed import MODEL_ID, ClapEmbedder

OUT = DATA_DIR / "onnx"
QUERIES = [
    "jazzy bassline with a hollow standup bass sound",
    "aggressive growling distorted bass",
    "dreamy ambient pad with slow evolving movement",
    "glassy shimmering bell pluck",
    "this is the sound of deep sub bass",
]


class TextTower(  # thin wrapper: text encoder + projection + L2 norm, tensor-only I/O
    __import__("torch").nn.Module
):
    def __init__(self, clap_model):
        super().__init__()
        self.text_model = clap_model.text_model
        self.text_projection = clap_model.text_projection

    def forward(self, input_ids, attention_mask):
        import torch

        out = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.text_projection(out.pooler_output)
        return emb / torch.linalg.norm(emb, dim=-1, keepdim=True)


def main() -> None:
    import torch
    from onnxruntime import InferenceSession
    from onnxruntime.quantization import QuantType, quantize_dynamic

    OUT.mkdir(parents=True, exist_ok=True)
    embedder = ClapEmbedder(device="cpu")
    tower = TextTower(embedder.model).eval()

    tok = embedder.processor.tokenizer
    tok.save_pretrained(OUT / "tokenizer")
    enc = tok(QUERIES, return_tensors="pt", padding=True)

    fp32_path = OUT / "clap-text-fp32.onnx"
    torch.onnx.export(
        tower,
        (enc["input_ids"], enc["attention_mask"]),
        str(fp32_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["text_embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "text_embedding": {0: "batch"},
        },
        opset_version=17,
        dynamo=False,  # legacy exporter: dynamo graphs trip quantize_dynamic shape infer
    )
    int8_path = OUT / "clap-text-int8.onnx"
    quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)

    # ---- parity + latency vs the torch reference used to build the index
    ref = embedder.embed_text(QUERIES)
    feeds = {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()}
    report = {}
    for name, path in [("fp32", fp32_path), ("int8", int8_path)]:
        sess = InferenceSession(str(path), providers=["CPUExecutionProvider"])
        out = sess.run(None, feeds)[0]
        cos = np.sum(out * ref, axis=1) / (np.linalg.norm(out, axis=1) * np.linalg.norm(ref, axis=1))
        one = {k: v[:1] for k, v in feeds.items()}
        for _ in range(3):
            sess.run(None, one)  # warm
        t0 = time.time()
        n = 20
        for _ in range(n):
            sess.run(None, one)
        report[name] = {
            "size_mb": round(path.stat().st_size / 1e6, 1),
            "cosine_vs_torch": [round(float(c), 5) for c in cos],
            "single_query_ms": round((time.time() - t0) / n * 1000, 1),
        }

    print(json.dumps({"model": MODEL_ID, **report}, indent=2))
    worst = min(min(r["cosine_vs_torch"]) for r in report.values())
    print(f"\nworst cosine vs torch reference: {worst:.5f} "
          f"({'OK — same retrieval behavior' if worst > 0.999 else 'CHECK int8 tolerance' if worst > 0.99 else 'PARITY PROBLEM'})")


if __name__ == "__main__":
    main()
