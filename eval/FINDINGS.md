# Phase 0 findings log

## 2026-07-15 — first end-to-end retrieval (Day 1)

**Pipeline stood up end-to-end on a laptop**: 3,533 human-made patches (525 S&S bundled
+ 637 factory + 2,371 third-party) → 14,088 probe renders (4 probes, 8.3 min, median
0.21 s/render) → 13,902 CLAP embeddings (~9 min on MPS) → pooled index (7 MB) → CLI
search + golden eval.

### Wrapper ⇄ fxp equivalence (proposal de-risk item #1) — PROVEN
- Full nesting decoded & re-encodable (see `sps/wrapper.py` docstring):
  std-b64 → `<PLUGIN state=…>` → JUCE dot-b64 → `VC2!` copyXmlToBinary → `VST3PluginState`
  XML → IComponent == raw `sub3` stream == fxp chunk, byte-for-byte.
- surgepy loads converted bundled presets and renders audio. All 50 sampled
  round-trip checks pass. Bundled presets are patch revision 24 (== our pinned build).
- **Bonus**: every bundled preset carries its real name/author in the Surge XML
  (`Bonsai 808` by A.Liv, `Yearn` by Inigo Kennedy, …) — the `Preset_N` names were
  never the real identity, and provenance for THIRD_PARTY_NOTICES is recoverable.

### ⚠️ `laion/larger_clap_music` (HF hub port) is BROKEN
The proposal's recommended checkpoint collapses ALL audio (even sine vs white noise)
to ~0.98 cosine. Reproduced on transformers 4.57.6 AND 5.13.1, .bin AND safetensors,
weights load with zero missing keys — the hub conversion itself is bad.
Working Apache-2.0 siblings (sine-noise cosine ~0.42): **`larger_clap_music_and_speech`**
(current default) and `larger_clap_general`. `clap-htsat-unfused` also works.
Always run the synthetic-signal discrimination battery before trusting a checkpoint.

Also: transformers v5 returns `BaseModelOutputWithPooling` from `get_*_features`
(projected vector in `.pooler_output`); handled in `sps/clap_embed.py`.

### Zero-shot template matters (+19% relative)
Category-precision@5 over 34 objective golden queries (unfiltered corpus,
`larger_clap_music_and_speech`):

| Template | P@5 |
|---|---|
| raw query | 0.376 |
| "a recording of {q}" | 0.276 |
| **"this is the sound of {q}"** | **0.447** |
| "this is the sound of {q}, a {role} instrument" | 0.424 |
| "{q}, a {role} synthesizer sound" | 0.412 |

### Product-style (role-filtered) retrieval looks like the win condition
Unfiltered category-precision is a diagnostic, not the product metric — in the real
flow the role/category filter comes from MIDI+LLM exactly as today, and the semantic
ranking only has to answer "WHICH bass?". Within-category contrast test:

- "aggressive growling distorted bass" → RETRO HAKKOO, Neuro, Voltage Bees
- "soft round warm mellow bass" → Sub 1, Smoothie-sas-hi, Smootie, FM Bass 5
  (zero overlap — today's system returns a uniform-random bass for both)
- "jazzy … hollow standup bass" → FM bass 6, Digibass, Deep End (FM basses = the
  classic hollow/woody recipe)
- "80s synthwave lead" → Boll, AF Brassy Lead, JP8K Supersaw
- "warm electric piano" → Gumdrops (×3 near-dupe cluster: bundled Hi/Low + 3rd-party
  original — implicit dedup validation), Vintage EPish
- Best-probe column tracks query type (bass→riff/low probes, pad→chord probe):
  the multi-probe design is doing its job.

### Model bake-off (golden P@5, unfiltered, 34 objective queries)

| Model × template | raw `{q}` | "this is the sound of {q}" |
|---|---|---|
| `larger_clap_music_and_speech` (default, `data/index/`) | 0.376 | **0.447** |
| `larger_clap_general` (`data/index-general/`) | **0.459** | 0.424 |

Template interacts with model (helps music_and_speech, hurts general). Best objective
config: general + raw. Both models give credible, non-overlapping within-category
contrast rankings with different flavors (music_and_speech picked FM basses for
"hollow standup"; general picked literal "Fretless Bass" for "soft round mellow").
Within noise of each other on 34 queries → **the blind listening A/B is the decider.**
Query either config: `SPS_CLAP_MODEL=laion/larger_clap_general SPS_INDEX_DIR=index-general
scripts/search_cli.py "..." --template "{q}"` (model and index dir must always pair).

### Integration groundwork (2026-07-15, later same day)

**Shuffle design (S&S 🎲 button over retrieval):** temperature-weighted sampling
WITHOUT replacement from the semantic top-N (~16, temp ≈ 0.3), excluding
already-tried patches — never a revert to random, never a rigid next-best descent
(raw neighbors are near-clones; MMR + sampling keeps variety relevant). This is the
same idiom as the SDK's existing `pickTopKWeighted` (semantic-match.ts). Degradation:
no description / no index → today's random behavior; every applied candidate still
lands in sound history (with real names) for "go back" recovery.

**CLAP text tower → ONNX** (`scripts/export_text_onnx.py`; legacy exporter —
dynamo graphs trip `quantize_dynamic` shape inference):

| Variant | Size | Cosine vs torch | Single query (CPU) |
|---|---|---|---|
| fp32 | 501 MB | **1.00000 (exact)** | 6.2 ms |
| int8 | 126 MB | 0.988–0.994 | 3.7 ms |

fp32 is the gateway-endpoint answer (deps: `onnxruntime` + `tokenizers` + model file;
no torch server-side). int8 reserved for a possible future in-app path; fp16
conversion hit onnxconverter Cast-node type errors — parked, not needed.

**Client index pack** (`scripts/export_client_pack.py` → `data/client-pack/`,
**36.9 MB**): manifest + per-patch rows (fxp **content sha256** for intersection
against the user's locally-installed Surge library, real names/authors, per-probe
observation indices) + raw little-endian Float32 vector files readable directly into
`Float32Array` — no numpy on the client.

### POC session stats
Whole pipeline (proposal → working retrieval) stood up in one session; corpus build
end-to-end on laptop: render 8.3 min + embed ~9 min per model; $0 spent, no API calls.

### Design decision (2026-07-15): NO categorical subsetting in retrieval
Steve's call, empirically vindicated the same hour: the worst "failure" query
("deep sub bass", 0/5 on category-precision) actually returned **Simple 808 (filed
under Leads)** and **Sub Drop (filed under FX)** at #1/#2 — correct sounds wearing
wrong folder labels. Open retrieval surfaces exactly the patches the category system
misfiles; a category mask would have hidden them. Consequences:

1. **Retrieval never filters by category/role.** Guardrails become CONTINUOUS,
   render-measured constraints: probe weighting from the part's note profile
   (implemented — low-sustain rerank tightens the sub-bass query further), register/
   pitch-validity overlap with the actual MIDI (to add), behavioral features (later).
2. Role language lives in the QUERY TEXT (the description already says "bassline")
   and in probe-weight selection — not in a mask. Derived role affinity (zero-shot
   anchors + probes-that-speak) survives only as an optional soft score prior and
   UI vocabulary. `--category` stays in the CLI as a debugging tool only.
3. **Metric implication: category-precision@k is DEPRECATED as a headline metric** —
   it penalizes cross-category hits that are the feature's whole point, so 0.45
   UNDERSTATES true quality. GATE 1 rests on the blind listening A/B (+ self-retrieval
   sanity), as the proposal's evaluation section already prescribed.

### Open items
- [x] GATE 1 listening protocol: **`eval/listen_ab.py` built** — blind A/B, semantic
      top-5 vs today's random-within-category, sides shuffled server-side, votes →
      data/ab_votes.jsonl. Run `.venv/bin/python eval/listen_ab.py` →
      http://localhost:8765 ; tally with `--report` (target: ≥70% semantic wins on
      decided queries). STILL TO DO: Steve actually votes through the 48 queries
      (and optionally repeats with SPS_INDEX_DIR=index-general SPS_TEMPLATE="{q}"
      for the model bake-off decision)
- [ ] Caption channel (Gemini) — second retrieval channel + display names; needs
      GEMINI_API_KEY locally (gateway holds the prod key)
- [ ] Silent-render follow-up: 44/14,132 probe renders silent (expected: wrong-register
      probes on niche patches); fold into role-affinity derivation
- [ ] Role-affinity derivation (probes-that-speak + register map) → replaces category
      substring filter with the product's role filter
