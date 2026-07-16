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

### Phase-1 generation strategy review (2026-07-15)
Compared a "brute-force random patches + CLAP vocabulary-threshold filter" proposal
against the plan. Verdict: keep **seeded mutation + CLAP-guided evolution** primary
(synth1B1: random sampling = perceptually clumped junk; steering > filtering for the
same CLAP compute), but adopt four refinements into the gate stack: category-clarity
margin (best−mean anchor similarity), negative-anchor contrast set, prompt-ensembled
anchors, cluster-then-caption vocabulary growth. Plus a ~50k **random-arm control**
through the identical gates so random-vs-seeded is settled by campaign metrics, not
argument. Anchors are for gating/steering only — retrieval stays open-vocabulary.
Details in the proposal §6.2.

### Anchor-conditioned seeding (2026-07-15, Steve's refinement)
Campaign parent selection = the anchor's top-k nearest patches in the existing index
(retrieval doubles as seed discovery). Live demo: "muted trumpet with a harmon mute"
peaks at ~0.58 against Flute/Clarinet/brassy leads → a SPARSE anchor, ideal campaign
target; well-covered anchors (0.7+) skip generation entirely — the anchor vocabulary
self-prioritizes by need. Identity control while deviating: CLAP-to-anchor fitness
(off-anchor offspring die), identity-core parameter analysis (low variance across
exemplars = preserve; high variance = explore), within-anchor crossover. Recorded in
proposal §6.2 as the parent-selection policy.

### Phase 1 vertical slice LIVE (2026-07-15) — first generation campaign
The full anchor-conditioned evolution loop runs end-to-end:

- **Parameter layer** (`sps/params.py`): 698 mutable params across OSC/MIX/FILTER/
  ENV/LFO/FX (GLOBAL excluded); children derive by loadPatch(parent)+value deltas
  (the ~700 params do NOT carry mod routings/wavetables — deltas preserve them);
  osc-type guard maps the enum by display-probing and forbids Wavetable/Window/
  Audio In (indices 2/7/4 at 1.3.4); exemplar-variance-scaled mutation.
- **Anchor coverage scan** (116 anchors, prompt-ensembled): 24 covered / 52 sparse /
  40 empty — the corpus is bass-rich (bass-wobble 0.717) and acoustic-imitation-poor.
  Campaign targets are self-prioritizing.
- **First campaign — "muted trumpet with a harmon mute"** (sparse, ceiling 0.562):
  pop 24 × 8 generations = **10 seconds wall**. Best evolved anchor-similarity
  **0.654 (beat the corpus ceiling by +0.09)**; fitness = anchor − 0.3·max(neg) held
  back noise-drift (top mutant had neg 0.40 — the penalty is load-bearing).
  Crossover produced 2/8 survivors. **8/8 survivors pass v1 gates** (objective,
  clarity margin ~0.48, negative Δ ~0.28, novelty 0.93–0.96 < 0.985 dup threshold).
- Cost math: ~10 s/campaign → all 92 sparse+empty anchors ≈ minutes, not hours.
  Population/generation budgets have huge headroom before compute matters.
- CLAP-says vs ear-says: the numbers claim "more muted-trumpet-like than anything
  in the corpus"; the listening verdict (and GATE 2's blind A/B vs factory) remains
  the binding quality bar. Aesthetics scoring (AudioBox/PAM) not yet in the gates.

### Full sweep + absorption — THE LOOP IS CLOSED (2026-07-15, later)
- **Sweep**: 91 campaigns (all sparse+empty anchors), 62 min, pop 32 × 10 gens.
  **85/91 beat their corpus ceiling**; 658 gate-passing survivors. Biggest gains in
  EMPTY anchors (pan flute +0.24, solo violin +0.22, harp +0.16). Steve's ear test
  passed on the first campaign ("played the trumpet and it's GREAT"). Notable:
  the hammond campaign gained similarity but went **0/8 on gates** — clarity/negative
  gates rejected a whole campaign with no human involved.
- **Absorption**: top-3 per anchor + intra-campaign dedup (≥0.97 cos) →
  **193 generated patches** absorbed with campaign provenance; corpus now
  **3,716 patches / 14,674 observations**; index pack v2 exported (38.9 MB, local).
- **Closed-loop retrieval**: "airy pan flute" → generated top-3 (0.617 vs old
  ceiling 0.445); "muted trumpet" → generated #1; "expressive solo violin" →
  generated #1/#2/#4. Search finds gaps → evolution fills them → search serves them.
- **Ship gate**: pack v2 does NOT ship until GATE 2 — blind A/B of generated vs
  factory over golden queries (listen_ab.py) — passes. One heard anchor ≠ 193
  vetted patches.

### Guiding principle (2026-07-15, Steve): DEPTH OVER BREADTH
Not the EVERYTHING app — the goal is *truly awesome music*, and the house taste is
electronic. Consequences for this pipeline: anchor vocabulary v2 goes DENSE in
electronic sound-design territory (many shades of bass/pad/lead/texture, production
language) and drops the acoustic-imitation checklist to research-only; absorption
prefers top-1-per-anchor with stricter gates over corpus growth; every automated
threshold (clarity, negΔ, aesthetics, future LLM-judge) gets calibrated against
Steve's listening votes before any scale-up. Corpus KPI stays "does a producer say
GREAT more often" — never counts.

### Depth-over-breadth build-out (2026-07-15, latest)
- **Anchor vocabulary v2** (`eval/anchors_v2.json`): 126 core anchors DENSE in
  electronic sound design (32 bass shades, 22 pads, dub/garage/rave stabs, textures)
  in producer language; the v1 acoustic-imitation set demoted to a 24-anchor
  research tier; taste-negatives added ("cheap general midi demo", "muddy rumble").
  V2-core coverage vs the enlarged corpus: **19 empty / 60 sparse / 47 covered**.
- **Quality judges wired into verification** (all factory-calibrated, never absolute):
  AudioBox-Aesthetics PQ/CE floors at the factory 25th percentile
  (`scripts/aesthetics_baseline.py`, 240-sample calibration, 44 s); CLAP
  quality-contrast (PAM-style prompt pairs on existing embeddings — zero new model);
  optional **Gemini LLM listening judge** (`sps/judge.py`, REST, key-optional,
  SPS_JUDGE=1) with a producer rubric. Trumpet campaign re-verified: 7/8 pass with
  the new floors (one culled — selection pressure rising as intended).
- **Taste calibration loop** (`scripts/listen_survivors.py`): plays survivors,
  records keep/reject votes with all metrics; `--calibrate` fits threshold floors
  to the votes (best-accuracy split per metric). Steve's ear = the ground truth all
  gates get tuned against.
- Vector-store versioning fixed: anchor_vectors.npz always carries every tier;
  verification of old campaigns falls back to embedding the manifest's own text.

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
