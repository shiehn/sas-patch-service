# sas-patch-service

**Natural-language → Surge XT patch retrieval.** Type *"jazzy bassline with a hollow
standup bass sound"* and get back the synth patches that actually *sound* like that —
matched by audio embeddings, not by names, folders, or tags.

A complete, self-contained pipeline — corpus ingest, headless probe rendering,
audio-embedding index, search CLI, and a listening-test harness — that runs
end-to-end on a laptop over the ~3.5k human-made patches distributed with Surge XT.
No cloud, no GPU required. Built as research for
[Signals & Sorcery](https://signalsandsorcery.org), but useful on its own to anyone
with a pile of synth patches and a need to find the right one by describing it.

```
$ .venv/bin/python scripts/search_cli.py "aggressive growling distorted bass" -k 4
 1  0.517  RETRO HAKKOO      2  0.491  Neuro      3  0.487  Voltage Bees   ...

$ .venv/bin/python scripts/search_cli.py "soft round warm mellow bass" -k 4
 1  0.562  Sub 1             2  0.551  Smoothie   3  0.533  Smootie        ...
```

Same corpus, same code — disjoint results. The ranking heard the difference.

## How it works

A synth patch is not a recording — it's a *conditional* sound machine. So the system
listens to every patch played several ways, and searches over what it heard:

```
                     OFFLINE (index build)
 ┌─ corpus ─────────────┐
 │ Surge XT's in-box    │   1. render: each patch is loaded into a headless Surge
 │ library (factory +   │      (surgepy, pinned @ release_xt_1.3.4) and played
 │ third-party .fxp) +  │──▶   through 4 deterministic MIDI probes: a low 8th-note
 │ optional app presets │      riff, a sustained C2, a mid-register phrase, a held
 └──────────────────────┘      Cmaj7 → 48 kHz audio per (patch, probe)
                            2. embed: every render goes through the AUDIO tower of
                               LAION-CLAP → one 512-d vector per (patch, probe);
                               per-patch vector = mean of its probe vectors
                            3. index: vectors + metadata land in data/index/

                     ONLINE (query)
 "hollow standup bass" ──▶ CLAP TEXT tower ──▶ 512-d vector in the SAME joint space
                            │
                            ├─ stage 1: cosine vs pooled per-patch vectors
                            ├─ stage 2: rerank top candidates by their single
                            │           best-matching probe (register/articulation
                            │           awareness falls out of the probe design)
                            └─ top-k patches, each openable as a stock Surge .fxp
```

Because CLAP was contrastively trained on millions of (audio, caption) pairs, text
and audio land in one shared vector space — the query never touches patch names or
category folders. **There is deliberately no category filtering**: the folder
taxonomy is author-chosen and noisy, and open retrieval routinely surfaces correct
sounds that folders misfile (a perfect 808 sub filed under "Leads", a sub-drop under
"FX"). Guardrails are continuous instead: probe weighting by how the part will be
played, and (planned) register-validity measured from the renders.

Two supporting pieces make this useful beyond a demo:

- **`sps/wrapper.py`** — lossless converters between stock Surge `.fxp` files and the
  Tracktion/JUCE plugin-state wrapper that hosts persist (std-base64 → `<PLUGIN>`
  element → JUCE MemoryBlock dot-base64 → `VC2!` `copyXmlToBinary` container →
  `VST3PluginState` XML → the raw `sub3` patch stream, which is byte-identical to an
  fxp chunk). Any patch this system retrieves can be applied to a hosted Surge XT
  instance, and any host-saved state can be turned back into a portable `.fxp`.
- **`eval/`** — a versioned golden query set, an offline eval runner, and a blind
  A/B listening app (semantic top-5 vs. random-pick baseline, sides shuffled) so
  retrieval quality is measured by ears, not vibes.

Empirical findings so far (including one broken-model gotcha worth reading before
you trust any CLAP checkpoint): **[`eval/FINDINGS.md`](eval/FINDINGS.md)**.

## Setup

Prereqs: macOS with Xcode command-line tools, `cmake`, and [`uv`](https://docs.astral.sh/uv/)
(Linux should work identically; Windows: surgepy has upstream MSVC CI but this repo's
pipeline is currently exercised on macOS only — it's offline dev tooling, nothing
here ships to end users).

```bash
# 1. Python env (3.12) + deps
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt

# 2. Surge XT source, pinned so generated/saved patches use streaming revision 24
#    (loads in Surge 1.3.4 AND newer). Also provides the 3,008-patch corpus.
git clone --recursive --depth 1 --shallow-submodules --branch release_xt_1.3.4 \
  https://github.com/surge-synthesizer/surge.git third_party/surge

# 3. (macOS 15+/26 SDK only) the pinned JUCE uses a removed API in a build-time
#    helper; stub it out — window snapshots are never used headlessly:
git -C third_party/surge/libs/JUCE apply "$PWD/patches/juce-macos26-cgwindowlist.patch"

# 4. Build surgepy (the official headless Surge python bindings).
#    Both python vars matter: the pinned pybind11 reads the legacy PYTHON_EXECUTABLE.
cmake -S third_party/surge -B third_party/surge/ignore/bpy \
  -DSURGE_BUILD_PYTHON_BINDINGS=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DPYTHON_EXECUTABLE=$PWD/.venv/bin/python -DPython_EXECUTABLE=$PWD/.venv/bin/python
cmake --build third_party/surge/ignore/bpy --parallel --target surgepy
```

## Run the pipeline — retrieval

```bash
# ingest the in-box Surge library (3,008 patches) into data/corpus.jsonl
.venv/bin/python scripts/ingest_inbox.py

# optional, Signals & Sorcery development only (needs ../sas-app checkout)
.venv/bin/python scripts/ingest_bundled.py
.venv/bin/python scripts/equivalence_check.py

# render probes → embed → search
.venv/bin/python scripts/render_probes.py
.venv/bin/python scripts/embed_clap.py
.venv/bin/python scripts/search_cli.py "glassy shimmering bell pluck" -k 8 --play

# offline eval + blind A/B (semantic vs random baseline)
.venv/bin/python scripts/eval_golden.py -k 5
.venv/bin/python eval/listen_ab.py            # → http://localhost:8765
.venv/bin/python eval/listen_ab.py --report
```

## Run the pipeline — generation (the corpus factory)

Patches are *bred*, not just found. Campaigns are anchor-conditioned evolution:
seeds come from the retrieval index itself (an anchor's nearest existing patches),
children derive by `loadPatch(parent)` + parameter deltas (modulation routing and
wavetables preserved), fitness is CLAP similarity to the prompt-ensembled anchor
minus a negative-anchor penalty, and survivors face a gate stack calibrated on the
factory corpus — never absolute thresholds.

```bash
# 1. score the anchor vocabulary against the index: covered / sparse / empty
.venv/bin/python scripts/anchor_coverage.py            # eval/anchors_v2.json, core tier

# 2. one campaign (~seconds) or a sweep over every under-covered anchor
.venv/bin/python scripts/run_campaign.py pad-dub-chord --pop 40 --gens 12
.venv/bin/python scripts/run_sweep.py --pop 40 --gens 12
#    retry rounds with transient-aware fitness + FX-weighted mutation:
.venv/bin/python scripts/run_sweep.py --tag r2 --profile transient --anchors <ids>

# 3. verification: full probe renders + gates (objective, clarity margin,
#    negative contrast, novelty-vs-index, AudioBox/CLAP quality floors at
#    factory percentiles; optional Gemini listening judge via SPS_JUDGE=1)
.venv/bin/python scripts/verify_survivors.py pad-dub-chord
.venv/bin/python scripts/reverify_all.py               # after gate changes

# 4. human loops: taste votes calibrate thresholds; blind A/B decides shipping
.venv/bin/python scripts/listen_survivors.py --limit 30 && \
.venv/bin/python scripts/listen_survivors.py --calibrate
.venv/bin/python eval/listen_ab.py --mode gate2        # generated vs human, blinded

# 5. absorb winners into the corpus; ship a curated pack
.venv/bin/python scripts/absorb_survivors.py --per-anchor 3
.venv/bin/python scripts/embed_clap.py && .venv/bin/python scripts/export_client_pack.py \
  --curated data/curated_ship_list.json
.venv/bin/python scripts/build_index_pack.py --version N

# extras: random-generation control arm; Gemini captions (needs GEMINI_API_KEY)
.venv/bin/python scripts/run_random_arm.py --count 800
.venv/bin/python scripts/caption_corpus.py --limit 100
```

Ground rules baked into the factory: **depth over breadth** (the anchor vocabulary
is dense in electronic sound design; acoustic imitations are a research tier), no
categorical filtering anywhere in retrieval, generated content ships only after a
blind generated-vs-human A/B — losing families stay in the lab.

Model/index variants for bake-offs (model and index dir must always pair):

```bash
SPS_CLAP_MODEL=laion/larger_clap_general SPS_INDEX_DIR=index-general \
  .venv/bin/python scripts/embed_clap.py
SPS_CLAP_MODEL=laion/larger_clap_general SPS_INDEX_DIR=index-general \
  .venv/bin/python scripts/search_cli.py "..." --template "{q}"
```

## Repo layout

| Path | What |
|---|---|
| `sps/juce_b64.py` | JUCE MemoryBlock "dot-base64" codec (not RFC 4648!) |
| `sps/wrapper.py` | Tracktion `<PLUGIN>` state ⇄ Surge patch stream, all layers |
| `sps/fxp.py` | Surge `.fxp` reader/writer (VST2 FXP chunk container) |
| `sps/probes.py` | The deterministic MIDI probe suite (v1: 4 probes) |
| `sps/render.py` | Headless probe rendering through surgepy |
| `sps/clap_embed.py` | LAION-CLAP audio/text embedding (+ broken-checkpoint notes) |
| `scripts/` | Pipeline entry points (ingest → render → embed → search → eval) |
| `eval/golden_queries.json` | Versioned golden query set |
| `eval/listen_ab.py` | Blind A/B listening app (stdlib-only web server) |
| `eval/FINDINGS.md` | Running findings log — read this |
| `patches/` | Build fixes for the pinned third-party tree |
| `data/`, `third_party/` | (gitignored) corpus, renders, indexes, Surge source |

## Licensing

- **This repo's code: MIT** (see `LICENSE`).
- It *builds and runs* [Surge XT](https://github.com/surge-synthesizer/surge)
  (GPL-3.0) locally via its official `surgepy` bindings, and reads the patch content
  that ships with Surge. **Neither Surge's code nor its patch content is included in
  or redistributed by this repository** (`third_party/` and `data/` are gitignored).
  If you distribute a bundle that includes surgepy or Surge content, the GPL and
  Surge's content licensing apply to that distribution — see Surge's
  [licensing FAQ](https://surge-synthesizer.github.io/faq/) and
  [issue #6741](https://github.com/surge-synthesizer/surge/issues/6741).

## Status / roadmap

Phase 0 of the Signals & Sorcery *Semantic Patch Service* initiative — prove
retrieval quality over the existing human-made corpus before building anything
bigger. Next: blind listening gate, register-validity features, a caption-based
second retrieval channel, then (phased) generation of new curated patches and a
serving tier. Design doc lives in the S&S platform repo
(`sas-app/docs/semantic-patch-service-proposal.md`).
