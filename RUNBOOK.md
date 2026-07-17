# Runbook — adding generated patches to the product

The recurring ritual: **breed → gate → listen → ship**. Generation and gating are
fully automatic; your ears gate *shipping* (blind A/B, per anchor, once — re-votes
supersede). A full cycle is ~1–3 hours of laptop compute plus ~10 minutes of
listening.

Everything runs from this repo with the `.venv` python. Costs: $0 unless noted.

---

## 0. Decide the territory

Add new anchors (producer-language descriptions of sounds you want to exist) to the
current `eval/anchors_vN.json` — or create `anchors_vN+1.json` bumping `version`.
House rules: **depth over breadth** (dense electronic territory; acoustic imitations
go in `tier: "research"`), and aim at ground where the corpus is weak — evolution
wins *territory*, not head-to-head fights with strong human patches (measured:
53% ship-rate in new territory vs 0% retrying lost families).

```bash
.venv/bin/python scripts/anchor_coverage.py --vocab anchors_vN.json
# → covered / sparse / empty per anchor; sparse+empty are your campaign targets
```

## 1. Breed

```bash
# everything under-covered:
.venv/bin/python scripts/run_sweep.py --pop 40 --gens 12
# or explicit targets:
.venv/bin/python scripts/run_sweep.py --anchors pad-foo,texture-bar --pop 48 --gens 14
```

~1–2 min per anchor. Verification (all probes + gate stack) runs inline: objective
checks, clarity margin, negative-anchor contrast, novelty-vs-index, AudioBox/CLAP
quality floors at factory percentiles. Survivors + metrics land in
`data/campaigns/<anchor>/`.

Known gate caveat: the AudioBox floors are class-blind and currently punish
intentional lo-fi/noise craft (vinyl crackle, rain ambience). Until per-family
floors land, expect noisy-texture anchors to fail gates despite good similarity.

## 2. Absorb the gate-passers

```bash
.venv/bin/python scripts/absorb_survivors.py --per-anchor 3        # or --dirs a,b,c
.venv/bin/python scripts/embed_clap.py                             # ~10 min (MPS)
```

This puts them in the *local* corpus/index only — nothing ships yet.

## 3. Listen (the ship gate)

```bash
.venv/bin/python eval/listen_ab.py --mode gate2 --anchors <new-anchor-ids>
# → http://localhost:8765 — blinded pairs: generated top-5 vs human top-5
#   per description. Vote A/B/tie; tie counts as parity (ships). ~40s per pair.
.venv/bin/python eval/listen_ab.py --report                        # tally
```

Rules of the gate: per-anchor verdicts; your **latest vote supersedes** earlier
rounds; won/tied anchors become ship-eligible; losses stay lab-side. (Open policy
question, deliberately unresolved: whether *empty*-territory anchors — where the
"human side" barely exists — may ship on gates + spot-checks without a full A/B.)

## 4. Ship

```bash
.venv/bin/python scripts/build_ship_list.py            # votes → curated_ship_list.json
.venv/bin/python scripts/export_client_pack.py --curated data/curated_ship_list.json
.venv/bin/python scripts/build_index_pack.py --version <N+1>       # NEVER reuse a version
gsutil cp -n dist/sas-patch-index-pack-v<N+1>.zip gs://docs-assets/
gsutil acl ch -u AllUsers:R gs://docs-assets/sas-patch-index-pack-v<N+1>.zip   # ⚠️ required — uploads land private
```

Then in **sas-app** update `PATCH_INDEX_PACK` in
`src/shared/constants/sample-packs.ts`: `expectedVersion`, `downloadUrl`,
`sizeBytes`, `sha256` (all printed by `build_index_pack.py`). Run
`npx jest src/main/services/__tests__/pack-download-service.test.ts` + `npx tsc
--noEmit`, commit. Users' apps auto-download the new pack at next launch.

What the pack contains: the **index** for the human corpus (users' own Surge
installs supply those files) plus the **carried .fxp bytes of curated generated
patches** (they exist nowhere else — `generated/` dir inside the pack).

## Optional extras

```bash
# taste-threshold calibration votes (keep/reject singles, feeds gate tuning):
.venv/bin/python scripts/listen_survivors.py --limit 30 && \
.venv/bin/python scripts/listen_survivors.py --calibrate

# evocative names + a future text-retrieval channel (~$0.60/1k patches):
GEMINI_API_KEY=$(cat .gemini_key) .venv/bin/python scripts/caption_corpus.py
```

## Hard-won rules (don't relearn these)

- **Model and index must always pair** — vectors from different CLAP models are
  incomparable; the client refuses mismatches. Changing the embedding model means
  re-embedding everything + a coordinated gateway (ONNX) + pack release.
- Automated taste doesn't exist: three experiments (metric floors, quality judges,
  LLM listener) all failed to predict Steve's keep/reject within gate-passers.
  The blind A/B is the only binding instrument.
- Published pack versions are immutable — always bump, never overwrite.
- Bucket uploads are private by default; apply the AllUsers ACL every time.
