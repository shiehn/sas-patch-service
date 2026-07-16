"""Anchor-conditioned evolutionary patch generation.

One campaign = one anchor phrase. Seeds are the anchor's nearest existing patches
(retrieval doubles as seed discovery); children derive by loadPatch(seed) + value
deltas (mod routing / wavetables preserved); fitness is CLAP similarity to the
prompt-ensembled anchor vector minus a capped negative-anchor penalty, measured on
the anchor's dominant probe. Identity control: exemplar-variance-scaled mutation +
within-anchor crossover + fitness policing (off-anchor offspring die).
"""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .probes import PROBES, Probe
from .render import probe_stats, render_probe, to_mono_16bit_ok

NEG_WEIGHT = 0.3


@dataclass
class Individual:
    ident: str
    parent_fxp: str                 # ALWAYS an original seed file
    delta: Dict[str, float]         # composed value delta on top of the parent
    gen: int
    op: str                         # 'seed' | 'mutate' | 'crossover'
    fitness: Optional[float] = None
    anchor_cos: Optional[float] = None
    neg_cos: Optional[float] = None
    stats: Dict[str, float] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class CampaignConfig:
    anchor_id: str
    anchor_text: str
    seeds: List[str]                # fxp paths, best-match first
    probe_id: str
    population: int = 24
    generations: int = 8
    elite: int = 6
    rng_seed: int = 20260715
    neg_weight: float = NEG_WEIGHT


def _probe_by_id(probe_id: str) -> Probe:
    for p in PROBES:
        if p.id == probe_id:
            return p
    raise KeyError(f"unknown probe {probe_id}")


# ---- render worker (module-level for pickling) -----------------------------------

def _eval_render(task: Tuple[str, Dict[str, float], str]) -> Tuple[Optional[np.ndarray], Dict[str, float]]:
    fxp_path, delta, probe_id = task
    try:
        audio = render_probe(fxp_path, _probe_by_id(probe_id), param_delta=delta or None)
        stats = probe_stats(audio)
        mono = to_mono_16bit_ok(audio)
        if mono is None or stats["peak"] < 1e-5 or stats["activity"] < 0.05:
            return None, stats
        return mono, stats
    except Exception as e:  # noqa: BLE001 — a broken child just scores None
        return None, {"error": str(e)}


class Campaign:
    def __init__(
        self,
        config: CampaignConfig,
        anchor_vec: np.ndarray,
        negative_vecs: np.ndarray,
        embedder,                    # sps.clap_embed.ClapEmbedder (main process, MPS/CPU)
        out_dir: Path,
        workers: int = 6,
    ) -> None:
        self.config = config
        self.anchor_vec = anchor_vec.astype(np.float32)
        self.negative_vecs = negative_vecs.astype(np.float32)
        self.embedder = embedder
        self.out_dir = out_dir
        self.workers = workers
        self.rng = random.Random(config.rng_seed)
        self.history: List[Dict[str, float]] = []
        self.counter = 0

        # genetics bookkeeping synth (values only; renders happen in workers)
        import sys

        from . import SURGE_SRC

        sys.path.insert(0, str(SURGE_SRC / "ignore" / "bpy" / "src" / "surge-python"))
        import surgepy  # type: ignore

        from .params import SurgeParams

        self._synth = surgepy.createSurge(48000)
        self._synth.loadPatch(config.seeds[0])
        self.params = SurgeParams(self._synth)
        self.allowed_osc = {
            sp.key: self.params.allowed_osc_type_values(sp) for sp in self.params.osc_type_specs()
        }

        # exemplar snapshots → identity-core sigma map
        self.seed_snapshots: Dict[str, Dict[str, float]] = {}
        for seed in config.seeds:
            self._synth.loadPatch(seed)
            self.seed_snapshots[seed] = self.params.snapshot()
        from .params import exemplar_sigma_map

        self.sigma_map = exemplar_sigma_map(list(self.seed_snapshots.values()), self.params.specs)

    # ---- genetics ---------------------------------------------------------------

    def _values_of(self, ind: Individual) -> Dict[str, float]:
        base = dict(self.seed_snapshots[ind.parent_fxp])
        base.update(ind.delta)
        return base

    def _new_id(self, op: str, gen: int) -> str:
        self.counter += 1
        return f"{self.config.anchor_id}-g{gen}-{op}{self.counter}"

    def _mutant(self, parent: Individual, gen: int) -> Individual:
        from .params import MutationConfig, mutate_values

        delta = dict(parent.delta)
        delta.update(
            mutate_values(
                self.params,
                self._values_of(parent),
                self.rng,
                MutationConfig(),
                sigma_map=self.sigma_map,
                allowed_osc=self.allowed_osc,
            )
        )
        return Individual(self._new_id("m", gen), parent.parent_fxp, delta, gen, "mutate")

    def _crossover(self, a: Individual, b: Individual, gen: int) -> Individual:
        from .params import crossover_values

        delta = dict(a.delta)
        delta.update(
            crossover_values(self.params, self._values_of(a), self._values_of(b), self.rng)
        )
        return Individual(self._new_id("x", gen), a.parent_fxp, delta, gen, "crossover")

    # ---- evaluation ---------------------------------------------------------------

    def _evaluate(self, pool: ProcessPoolExecutor, individuals: List[Individual]) -> None:
        todo = [ind for ind in individuals if ind.fitness is None]
        if not todo:
            return
        tasks = [(ind.parent_fxp, ind.delta, self.config.probe_id) for ind in todo]
        renders = list(pool.map(_eval_render, tasks, chunksize=2))

        waves = []
        alive: List[Individual] = []
        for ind, (mono, stats) in zip(todo, renders):
            ind.stats = stats
            if mono is None:
                ind.fitness = -1.0  # silent/broken — evolutionary dead end
            else:
                waves.append(mono)
                alive.append(ind)
        if alive:
            embs = self.embedder.embed_audio(waves)
            for ind, emb in zip(alive, embs):
                ind.embedding = emb.astype(np.float32)
                ind.anchor_cos = float(emb @ self.anchor_vec)
                ind.neg_cos = float(np.max(self.negative_vecs @ emb))
                ind.fitness = ind.anchor_cos - self.config.neg_weight * max(0.0, ind.neg_cos)

    # ---- main loop ------------------------------------------------------------------

    def run(self) -> Dict:
        cfg = self.config
        population: List[Individual] = [
            Individual(self._new_id("s", 0), seed, {}, 0, "seed") for seed in cfg.seeds
        ]
        while len(population) < cfg.population:
            population.append(self._mutant(self.rng.choice(population[: len(cfg.seeds)]), 0))

        best_ever: Optional[Individual] = None
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=self.workers) as pool:
            for gen in range(cfg.generations):
                self._evaluate(pool, population)
                population.sort(key=lambda i: i.fitness or -1.0, reverse=True)
                gen_best = population[0]
                if best_ever is None or (gen_best.fitness or -1) > (best_ever.fitness or -1):
                    best_ever = gen_best
                evaluated = [i for i in population if (i.fitness or -1) > -1]
                self.history.append({
                    "gen": gen,
                    "best": round(gen_best.fitness or -1, 4),
                    "best_anchor_cos": round(gen_best.anchor_cos or -1, 4),
                    "mean": round(float(np.mean([i.fitness for i in evaluated])) if evaluated else -1, 4),
                    "dead": len(population) - len(evaluated),
                })
                print(f"  gen {gen}: best={self.history[-1]['best']} "
                      f"(anchor {self.history[-1]['best_anchor_cos']}) "
                      f"mean={self.history[-1]['mean']} dead={self.history[-1]['dead']} "
                      f"[{time.time()-t0:.0f}s]", flush=True)

                if gen == cfg.generations - 1:
                    break
                elite = population[: cfg.elite]
                nxt = list(elite)
                while len(nxt) < cfg.population - 2:
                    nxt.append(self._mutant(self.rng.choice(elite), gen + 1))
                for _ in range(2):
                    a, b = self.rng.sample(elite, 2) if len(elite) >= 2 else (elite[0], elite[0])
                    nxt.append(self._crossover(a, b, gen + 1))
                population = nxt

        return self._finish(population, best_ever)

    def _finish(self, population: List[Individual], best_ever: Optional[Individual]) -> Dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        survivors = sorted(
            {i.ident: i for i in population if (i.fitness or -1) > -1}.values(),
            key=lambda i: i.fitness or -1.0,
            reverse=True,
        )[:8]

        saved = []
        for rank, ind in enumerate(survivors, 1):
            self._synth.loadPatch(ind.parent_fxp)
            self.params.apply(ind.delta)
            out = self.out_dir / f"{ind.ident}-f{(ind.fitness or 0):.3f}.fxp"
            self._synth.savePatch(str(out))
            saved.append({
                "rank": rank, "id": ind.ident, "fxp": str(out), "op": ind.op,
                "parent": ind.parent_fxp, "delta_size": len(ind.delta),
                "fitness": round(ind.fitness or -1, 4),
                "anchor_cos": round(ind.anchor_cos or -1, 4),
                "neg_cos": round(ind.neg_cos or -1, 4),
            })

        manifest = {
            "anchor_id": self.config.anchor_id,
            "anchor_text": self.config.anchor_text,
            "probe": self.config.probe_id,
            "seeds": self.config.seeds,
            "config": {
                "population": self.config.population,
                "generations": self.config.generations,
                "elite": self.config.elite,
                "rng_seed": self.config.rng_seed,
                "neg_weight": self.config.neg_weight,
            },
            "history": self.history,
            "best_ever": round(best_ever.fitness or -1, 4) if best_ever else None,
            "survivors": saved,
        }
        (self.out_dir / "campaign.json").write_text(json.dumps(manifest, indent=2))
        return manifest
