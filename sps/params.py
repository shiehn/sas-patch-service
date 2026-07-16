"""Surge parameter layer for generation: enumerate, snapshot, mutate, crossover.

Ground rules (empirically established):
- Value accessors (getParamVal/Min/Max/Def/ValType, setParamVal, getParamDisplay)
  take the SurgePyNamedParam handle; handles belong to ONE synth instance and are
  re-enumerated per instance. Enumeration order is deterministic for a build, so
  a (group, entry, index) key is stable across instances.
- The ~766 control-group params do NOT carry modulation routings, step-seq/MSEG
  data, or embedded wavetables. Children therefore derive by loadPatch(parent)
  followed by setParamVal deltas — never by replaying values onto an init patch.
- Osc-type mutations never select wavetable-data types (Wavetable/Window — their
  sample data embeds into saved patches; provenance policy) nor Audio Input
  (silent without input).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

# surgepy getParamValType returns strings: 'int' | 'bool' | 'float'
VT_INT = "int"
VT_BOOL = "bool"
VT_FLOAT = "float"

# Mutable control groups. GLOBAL is excluded (scene mode, MIDI config, master
# levels — logistics, not timbre).
MUTABLE_GROUPS = ("OSC", "MIX", "FILTER", "ENV", "LFO", "FX")

FORBIDDEN_OSC_TYPES = ("wavetable", "window", "audio in")


@dataclass
class ParamSpec:
    key: str          # "GROUP/entry/index" — stable across instances
    group: str
    entry: int
    index: int
    name: str
    vmin: float
    vmax: float
    vdef: float
    valtype: str
    handle: object    # SurgePyNamedParam — valid only for the owning synth


class SurgeParams:
    """Parameter table bound to one surgepy synth instance."""

    def __init__(self, synth: object) -> None:
        from surgepy import constants as sc  # type: ignore

        self.synth = synth
        self.specs: List[ParamSpec] = []
        self.by_key: Dict[str, ParamSpec] = {}
        for group in MUTABLE_GROUPS:
            cg = getattr(sc, f"cg_{group}")
            entries = synth.getControlGroup(cg).getEntries()
            for e_idx, entry in enumerate(entries):
                for p_idx, p in enumerate(entry.getParams()):
                    spec = ParamSpec(
                        key=f"{group}/{e_idx}/{p_idx}",
                        group=group,
                        entry=e_idx,
                        index=p_idx,
                        name=p.getName(),
                        vmin=float(synth.getParamMin(p)),
                        vmax=float(synth.getParamMax(p)),
                        vdef=float(synth.getParamDef(p)),
                        valtype=str(synth.getParamValType(p)),
                        handle=p,
                    )
                    self.specs.append(spec)
                    self.by_key[spec.key] = spec

    # ---- snapshot / apply -----------------------------------------------------

    def snapshot(self) -> Dict[str, float]:
        return {s.key: float(self.synth.getParamVal(s.handle)) for s in self.specs}

    def apply(self, values: Dict[str, float]) -> None:
        for key, val in values.items():
            spec = self.by_key.get(key)
            if spec is not None:
                self.synth.setParamVal(spec.handle, float(val))

    # ---- osc-type guard ---------------------------------------------------------

    def osc_type_specs(self) -> List[ParamSpec]:
        return [s for s in self.specs if s.group == "OSC" and s.name.endswith("Type")]

    def allowed_osc_type_values(self, spec: ParamSpec) -> List[int]:
        """Map each enum index to its display name by probing, excluding
        wavetable-data types and audio input. Restores the original value."""
        original = self.synth.getParamVal(spec.handle)
        allowed: List[int] = []
        for v in range(int(spec.vmin), int(spec.vmax) + 1):
            self.synth.setParamVal(spec.handle, float(v))
            display = str(self.synth.getParamDisplay(spec.handle)).lower()
            if not any(f in display for f in FORBIDDEN_OSC_TYPES):
                allowed.append(v)
        self.synth.setParamVal(spec.handle, original)
        return allowed


# ---- exemplar statistics --------------------------------------------------------

def exemplar_sigma_map(
    snapshots: Sequence[Dict[str, float]],
    specs: Sequence[ParamSpec],
    base_sigma: float = 0.08,
) -> Dict[str, float]:
    """Identity-core-aware mutation scales: parameters the anchor's exemplars agree
    on (low cross-exemplar std, range-normalized) mutate gently; parameters they
    disagree on are the exploration dimensions. Returns per-key sigma as a fraction
    of the param range."""
    sigmas: Dict[str, float] = {}
    n = len(snapshots)
    for spec in specs:
        span = spec.vmax - spec.vmin
        if span <= 0 or n < 2:
            sigmas[spec.key] = base_sigma
            continue
        vals = [snap.get(spec.key, spec.vdef) for snap in snapshots]
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        norm_std = min(1.0, (var ** 0.5) / span)
        sigmas[spec.key] = base_sigma * (0.25 + 2.0 * norm_std)
    return sigmas


# ---- mutation / crossover -------------------------------------------------------

@dataclass
class MutationConfig:
    groups_per_child: Tuple[int, int] = (1, 3)   # how many (group, entry) blocks to touch
    params_per_group: Tuple[int, int] = (2, 6)
    base_sigma: float = 0.08                      # fraction of param range (floats)
    int_step_prob: float = 0.35
    bool_flip_prob: float = 0.15
    osc_type_switch_prob: float = 0.06
    # relative weight per control group when picking blocks to mutate; e.g.
    # {"FX": 3.0} triples the chance FX blocks are touched (production-polish
    # emphasis for the GATE-2 loss families)
    group_weights: Optional[Dict[str, float]] = None


def mutate_values(
    params: SurgeParams,
    parent: Dict[str, float],
    rng: random.Random,
    config: MutationConfig,
    sigma_map: Optional[Dict[str, float]] = None,
    allowed_osc: Optional[Dict[str, List[int]]] = None,
) -> Dict[str, float]:
    """Produce a child's value-delta dict (applied on top of loadPatch(parent))."""
    blocks: Dict[Tuple[str, int], List[ParamSpec]] = {}
    for spec in params.specs:
        blocks.setdefault((spec.group, spec.entry), []).append(spec)
    block_keys = list(blocks.keys())

    n_blocks = min(rng.randint(*config.groups_per_child), len(block_keys))
    if config.group_weights:
        chosen: List[Tuple[str, int]] = []
        pool = list(block_keys)
        weights = [config.group_weights.get(k[0], 1.0) for k in pool]
        for _ in range(n_blocks):
            total = sum(weights)
            roll = rng.random() * total
            for j, w in enumerate(weights):
                roll -= w
                if roll <= 0:
                    chosen.append(pool.pop(j))
                    weights.pop(j)
                    break
        picked = chosen
    else:
        picked = rng.sample(block_keys, n_blocks)
    delta: Dict[str, float] = {}
    for block_key in picked:
        specs = blocks[block_key]
        n_params = min(rng.randint(*config.params_per_group), len(specs))
        for spec in rng.sample(specs, n_params):
            current = parent.get(spec.key, spec.vdef)
            span = spec.vmax - spec.vmin
            if spec.group == "OSC" and spec.name.endswith("Type"):
                if allowed_osc and rng.random() < config.osc_type_switch_prob:
                    choices = allowed_osc.get(spec.key, [])
                    if choices:
                        delta[spec.key] = float(rng.choice(choices))
                continue
            if spec.valtype == VT_FLOAT and span > 0:
                sigma = (sigma_map or {}).get(spec.key, config.base_sigma) * span
                delta[spec.key] = min(spec.vmax, max(spec.vmin, current + rng.gauss(0, sigma)))
            elif spec.valtype == VT_INT and span > 0:
                if rng.random() < config.int_step_prob:
                    step = rng.choice([-2, -1, 1, 2])
                    delta[spec.key] = float(min(spec.vmax, max(spec.vmin, current + step)))
            elif spec.valtype == VT_BOOL:
                if rng.random() < config.bool_flip_prob:
                    delta[spec.key] = 0.0 if current >= 0.5 else 1.0
    return delta


def crossover_values(
    params: SurgeParams,
    parent_a: Dict[str, float],
    parent_b: Dict[str, float],
    rng: random.Random,
) -> Dict[str, float]:
    """Block-wise crossover: child = parent A's patch (incl. its mod routing),
    with whole (group, entry) blocks of values taken from parent B."""
    blocks: Dict[Tuple[str, int], List[ParamSpec]] = {}
    for spec in params.specs:
        blocks.setdefault((spec.group, spec.entry), []).append(spec)
    delta: Dict[str, float] = {}
    for block_key, specs in blocks.items():
        if rng.random() < 0.35:  # take this block from B
            for spec in specs:
                b_val = parent_b.get(spec.key)
                if b_val is not None and b_val != parent_a.get(spec.key):
                    # osc-type guard applies to crossover too
                    if spec.group == "OSC" and spec.name.endswith("Type"):
                        continue
                    delta[spec.key] = b_val
    return delta
