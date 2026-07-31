"""Surge parameter layer for generation: enumerate, snapshot, mutate, crossover.

Ground rules (empirically established):
- Value accessors (getParamVal/Min/Max/Def/ValType, setParamVal, getParamDisplay)
  take the SurgePyNamedParam handle; handles belong to ONE synth instance and are
  re-enumerated per instance. Enumeration order is deterministic for a build, so
  a (group, entry, index) key is stable across instances.
- The ~766 control-group params do NOT carry modulation routings, step-seq/MSEG
  data, or embedded wavetables. Children therefore derive by loadPatch(parent)
  followed by setParamVal deltas — never by replaying values onto an init patch.
  (Since 2026-07-30, deltas may ALSO carry curated mod-matrix depths via
  "MOD/<source>:<param key>" keys — see ACID_MOD_ROUTES.)
- Osc-type mutations never select wavetable-data types (Wavetable/Window — their
  sample data embeds into saved patches; provenance policy) nor Audio Input
  (silent without input).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# surgepy getParamValType returns strings: 'int' | 'bool' | 'float'
VT_INT = "int"
VT_BOOL = "bool"
VT_FLOAT = "float"

# Mutable control groups. GLOBAL is excluded AS A GROUP (scene mode, MIDI
# config, master levels — logistics), but the scene-level timbre/expression
# params hiding inside it are allowlisted by exact name below (acid program,
# 2026-07-30: portamento + mono play mode ARE the 303 slide contract).
MUTABLE_GROUPS = ("OSC", "MIX", "FILTER", "ENV", "LFO", "FX")

# cg_GLOBAL params that are timbre/expression, not logistics. Matched by EXACT
# getName() per entry; entry 0 (patch globals: Global Volume, Scene Mode,
# Split Point, ...) contains none of these names, so only the per-scene
# entries contribute. Play Mode's range is clamped to exclude Latch (5) —
# a latched voice never releases in headless probe renders.
MUTABLE_GLOBAL_NAMES = (
    "Portamento",            # slide glide time (log2 s; min = instant)
    "Play Mode",             # poly | mono | mono-st | mono(fp) | mono-st(fp)
    "Velocity > VCA Gain",   # velocity→amp accent depth (dB)
    "Feedback",              # scene filter-block feedback
    "Highpass",              # scene highpass
    "Waveshaper Type",       # acid grit
    "Waveshaper Drive",
)
_PLAY_MODE_MAX = 4.0  # exclude Latch

FORBIDDEN_OSC_TYPES = ("wavetable", "window", "audio in")

# ---- curated mod-matrix routings (delta channel "MOD/<source>:<param key>") ----
#
# The mod matrix is patch-stream state surgepy CAN write (setModDepth01) but the
# value snapshot never carried. Rather than opening every (source × target)
# pair, evolution gets a curated acid/motion set; depths ride the SAME delta
# dict as values, so render workers, survivor writes, and crossover inherit
# them with zero extra plumbing. Depth domain is Surge's normalized [-1, 1].
MOD_KEY_PREFIX = "MOD/"

@dataclass(frozen=True)
class ModRoute:
    source: str        # surgepy.constants name minus the "ms_" prefix
    target_name: str   # exact ParamSpec.name (first match wins = scene A)
    lo: float
    hi: float

ACID_MOD_ROUTES: Tuple[ModRoute, ...] = (
    ModRoute("velocity", "Filter 1 Cutoff", 0.0, 0.6),   # the 303 accent
    ModRoute("lfo1", "Filter 1 Cutoff", -0.6, 0.6),      # wobble depth
)


def mod_key(source: str, param_key: str) -> str:
    return f"{MOD_KEY_PREFIX}{source}:{param_key}"


def parse_mod_key(key: str) -> Optional[Tuple[str, str]]:
    """"MOD/velocity:FILTER/0/2" -> ("velocity", "FILTER/0/2"); None if not a MOD key."""
    if not key.startswith(MOD_KEY_PREFIX):
        return None
    source, sep, param_key = key[len(MOD_KEY_PREFIX):].partition(":")
    if not sep or not source or not param_key:
        return None
    return source, param_key


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
                    self._add_spec(group, e_idx, p_idx, p)

        # cg_GLOBAL: allowlisted scene-level timbre/expression params only.
        cg = synth.getControlGroup(sc.cg_GLOBAL)
        for e_idx, entry in enumerate(cg.getEntries()):
            for p_idx, p in enumerate(entry.getParams()):
                if p.getName() in MUTABLE_GLOBAL_NAMES:
                    self._add_spec("GLOBAL", e_idx, p_idx, p)

    def _add_spec(self, group: str, e_idx: int, p_idx: int, p: object) -> None:
        name = str(p.getName())
        vmax = float(self.synth.getParamMax(p))
        if group == "GLOBAL" and name == "Play Mode":
            vmax = min(vmax, _PLAY_MODE_MAX)
        spec = ParamSpec(
            key=f"{group}/{e_idx}/{p_idx}",
            group=group,
            entry=e_idx,
            index=p_idx,
            name=name,
            vmin=float(self.synth.getParamMin(p)),
            vmax=vmax,
            vdef=float(self.synth.getParamDef(p)),
            valtype=str(self.synth.getParamValType(p)),
            handle=p,
        )
        self.specs.append(spec)
        self.by_key[spec.key] = spec

    # ---- snapshot / apply -----------------------------------------------------

    def snapshot(self) -> Dict[str, float]:
        """Control-group values only. Mod-matrix depths are NOT snapshotted —
        children inherit the parent's routings via loadPatch and the delta's
        MOD/ keys override on top (see ACID_MOD_ROUTES)."""
        return {s.key: float(self.synth.getParamVal(s.handle)) for s in self.specs}

    def apply(self, values: Dict[str, float]) -> None:
        for key, val in values.items():
            if key.startswith(MOD_KEY_PREFIX):
                self._apply_mod(key, float(val))
                continue
            spec = self.by_key.get(key)
            if spec is not None:
                self.synth.setParamVal(spec.handle, float(val))

    def _apply_mod(self, key: str, depth: float) -> None:
        parsed = parse_mod_key(key)
        if parsed is None:
            return
        source, param_key = parsed
        spec = self.by_key.get(param_key)
        if spec is None:
            return
        try:
            from surgepy import constants as sc  # type: ignore

            ms_id = getattr(sc, f"ms_{source}", None)
            if ms_id is None:
                return
            mod_source = self.synth.getModSource(ms_id)
            if not self.synth.isValidModulation(spec.handle, mod_source):
                return
            self.synth.setModDepth01(spec.handle, mod_source, float(depth))
        except Exception:
            return  # a bad routing must never kill a render/save

    def find_first_by_name(self, name: str) -> Optional[ParamSpec]:
        """First spec (enumeration order == scene A first) with this exact name."""
        for spec in self.specs:
            if spec.name == name:
                return spec
        return None

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
    # curated mod-matrix routings (acid program): per child, chance of one
    # gaussian step on one route's depth. Routes default to ACID_MOD_ROUTES;
    # set mod_routes=() to disable entirely.
    mod_route_prob: float = 0.25
    mod_route_sigma: float = 0.15
    mod_routes: Tuple[ModRoute, ...] = field(default_factory=lambda: ACID_MOD_ROUTES)


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

    # Mod-matrix gene: one gaussian step on one curated route's depth. The
    # walk starts from the parent's DELTA depth (0.0 when the route was never
    # touched) — seeds' own routings still sound (loadPatch carries them);
    # this only steers the evolved override.
    if config.mod_routes and rng.random() < config.mod_route_prob:
        route = rng.choice(list(config.mod_routes))
        spec = params.find_first_by_name(route.target_name)
        if spec is not None:
            key = mod_key(route.source, spec.key)
            current = parent.get(key, 0.0)
            depth = min(route.hi, max(route.lo, current + rng.gauss(0, config.mod_route_sigma)))
            delta[key] = depth
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
