"""Agent-based predator-prey simulation on a toroidal grid.

Two layers:
  * vegetation layer  -- every fertile cell either has vegetation or is regrowing
  * animal layer      -- every cell is EMPTY, PREY or PREDATOR (at most one animal)

One tick ("period") runs these phases in order:
  1. vegetation regrowth   eaten fertile cells regrow after `veg_regrow_time` ticks
  2. movement              prey move first (flee predators / seek vegetation), then
                           predators move (chase prey). One cell per tick (Moore).
  3. predation             any prey within `predator_kill_radius` of a predator is
                           killed (up to `predator_max_kills` per predator per tick)
  4. grazing               each prey eats up to `prey_food_need` vegetation cells
                           within `prey_eat_radius` (own cell first)
  5. hunger / ageing       prey that ate < K cells get +1 hunger, die at
                           `prey_starve_time` consecutive underfed ticks;
                           predators that did not kill get +1 hunger, die at
                           `predator_starve_time` ticks without eating
  6. reproduction          fed prey / predators that killed reproduce with a
                           probability into a free neighbouring cell

All phases are vectorised with NumPy. Conflicts (two agents wanting the same cell)
are resolved randomly, so results are reproducible for a given seed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

EMPTY, PREY, PREDATOR = 0, 1, 2

# Render codes for the combined view
CODE_SOIL, CODE_VEG, CODE_PREY, CODE_PREDATOR, CODE_BARREN = 0, 1, 2, 3, 4


def _meta(desc: str, lo: float, hi: float, step: float = 1, group: str = "") -> dict:
    return {"desc": desc, "min": lo, "max": hi, "step": step, "group": group}


@dataclass
class Params:
    # --- world ---
    width: int = field(default=200, metadata=_meta("Grid width (cells)", 10, 1000, 1, "World"))
    height: int = field(default=200, metadata=_meta("Grid height (cells)", 10, 1000, 1, "World"))
    seed: int = field(default=42, metadata=_meta("Random seed", 0, 2**31 - 1, 1, "World"))

    # --- vegetation ---
    fertile_fraction: float = field(default=1.0, metadata=_meta(
        "Fraction of cells that can ever grow vegetation", 0, 1, 0.01, "Vegetation"))
    init_vegetation: float = field(default=0.6, metadata=_meta(
        "Initial fraction of fertile cells covered", 0, 1, 0.01, "Vegetation"))
    veg_regrow_time: int = field(default=20, metadata=_meta(
        "Ticks for an eaten cell to regrow (R)", 1, 1000, 1, "Vegetation"))

    # --- prey ---
    init_prey: float = field(default=0.06, metadata=_meta(
        "Initial fraction of cells with prey", 0, 1, 0.005, "Prey"))
    prey_food_need: int = field(default=1, metadata=_meta(
        "Vegetation cells a prey must eat per tick to be fed (K)", 1, 25, 1, "Prey"))
    prey_eat_radius: int = field(default=1, metadata=_meta(
        "Radius prey can graze (0 = own cell only)", 0, 3, 1, "Prey"))
    prey_starve_time: int = field(default=6, metadata=_meta(
        "Consecutive underfed ticks before prey dies (S)", 1, 500, 1, "Prey"))
    prey_repro_prob: float = field(default=0.08, metadata=_meta(
        "Probability per tick a prey reproduces", 0, 1, 0.005, "Prey"))
    prey_repro_requires_fed: bool = field(default=True, metadata=_meta(
        "Prey only reproduce on ticks they were fed", 0, 1, 1, "Prey"))
    prey_max_age: int = field(default=0, metadata=_meta(
        "Max prey lifespan in ticks (0 = immortal)", 0, 10000, 1, "Prey"))
    prey_sense_radius: int = field(default=3, metadata=_meta(
        "How far prey sense predators / vegetation", 1, 10, 1, "Prey"))
    prey_directed_prob: float = field(default=0.7, metadata=_meta(
        "Probability a prey moves smartly (flee/seek food) vs randomly", 0, 1, 0.05, "Prey"))
    prey_fear: float = field(default=5.0, metadata=_meta(
        "Weight of predator avoidance vs food seeking", 0, 50, 0.5, "Prey"))

    # --- predators ---
    init_predator: float = field(default=0.01, metadata=_meta(
        "Initial fraction of cells with predators", 0, 1, 0.005, "Predator"))
    predator_starve_time: int = field(default=12, metadata=_meta(
        "Ticks without eating before predator dies (N)", 1, 500, 1, "Predator"))
    predator_kill_radius: int = field(default=1, metadata=_meta(
        "'Vicinity' radius in which a predator kills prey", 1, 3, 1, "Predator"))
    predator_max_kills: int = field(default=1, metadata=_meta(
        "Max prey killed per predator per tick (satiation)", 1, 25, 1, "Predator"))
    predator_repro_prob: float = field(default=0.3, metadata=_meta(
        "Probability a predator reproduces on a tick it ate", 0, 1, 0.01, "Predator"))
    predator_max_age: int = field(default=0, metadata=_meta(
        "Max predator lifespan in ticks (0 = immortal)", 0, 10000, 1, "Predator"))
    predator_sense_radius: int = field(default=4, metadata=_meta(
        "How far predators sense prey", 1, 10, 1, "Predator"))
    predator_directed_prob: float = field(default=0.7, metadata=_meta(
        "Probability a predator chases vs moves randomly", 0, 1, 0.05, "Predator"))

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "Params":
        p = cls()
        for f in fields(cls):
            if d and f.name in d and d[f.name] is not None and d[f.name] != "":
                v = d[f.name]
                if f.type in ("bool", bool):
                    v = v if isinstance(v, bool) else str(v).lower() in ("1", "true", "yes", "on")
                elif f.type in ("int", int):
                    v = int(float(v))
                else:
                    v = float(v)
                setattr(p, f.name, v)
        return p

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def schema(cls) -> list[dict[str, Any]]:
        out = []
        for f in fields(cls):
            m = dict(f.metadata)
            m.update(name=f.name, default=f.default,
                     type={"bool": "bool", "int": "int"}.get(str(f.type), "float"))
            out.append(m)
        return out


def moore_offsets(radius: int, include_center: bool = False) -> np.ndarray:
    offs = [(dy, dx) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)
            if include_center or (dy, dx) != (0, 0)]
    return np.array(offs, dtype=np.int64).reshape(-1, 2)


def box_sum(a: np.ndarray, r: int) -> np.ndarray:
    """Toroidal (2r+1)x(2r+1) neighbourhood sum, separable."""
    a = a.astype(np.float32)
    if r <= 0:
        return a
    s = a.copy()
    for k in range(1, r + 1):
        s += np.roll(a, k, 0) + np.roll(a, -k, 0)
    t = s.copy()
    for k in range(1, r + 1):
        t += np.roll(s, k, 1) + np.roll(s, -k, 1)
    return t


STEP_OFFSETS = moore_offsets(1)  # 8 possible moves

METRIC_KEYS = ["tick", "prey", "predators", "vegetation",
               "prey_births", "predator_births", "prey_eaten", "prey_starved",
               "predator_starved", "prey_old_age", "predator_old_age", "prey_fed_frac"]


class Simulation:
    def __init__(self, params: Params | None = None):
        self.p = params or Params()
        p = self.p
        self.rng = np.random.default_rng(p.seed)
        H, W = self.H, self.W = p.height, p.width
        rng = self.rng

        self.fertile = rng.random((H, W)) < p.fertile_fraction
        self.veg = self.fertile & (rng.random((H, W)) < p.init_vegetation)
        self.veg_timer = np.zeros((H, W), dtype=np.int32)
        regrowing = self.fertile & ~self.veg
        self.veg_timer[regrowing] = rng.integers(0, p.veg_regrow_time, regrowing.sum())

        self.animal = np.zeros((H, W), dtype=np.int8)
        self.hunger = np.zeros((H, W), dtype=np.int32)
        self.age = np.zeros((H, W), dtype=np.int32)
        n = H * W
        n_prey = min(n, int(round(p.init_prey * n)))
        n_pred = min(n - n_prey, int(round(p.init_predator * n)))
        cells = rng.permutation(n)
        self.animal.flat[cells[:n_prey]] = PREY
        self.animal.flat[cells[n_prey:n_prey + n_pred]] = PREDATOR
        # random initial age so max-age deaths don't all happen on the same tick
        if p.prey_max_age > 0:
            m = self.animal == PREY
            self.age[m] = rng.integers(0, p.prey_max_age, m.sum())
        if p.predator_max_age > 0:
            m = self.animal == PREDATOR
            self.age[m] = rng.integers(0, p.predator_max_age, m.sum())

        self.tick = 0
        self.history: list[dict[str, float]] = []
        self.history.append(self._metrics({}))

    # ------------------------------------------------------------------ helpers
    def _positions(self, species: int) -> tuple[np.ndarray, np.ndarray]:
        ys, xs = np.nonzero(self.animal == species)
        order = self.rng.permutation(len(ys))
        return ys[order], xs[order]

    def _wrap(self, ys, xs, dy, dx):
        return (ys + dy) % self.H, (xs + dx) % self.W

    def _shuffled(self, offs: np.ndarray) -> np.ndarray:
        return offs[self.rng.permutation(len(offs))]

    # ------------------------------------------------------------------ phases
    def _regrow(self):
        regrowing = self.fertile & ~self.veg
        self.veg_timer[regrowing] += 1
        grown = regrowing & (self.veg_timer >= self.p.veg_regrow_time)
        self.veg[grown] = True
        self.veg_timer[grown] = 0

    def _move(self, species: int, score_field: np.ndarray | None, directed_prob: float):
        ys, xs = self._positions(species)
        n = len(ys)
        if n == 0:
            return
        H, W = self.H, self.W
        ty = (ys[:, None] + STEP_OFFSETS[None, :, 0]) % H  # (n, 8)
        tx = (xs[:, None] + STEP_OFFSETS[None, :, 1]) % W
        free = self.animal[ty, tx] == EMPTY
        noise = self.rng.random((n, 8)).astype(np.float32)
        if score_field is not None and directed_prob > 0:
            directed = self.rng.random(n) < directed_prob
            score = np.where(directed[:, None], score_field[ty, tx] + 1e-3 * noise, noise)
        else:
            score = noise
        score = np.where(free, score, -np.inf)
        choice = np.argmax(score, axis=1)
        can_move = free.any(axis=1)
        idx = np.nonzero(can_move)[0]
        if len(idx) == 0:
            return
        sy, sx = ys[idx], xs[idx]
        dy_, dx_ = ty[idx, choice[idx]], tx[idx, choice[idx]]
        # resolve target conflicts: agents are already in random order -> first wins
        flat_t = dy_ * W + dx_
        _, first = np.unique(flat_t, return_index=True)
        sy, sx, dy_, dx_ = sy[first], sx[first], dy_[first], dx_[first]
        self.animal[dy_, dx_] = species
        self.hunger[dy_, dx_] = self.hunger[sy, sx]
        self.age[dy_, dx_] = self.age[sy, sx]
        self.animal[sy, sx] = EMPTY
        self.hunger[sy, sx] = 0
        self.age[sy, sx] = 0

    def _place_offspring(self, ys, xs, species) -> int:
        """Place one offspring per parent into a random free Moore neighbour."""
        if len(ys) == 0:
            return 0
        placed = np.zeros(len(ys), dtype=bool)
        for dy, dx in self._shuffled(STEP_OFFSETS):
            todo = ~placed
            if not todo.any():
                break
            ty, tx = self._wrap(ys, xs, dy, dx)
            ok = todo & (self.animal[ty, tx] == EMPTY)
            self.animal[ty[ok], tx[ok]] = species
            self.hunger[ty[ok], tx[ok]] = 0
            self.age[ty[ok], tx[ok]] = 0
            placed |= ok
        return int(placed.sum())

    # ------------------------------------------------------------------ step
    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self._step()

    def _step(self):
        p, rng = self.p, self.rng
        ev: dict[str, float] = {}

        # 1. vegetation
        self._regrow()

        # 2. movement: prey flee predators & seek vegetation, predators chase prey
        if p.prey_directed_prob > 0:
            pred_density = box_sum(self.animal == PREDATOR, p.prey_sense_radius)
            veg_density = box_sum(self.veg, max(p.prey_eat_radius, 1))
            prey_score = veg_density - p.prey_fear * pred_density
        else:
            prey_score = None
        self._move(PREY, prey_score, p.prey_directed_prob)
        prey_density = (box_sum(self.animal == PREY, p.predator_sense_radius)
                        if p.predator_directed_prob > 0 else None)
        self._move(PREDATOR, prey_density, p.predator_directed_prob)

        # 3. predation
        pys, pxs = self._positions(PREDATOR)
        kills = np.zeros(len(pys), dtype=np.int32)
        for dy, dx in self._shuffled(moore_offsets(p.predator_kill_radius)):
            ty, tx = self._wrap(pys, pxs, dy, dx)
            hit = (kills < p.predator_max_kills) & (self.animal[ty, tx] == PREY)
            self.animal[ty[hit], tx[hit]] = EMPTY
            self.hunger[ty[hit], tx[hit]] = 0
            self.age[ty[hit], tx[hit]] = 0
            kills += hit
        ev["prey_eaten"] = int(kills.sum())

        # 4. grazing (own cell first, then neighbours in random order)
        qys, qxs = self._positions(PREY)
        eaten = np.zeros(len(qys), dtype=np.int32)
        offs = self._shuffled(moore_offsets(p.prey_eat_radius)) if p.prey_eat_radius > 0 \
            else np.zeros((0, 2), dtype=np.int64)
        offs = np.vstack([[[0, 0]], offs])
        for dy, dx in offs:
            ty, tx = self._wrap(qys, qxs, dy, dx)
            bite = (eaten < p.prey_food_need) & self.veg[ty, tx]
            self.veg[ty[bite], tx[bite]] = False
            self.veg_timer[ty[bite], tx[bite]] = 0
            eaten += bite
        fed = eaten >= p.prey_food_need
        ev["prey_fed_frac"] = float(fed.mean()) if len(fed) else 0.0

        # 5. hunger & ageing
        self.hunger[qys, qxs] = np.where(fed, 0, self.hunger[qys, qxs] + 1)
        self.hunger[pys, pxs] = np.where(kills > 0, 0, self.hunger[pys, pxs] + 1)
        self.age[qys, qxs] += 1
        self.age[pys, pxs] += 1

        prey_starve = self.hunger[qys, qxs] >= p.prey_starve_time
        prey_old = (self.age[qys, qxs] >= p.prey_max_age) & ~prey_starve if p.prey_max_age > 0 \
            else np.zeros(len(qys), bool)
        pred_starve = self.hunger[pys, pxs] >= p.predator_starve_time
        pred_old = (self.age[pys, pxs] >= p.predator_max_age) & ~pred_starve if p.predator_max_age > 0 \
            else np.zeros(len(pys), bool)
        ev["prey_starved"] = int(prey_starve.sum())
        ev["prey_old_age"] = int(prey_old.sum())
        ev["predator_starved"] = int(pred_starve.sum())
        ev["predator_old_age"] = int(pred_old.sum())
        q_dead = prey_starve | prey_old
        p_dead = pred_starve | pred_old
        for ys, xs, dead in ((qys, qxs, q_dead), (pys, pxs, p_dead)):
            self.animal[ys[dead], xs[dead]] = EMPTY
            self.hunger[ys[dead], xs[dead]] = 0
            self.age[ys[dead], xs[dead]] = 0

        # 6. reproduction (predators first, then prey)
        pr = ~p_dead & (kills > 0) & (rng.random(len(pys)) < p.predator_repro_prob)
        ev["predator_births"] = self._place_offspring(pys[pr], pxs[pr], PREDATOR)
        eligible = ~q_dead & (fed if p.prey_repro_requires_fed else True)
        qr = eligible & (rng.random(len(qys)) < p.prey_repro_prob)
        ev["prey_births"] = self._place_offspring(qys[qr], qxs[qr], PREY)

        self.tick += 1
        self.history.append(self._metrics(ev))

    # ------------------------------------------------------------------ output
    def _metrics(self, ev: dict[str, float]) -> dict[str, float]:
        m: dict[str, float] = {k: 0.0 for k in METRIC_KEYS}
        m.update(ev)
        m["tick"] = self.tick
        m["prey"] = int((self.animal == PREY).sum())
        m["predators"] = int((self.animal == PREDATOR).sum())
        m["vegetation"] = int(self.veg.sum())
        return m

    def render_codes(self) -> np.ndarray:
        """uint8 grid: 0 soil, 1 vegetation, 2 prey, 3 predator, 4 barren."""
        out = np.where(self.fertile, CODE_SOIL, CODE_BARREN).astype(np.uint8)
        out[self.veg] = CODE_VEG
        out[self.animal == PREY] = CODE_PREY
        out[self.animal == PREDATOR] = CODE_PREDATOR
        return out

    @property
    def extinct(self) -> bool:
        last = self.history[-1]
        return last["prey"] == 0 and last["predators"] == 0
