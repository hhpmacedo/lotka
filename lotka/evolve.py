"""Genetic optimisation of simulation parameters ("DNA" = a parameter set).

    python -m lotka.evolve --objective coexistence --pop 48 --gens 30 --out runs/evolve/coexist
    python -m lotka.evolve --objective target_cycles --arg period=120 --arg cv=0.4
    python -m lotka.evolve --objective msy --arg species=predators
    python -m lotka.evolve --objective custom --arg "expr=mean_predators/cells*100 - 5*cv_prey"
    python -m lotka.evolve --objective qd --gens 40          # MAP-Elites behaviour map
    python -m lotka.evolve --resume runs/evolve/coexist --gens 50   # continue a run
    python -m lotka.evolve --list                            # objectives, genes, metrics

Lifespans default to prey 150 / predators 300 ticks during evolution (--immortal to disable, or
--fix/--gene prey_max_age=...), and coexistence only counts if both species are still being born
in the second half of the run (--allow-static to disable). This stops "immortal, non-breeding
predators" from scoring as perfect coexistence.

Every genome is evaluated on several seeds (common random numbers: the same seeds for every
genome) so luck can't be selected for; the top genomes are then re-validated on fresh seeds.

Outputs in --out:
  config.json        exact settings (used by --resume)
  generations.csv    per generation: best / mean / median fitness, diversity, archive coverage (qd)
  evals.csv          every evaluation: genome, fitness, aggregated metrics (a dataset for analysis)
  checkpoint.json    population + RNG state after each generation
  best.json          best genome, validated on hold-out seeds; loadable in the dashboard
  archive.json       (qd only) the behaviour map: best genome per cell
  evolution.png      fitness curves (or the qd map)
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import operator
import os
import signal
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .batch import summarize
from .sim import Params, Simulation

# --------------------------------------------------------------------------- genome
# name: (low, high). ints/bools are decoded from the continuous gene.
DEFAULT_GENES: dict[str, tuple[float, float]] = {
    "fertile_fraction": (0.2, 1.0),
    "veg_regrow_time": (2, 100),
    "prey_food_need": (1, 4),
    "prey_eat_radius": (0, 2),
    "prey_starve_time": (1, 40),
    "prey_repro_prob": (0.0, 0.6),
    "prey_repro_requires_fed": (0, 1),
    "prey_sense_radius": (1, 6),
    "prey_directed_prob": (0.0, 1.0),
    "prey_fear": (0.0, 20.0),
    "predator_starve_time": (1, 60),
    "predator_kill_radius": (1, 2),
    "predator_max_kills": (1, 6),
    "predator_repro_prob": (0.0, 1.0),
    "predator_sense_radius": (1, 8),
    "predator_directed_prob": (0.0, 1.0),
    "init_prey": (0.01, 0.2),
    "init_predator": (0.002, 0.05),
}

_TYPES = {f["name"]: f["type"] for f in Params.schema()}

# Applied during evolution unless evolved or fixed explicitly: without a finite lifespan a few
# immortal, non-breeding animals can "coexist" forever, which the optimiser happily exploits.
DEFAULT_FIXED = {"prey_max_age": 150, "predator_max_age": 300}


@dataclass
class Genome:
    names: list[str]
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def from_spec(cls, genes: dict[str, tuple[float, float]]):
        for k in genes:
            if k not in _TYPES:
                raise SystemExit(f"unknown gene {k}")
        names = list(genes)
        return cls(names, np.array([genes[k][0] for k in names], float),
                   np.array([genes[k][1] for k in names], float))

    def decode(self, u: np.ndarray) -> dict:
        out = {}
        for k, ui, lo, hi in zip(self.names, u, self.lo, self.hi):
            v = lo + float(ui) * (hi - lo)
            t = _TYPES[k]
            out[k] = bool(v >= 0.5) if t == "bool" else int(round(v)) if t == "int" else round(v, 5)
        return out

    def encode(self, params: dict) -> np.ndarray:
        u = []
        for k, lo, hi in zip(self.names, self.lo, self.hi):
            v = float(params[k])
            u.append(0.5 if hi == lo else min(max((v - lo) / (hi - lo), 0.0), 1.0))
        return np.array(u)


# --------------------------------------------------------------------------- evaluation
def evaluate(job: dict) -> dict:
    """Worker: run one (params, seed) simulation and return per-seed metrics."""
    p = Params.from_dict({**job["params"], "seed": job["seed"]})
    sim = Simulation(p)
    T = job["ticks"]
    for _ in range(T):
        sim.step()
        last = sim.history[-1]
        if last["prey"] == 0 or (last["predators"] == 0 and job["stop_on_pred"]):
            break
    m = summarize(sim.history, T)
    cells = p.width * p.height
    ext = [t for t in (m["prey_extinct_tick"], m["predator_extinct_tick"]) if t >= 0]
    alive = min(ext) if ext else T
    half = sim.history[len(sim.history) // 2:]
    m["yield_prey"] = float(np.mean([h["harvested_prey"] for h in half])) / cells * 1000
    m["yield_predators"] = float(np.mean([h["harvested_predators"] for h in half])) / cells * 1000
    m["prey_births_tail"] = int(sum(h["prey_births"] for h in half))
    m["predator_births_tail"] = int(sum(h["predator_births"] for h in half))
    coexist = m["outcome"] == "coexist"
    # a run only counts as coexistence if both populations keep reproducing (no immortal survivors)
    m["static"] = float(coexist and (m["prey_births_tail"] == 0 or m["predator_births_tail"] == 0))
    if job.get("require_turnover", True) and m["static"]:
        coexist = False
    m.update(cells=cells, ticks=T, survival=alive / T, coexist=float(coexist))
    m["resilience"] = (min(m["min_prey_after_transient"] / m["mean_prey"],
                           m["min_predators_after_transient"] / m["mean_predators"])
                       if m["coexist"] and m["mean_prey"] > 0 and m["mean_predators"] > 0 else 0.0)
    m["density_prey"] = m["mean_prey"] / cells
    m["density_predators"] = m["mean_predators"] / cells
    if job.get("lv_model") and m["coexist"]:
        from .lv import compare
        try:
            r = compare(sim, job["lv_model"], fit_start=min(200, T // 5))
            A, M = r["abm_stats"], r["model_stats"]
            err = 0.0
            for k in ("prey", "predators"):
                am, mm = A[k]["mean"] or 0, M[k]["mean"] or 0
                err += abs(am - mm) / max(am, 1)
                err += abs((A[k]["cv"] or 0) - (M[k]["cv"] or 0))
                ap, mp = A[k]["period"], M[k]["period"]
                err += abs(ap - mp) / ap if ap and mp else 1.0
            m["lv_error"] = err
        except Exception:  # noqa: BLE001
            m["lv_error"] = 10.0
    for k, v in list(m.items()):
        if isinstance(v, float) and not math.isfinite(v):
            m[k] = None
    return m


# --------------------------------------------------------------------------- objectives
def _viable(fn):
    """Coexisting runs score fn(m) >= 0; failed runs score survival - 1 in [-1, 0)."""
    def per_seed(m, args):
        if m["coexist"]:
            return fn(m, args)
        return (0.5 if m.get("static") else m["survival"]) - 1.0
    return per_seed


def _coexistence(ms, args):
    # static runs (one species no longer reproduces) only get half credit for survival
    surv = np.array([m["survival"] * (0.5 if m.get("static") and not m["coexist"] else 1.0) for m in ms])
    res = np.array([m["resilience"] for m in ms])
    return float(surv.mean() + 0.5 * res.mean() - 0.25 * surv.std())


def _target_cycles(m, args):
    P, cv = float(args.get("period", 150)), float(args.get("cv", 0.3))
    per = m["period_prey"]
    err = (abs(per - P) / P if per else 1.0) + abs((m["cv_prey"] or 0) - cv) / cv
    return 1.0 / (1.0 + err)


def _msy(m, args):
    sp = args.get("species", "predators")
    if sp == "total":
        return 100 * (m["density_prey"] + m["density_predators"])
    return 100 * m[f"density_{sp}"]


def _amplitude(m, args):
    return m["cv_prey"] or 0.0


def _yield(m, args):
    sp = args.get("species", "prey")
    y = m["yield_prey"] + m["yield_predators"] if sp == "total" else m[f"yield_{sp}"]
    return y - float(args.get("cv_weight", 0)) * (m["cv_prey"] or 0)


def _lv_match(m, args):
    return 1.0 / (1.0 + m.get("lv_error", 10.0))


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"min": min, "max": max, "abs": abs, "log": math.log, "sqrt": math.sqrt}


def safe_eval(expr: str, names: dict) -> float:
    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            v = names.get(n.id)
            if v is None:
                raise ValueError(f"metric {n.id} unavailable")
            return float(v)
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError(f"unsupported expression element: {ast.dump(n)}")
    return float(ev(ast.parse(expr, mode="eval")))


def _custom(m, args):
    try:
        return safe_eval(args["expr"], m)
    except (ValueError, ZeroDivisionError, OverflowError):
        return -1e6


@dataclass
class Objective:
    desc: str
    per_seed: Callable | None = None      # f(metrics, args) -> score; averaged over seeds
    aggregate: Callable | None = None     # f(list[metrics], args) -> score (overrides per_seed)
    stop_on_pred: bool = True             # stop a run early when predators die
    args: str = ""


OBJECTIVES: dict[str, Objective] = {
    "coexistence": Objective(
        "Robust coexistence: survival fraction of both species + 0.5 x resilience (lowest/mean "
        "population after transient), minus 0.25 x std of survival across seeds.",
        aggregate=_coexistence),
    "target_cycles": Objective(
        "Sustained cycles with a target prey period and amplitude (CV). Score 1/(1+error) "
        "when both species survive, survival-1 otherwise.", per_seed=_viable(_target_cycles),
        args="period=150 cv=0.3"),
    "msy": Objective(
        "Maximum sustainable yield: maximise mean density (% of grid) of a species while "
        "both coexist.", per_seed=_viable(_msy), args="species=predators|prey|total"),
    "amplitude": Objective(
        "Edge of collapse: maximise prey oscillation amplitude (CV) while both survive.",
        per_seed=_viable(_amplitude)),
    "lv_match": Objective(
        "Where does Lotka-Volterra theory hold? Minimise the gap (means, CV, period) between "
        "the ABM and a fitted LV model.", per_seed=_viable(_lv_match), args="model=chain|rm|logistic|classic"),
    "yield": Objective(
        "Sustainable harvest: maximise harvest per tick per 1000 cells while both species coexist "
        "(evolve the harvest_* control genes; optional penalty on prey CV).",
        per_seed=_viable(_yield), args="species=prey|predators|total cv_weight=0"),
    "custom": Objective(
        "Any expression over per-seed metrics, averaged over seeds (see --list for metric names).",
        per_seed=_custom, stop_on_pred=False, args='expr="mean_predators/cells*100 - 5*cv_prey"'),
    "qd": Objective(
        "Quality-diversity (MAP-Elites): fill a map of behaviours (default prey CV x prey period), "
        "keeping the most robust coexisting genome per cell.", aggregate=_coexistence,
        args="x=cv_prey:0:1.5:12 y=period_prey:20:400:12"),
}

METRIC_NAMES = ["survival", "coexist", "resilience", "mean_prey", "mean_predators", "mean_vegetation",
                "density_prey", "density_predators", "cv_prey", "cv_predators", "cv_vegetation",
                "period_prey", "period_predators", "predator_lag", "max_prey", "max_predators",
                "min_prey_after_transient", "min_predators_after_transient",
                "prey_extinct_tick", "predator_extinct_tick", "cells", "ticks", "lv_error",
                "prey_births_tail", "predator_births_tail", "static", "yield_prey", "yield_predators"]


def fitness_of(obj: Objective, ms: list[dict], args: dict) -> float:
    if obj.aggregate:
        return obj.aggregate(ms, args)
    return float(np.mean([obj.per_seed(m, args) for m in ms]))


def aggregate_metrics(ms: list[dict]) -> dict:
    out = {}
    for k in METRIC_NAMES:
        vals = [m.get(k) for m in ms if m.get(k) is not None]
        out[k] = round(float(np.mean(vals)), 5) if vals else None
    out["coexist_frac"] = out.pop("coexist")
    return out


# --------------------------------------------------------------------------- engine
class Evolver:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out = Path(cfg["out"])
        self.out.mkdir(parents=True, exist_ok=True)
        self.obj = OBJECTIVES[cfg["objective"]]
        self.args = cfg["args"]
        self.genome = Genome.from_spec({k: tuple(v) for k, v in cfg["genes"].items()})
        self.fixed = cfg["fixed"]
        self.seeds = list(range(cfg["seeds"]))
        self.rng = np.random.default_rng(cfg["rng_seed"])
        self.gen = 0
        self.n_evals = 0
        self.pop: np.ndarray | None = None
        self.fit: np.ndarray | None = None
        self.metrics: list[dict] = []
        self.archive: dict[str, dict] = {}   # qd

    # ---- evaluation of a batch of genomes (parallel over genome x seed)
    def evaluate_batch(self, U: np.ndarray, pool, seeds=None) -> tuple[np.ndarray, list[dict]]:
        seeds = seeds or self.seeds
        lv_model = self.args.get("model", "chain") if self.cfg["objective"] == "lv_match" else None
        jobs = []
        for u in U:
            params = {**self.fixed, **self.genome.decode(u)}
            for s in seeds:
                jobs.append({"params": params, "seed": s, "ticks": self.cfg["ticks"],
                             "stop_on_pred": self.obj.stop_on_pred, "lv_model": lv_model,
                             "require_turnover": self.cfg.get("require_turnover", True)})
        results = list(pool.map(evaluate, jobs, chunksize=1)) if pool else [evaluate(j) for j in jobs]
        self.n_evals += len(jobs)
        fits, aggs = [], []
        k = len(seeds)
        for i in range(len(U)):
            ms = results[i * k:(i + 1) * k]
            fits.append(fitness_of(self.obj, ms, self.args))
            aggs.append(aggregate_metrics(ms))
        return np.array(fits), aggs

    # ---- variation operators (on the unit hypercube)
    def tournament(self, k=3) -> np.ndarray:
        idx = self.rng.integers(0, len(self.pop), k)
        return self.pop[idx[np.argmax(self.fit[idx])]]

    def crossover(self, a, b):
        if self.rng.random() > self.cfg["cx_prob"]:
            return a.copy()
        alpha = 0.3  # BLX-alpha blend
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        d = hi - lo
        return self.rng.uniform(lo - alpha * d, hi + alpha * d)

    def mutate(self, u):
        n = len(u)
        rate = self.cfg["mut_rate"] if self.cfg["mut_rate"] > 0 else 1.5 / n
        mask = self.rng.random(n) < rate
        u = u + mask * self.rng.normal(0, self.cfg["mut_sigma"], n)
        return np.clip(u, 0, 1)

    # ---- qd helpers
    def _cell(self, agg: dict):
        key = []
        for ax in ("x", "y"):
            name, lo, hi, bins = self.qd_axes[ax]
            v = agg.get(name)
            if v is None or not agg.get("coexist_frac"):
                return None
            key.append(int(min(max((v - lo) / (hi - lo), 0), 0.999999) * bins))
        return f"{key[0]},{key[1]}"

    def _qd_insert(self, u, f, agg):
        c = self._cell(agg)
        if c is None:
            return False
        if c not in self.archive or f > self.archive[c]["fitness"]:
            self.archive[c] = {"u": u.tolist(), "fitness": float(f), "metrics": agg,
                               "params": {**self.fixed, **self.genome.decode(u)}}
            return True
        return False

    @property
    def qd_axes(self):
        axes = {}
        for ax, default in (("x", "cv_prey:0:1.5:12"), ("y", "period_prey:20:400:12")):
            name, lo, hi, bins = self.args.get(ax, default).split(":")
            axes[ax] = (name, float(lo), float(hi), int(bins))
        return axes

    # ---- logging
    def _log_evals(self, U, F, A):
        path = self.out / "evals.csv"
        new = not path.exists()
        with path.open("a", newline="") as f:
            w = csv.writer(f)
            mk = list(A[0].keys())
            if new:
                w.writerow(["gen", "fitness"] + self.genome.names + mk)
            for u, fi, a in zip(U, F, A):
                d = self.genome.decode(u)
                w.writerow([self.gen, round(float(fi), 6)] + [d[k] for k in self.genome.names] + [a[k] for k in mk])

    def _log_gen(self, t0):
        path = self.out / "generations.csv"
        new = not path.exists()
        F = self.fit
        row = {"gen": self.gen, "evals": self.n_evals, "best": round(float(F.max()), 6),
               "mean": round(float(F.mean()), 6), "median": round(float(np.median(F)), 6),
               "diversity": round(float(self.pop.std(axis=0).mean()), 4),
               "coexist_frac": round(float(np.mean([m["coexist_frac"] or 0 for m in self.metrics])), 3),
               "archive_cells": len(self.archive), "seconds": round(time.time() - t0, 1)}
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
        extra = f" cells={len(self.archive)}" if self.cfg["objective"] == "qd" else ""
        print(f"gen {self.gen:3d}  best {row['best']:.4f}  mean {row['mean']:.4f}  "
              f"coexist {row['coexist_frac']:.2f}  div {row['diversity']:.3f}{extra}  ({row['seconds']}s)", flush=True)

    def status(self, state: str, **extra):
        st = {"state": state, "pid": os.getpid(), "gen": self.gen, "gens": self.cfg["gens"],
              "evals": self.n_evals, "updated": time.time(), **extra}
        path = self.out / "status.json"
        if path.exists():
            try:
                st["started"] = json.loads(path.read_text()).get("started", time.time())
            except ValueError:
                st["started"] = time.time()
        else:
            st["started"] = time.time()
        if self.fit is not None and len(self.fit):
            st["best"] = float(np.max(self.fit))
        tmp = self.out / "status.json.tmp"
        tmp.write_text(json.dumps(st))
        tmp.replace(path)

    def checkpoint(self):
        state = {"gen": self.gen, "n_evals": self.n_evals, "pop": self.pop.tolist(), "fit": self.fit.tolist(),
                 "metrics": self.metrics, "rng": self.rng.bit_generator.state, "archive": self.archive}
        tmp = self.out / "checkpoint.json.tmp"
        tmp.write_text(json.dumps(state))
        tmp.replace(self.out / "checkpoint.json")
        self.status("running")
        if self.cfg["objective"] == "qd":
            (self.out / "archive.json").write_text(json.dumps(
                {"axes": self.qd_axes, "cells": self.archive}, indent=1))

    def restore(self):
        st = json.loads((self.out / "checkpoint.json").read_text())
        self.gen, self.n_evals = st["gen"], st["n_evals"]
        self.pop, self.fit = np.array(st["pop"]), np.array(st["fit"])
        self.metrics, self.archive = st["metrics"], st["archive"]
        self.rng.bit_generator.state = st["rng"]

    # ---- main loop
    def run(self, pool):
        cfg = self.cfg
        n = cfg["pop"]
        if self.pop is None:
            t0 = time.time()
            U = self.rng.random((n, len(self.genome.names)))
            # seed the population with the default parameters and the presets
            from .presets import PRESETS
            base = Params().to_dict()
            starts = [base] + [{**base, **pr["params"]} for pr in PRESETS.values()]
            for i, s in enumerate(starts[: n // 4]):
                U[i] = self.genome.encode(s)
            F, A = self.evaluate_batch(U, pool)
            self.pop, self.fit, self.metrics = U, F, A
            if cfg["objective"] == "qd":
                for u, f, a in zip(U, F, A):
                    self._qd_insert(u, f, a)
            self._log_evals(U, F, A)
            self._log_gen(t0)
            self.checkpoint()
        while self.gen < cfg["gens"]:
            t0 = time.time()
            if cfg["objective"] == "qd" and self.archive:
                elites = list(self.archive.values())
                kids = []
                for _ in range(n):
                    a = np.array(elites[self.rng.integers(len(elites))]["u"])
                    b = np.array(elites[self.rng.integers(len(elites))]["u"])
                    kids.append(self.mutate(np.clip(self.crossover(a, b), 0, 1)))
                U = np.array(kids)
                F, A = self.evaluate_batch(U, pool)
                for u, f, a in zip(U, F, A):
                    self._qd_insert(u, f, a)
                self.pop, self.fit, self.metrics = U, F, A
            else:
                order = np.argsort(-self.fit)
                n_elite = cfg["elite"]
                kids = [self.pop[i].copy() for i in order[:n_elite]]
                kid_fit = [self.fit[i] for i in order[:n_elite]]
                kid_met = [self.metrics[i] for i in order[:n_elite]]
                new = []
                while len(kids) + len(new) < n:
                    child = self.mutate(np.clip(self.crossover(self.tournament(), self.tournament()), 0, 1))
                    new.append(child)
                U = np.array(new)
                F, A = self.evaluate_batch(U, pool)
                self.pop = np.vstack([np.array(kids), U])
                self.fit = np.concatenate([kid_fit, F])
                self.metrics = kid_met + A
            self.gen += 1
            self._log_evals(U, F, A)
            self._log_gen(t0)
            self.checkpoint()

    def finalize(self, pool):
        """Re-validate the top genomes on hold-out seeds and write best.json."""
        if self.cfg["objective"] == "qd":
            cands = sorted(self.archive.values(), key=lambda e: -e["fitness"])[: self.cfg["validate_top"]]
            U = np.array([c["u"] for c in cands]) if cands else self.pop[:1]
        else:
            uniq, seen = [], set()
            for i in np.argsort(-self.fit):
                key = json.dumps(self.genome.decode(self.pop[i]), sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    uniq.append(self.pop[i])
                if len(uniq) >= self.cfg["validate_top"]:
                    break
            U = np.array(uniq)
        hold = list(range(10_000, 10_000 + self.cfg["validate_seeds"]))
        F, A = self.evaluate_batch(U, pool, seeds=hold)
        order = np.argsort(-F)
        ranking = []
        for r, i in enumerate(order):
            ranking.append({"rank": r + 1, "validated_fitness": round(float(F[i]), 6),
                            "params": {**self.fixed, **self.genome.decode(U[i])}, "metrics": A[i]})
        best = {"objective": self.cfg["objective"], "args": self.args, "desc": self.obj.desc,
                "validation_seeds": hold, "train_seeds": self.seeds, "generations": self.gen,
                "evaluations": self.n_evals, **ranking[0], "top": ranking}
        (self.out / "best.json").write_text(json.dumps(best, indent=2))
        print(f"\nvalidated on {len(hold)} hold-out seeds -> best fitness {ranking[0]['validated_fitness']:.4f}")
        bm = ranking[0]["metrics"]
        print(f"  coexist {bm['coexist_frac']}, mean prey {bm['mean_prey']}, mean predators {bm['mean_predators']}, "
              f"cv prey {bm['cv_prey']}, period {bm['period_prey']}")
        diff = {k: v for k, v in ranking[0]["params"].items() if k in self.genome.names}
        print("  genome:", json.dumps(diff))
        self.plot()

    def plot(self):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return
        rows = list(csv.DictReader((self.out / "generations.csv").open()))
        g = [int(r["gen"]) for r in rows]
        if self.cfg["objective"] == "qd":
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
            (xn, xl, xh, xb), (yn, yl, yh, yb) = self.qd_axes["x"], self.qd_axes["y"]
            M = np.full((yb, xb), np.nan)
            for k, e in self.archive.items():
                i, j = map(int, k.split(","))
                M[j, i] = e["fitness"]
            im = a2.imshow(M, origin="lower", extent=[xl, xh, yl, yh], aspect="auto", cmap="viridis")
            fig.colorbar(im, ax=a2, label="fitness (robust coexistence)")
            a2.set_xlabel(xn)
            a2.set_ylabel(yn)
            a2.set_title(f"MAP-Elites archive: {len(self.archive)}/{xb * yb} cells")
            a1.plot(g, [int(r["archive_cells"]) for r in rows], label="archive cells")
            a1.set_xlabel("generation")
            a1.legend()
        else:
            fig, a1 = plt.subplots(figsize=(9, 4.5))
            a1.plot(g, [float(r["best"]) for r in rows], label="best")
            a1.plot(g, [float(r["mean"]) for r in rows], label="mean")
            a1.plot(g, [float(r["median"]) for r in rows], label="median", ls="--")
            a1.set_xlabel("generation")
            a1.set_ylabel("fitness")
            a1.set_title(f"objective: {self.cfg['objective']} {self.args}")
            a1.legend()
        fig.tight_layout()
        fig.savefig(self.out / "evolution.png", dpi=110)
        plt.close(fig)


def _init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evolve simulation parameters",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--objective", choices=list(OBJECTIVES), default="coexistence")
    ap.add_argument("--arg", action="append", default=[], metavar="KEY=VALUE", help="objective argument")
    ap.add_argument("--pop", type=int, default=48)
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--seeds", type=int, default=3, help="training seeds per genome")
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--size", type=int, default=100, help="grid width = height during evolution")
    ap.add_argument("--genes", default="", help="comma list restricting which genes evolve (default: all)")
    ap.add_argument("--gene", action="append", default=[], metavar="NAME=LO:HI", help="add/override gene bounds")
    ap.add_argument("--fix", action="append", default=[], metavar="PARAM=VALUE", help="fixed non-evolved parameter")
    ap.add_argument("--immortal", action="store_true",
                    help="don't apply the default lifespans (prey 150, predators 300 ticks)")
    ap.add_argument("--allow-static", action="store_true",
                    help="count runs as coexistence even if a species stopped reproducing")
    ap.add_argument("--elite", type=int, default=2)
    ap.add_argument("--cx-prob", type=float, default=0.8)
    ap.add_argument("--mut-rate", type=float, default=0.0, help="per-gene mutation prob (0 = 1.5/n_genes)")
    ap.add_argument("--mut-sigma", type=float, default=0.12)
    ap.add_argument("--validate-top", type=int, default=5)
    ap.add_argument("--validate-seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--rng-seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--resume", default=None, help="continue a run directory (other settings from its config)")
    ap.add_argument("--list", action="store_true", help="list objectives, genes and metrics")
    a = ap.parse_args(argv)

    if a.list:
        print("Objectives:")
        for k, o in OBJECTIVES.items():
            print(f"  {k:14s} {o.desc}" + (f"\n  {'':14s} args: {o.args}" if o.args else ""))
        print("\nGenes (default bounds):")
        for k, (lo, hi) in DEFAULT_GENES.items():
            print(f"  {k:24s} {lo} .. {hi}  ({_TYPES[k]})")
        print("\nMetrics for custom expressions (per seed):\n  " + ", ".join(METRIC_NAMES))
        return

    if a.resume:
        out = Path(a.resume)
        cfg = json.loads((out / "config.json").read_text())
        cfg["gens"] = max(cfg["gens"], a.gens)
        cfg["workers"] = a.workers
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        ev = Evolver(cfg)
        ev.restore()
        print(f"resuming {out} at generation {ev.gen} -> {cfg['gens']}")
    else:
        genes = dict(DEFAULT_GENES)
        if a.genes:
            keep = [g.strip() for g in a.genes.split(",") if g.strip()]
            genes = {k: DEFAULT_GENES.get(k, (0, 1)) for k in keep}
        for g in a.gene:
            k, rng = g.split("=", 1)
            lo, hi = rng.split(":")
            genes[k] = (float(lo), float(hi))
        fixed = {"width": a.size, "height": a.size}
        if not a.immortal:
            fixed.update({k: v for k, v in DEFAULT_FIXED.items() if k not in genes})
        for f in a.fix:
            k, v = f.split("=", 1)
            fixed[k] = v
            genes.pop(k, None)
        fixed = {k: v for k, v in Params.from_dict(fixed).to_dict().items() if k in fixed}
        args = {}
        for s in a.arg:
            k, v = s.split("=", 1)
            args[k] = v
        if a.objective == "custom" and "expr" not in args:
            raise SystemExit('custom objective needs --arg "expr=..."')
        out = Path(a.out or f"runs/evolve/{a.objective}_{time.strftime('%Y%m%d_%H%M%S')}")
        if (out / "checkpoint.json").exists():
            raise SystemExit(f"{out} already has a run; use --resume {out} or another --out")
        cfg = {"objective": a.objective, "args": args, "pop": a.pop, "gens": a.gens, "seeds": a.seeds,
               "ticks": a.ticks, "genes": genes, "fixed": fixed, "elite": a.elite, "cx_prob": a.cx_prob,
               "mut_rate": a.mut_rate, "mut_sigma": a.mut_sigma, "validate_top": a.validate_top,
               "validate_seeds": a.validate_seeds, "rng_seed": a.rng_seed, "out": str(out),
               "workers": a.workers, "require_turnover": not a.allow_static}
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(cfg, indent=2))
        ev = Evolver(cfg)
        print(f"objective={a.objective} {args}  pop={a.pop} gens={a.gens} seeds={a.seeds} ticks={a.ticks} "
              f"grid={a.size}x{a.size} genes={len(genes)} workers={a.workers} -> {out}")

    def _term(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _term)
    t0 = time.time()
    ev.status("running", phase="evolving")
    pool = ProcessPoolExecutor(a.workers, initializer=_init_worker) if a.workers > 1 else None
    try:
        ev.run(pool)
        ev.status("running", phase="validating")
        ev.finalize(pool)
        if pool:
            pool.shutdown()
    except KeyboardInterrupt:
        if pool:
            procs = list((pool._processes or {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for pr in procs:
                pr.terminate()
        ev.status("stopped")
        print(f"\nstopped at generation {ev.gen}; continue with --resume {out}", flush=True)
        return
    except Exception as e:
        if pool:
            pool.shutdown(wait=False, cancel_futures=True)
        ev.status("failed", error=str(e))
        raise
    ev.status("finished")
    print(f"done: {ev.n_evals} simulations in {time.time() - t0:.0f}s -> {out}/best.json")


if __name__ == "__main__":
    main()
