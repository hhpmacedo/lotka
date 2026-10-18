"""Lotka-Volterra family of models, fitted to (or derived from) the agent-based simulation.

Densities are per cell (count / number of cells); time unit = one tick.

Fitted continuous-time models (rates estimated from the ABM's per-tick event counts):

  classic   dx/dt = a x            - F(x) y        F(x) = b x          (Lotka 1925, Volterra 1926)
            dy/dt = c F(x) y - g y
  logistic  dx/dt = r x (1 - x/K)  - F(x) y        F(x) = b x          (prey self-limitation)
  rm        dx/dt = r x (1 - x/K)  - F(x) y        F(x) = a x/(1+a h x) (Rosenzweig-MacArthur,
            dy/dt = c F(x) y - g y                                       Holling type II satiation)

Rule-derived model (no fitting):

  meanfield  discrete-time map of the *exact* ABM rules under the well-mixed assumption
             (vegetation + hunger-structured prey and predators). The gap between the ABM
             and this model is the effect of space (clustering, local depletion, waves).

Rate fitting uses the event counters recorded every tick:
  predation    prey_eaten         ~ F(x) y          -> b (or a, h)
  conversion   predator_births    ~ c * prey_eaten  -> c
  pred. death  predator_starved   ~ g y             -> g
  prey growth  births - starved   ~ a x  or r x (1 - x/K)
"""

from __future__ import annotations

from collections import deque
from math import comb, inf, isfinite, pi

import numpy as np

from .sim import Params

MODELS = {
    "classic": "Classic Lotka-Volterra",
    "logistic": "Logistic-prey Lotka-Volterra",
    "rm": "Rosenzweig-MacArthur (type II)",
    "chain": "Vegetation-prey-predator chain",
    "meanfield": "Mean-field of ABM rules",
}

EQUATIONS = {
    "classic": ["dx/dt = a·x − b·x·y", "dy/dt = c·b·x·y − g·y"],
    "logistic": ["dx/dt = r·x·(1 − x/K) − b·x·y", "dy/dt = c·b·x·y − g·y"],
    "rm": ["dx/dt = r·x·(1 − x/K) − a·x·y/(1 + a·h·x)", "dy/dt = c·a·x·y/(1 + a·h·x) − g·y"],
    "chain": ["dv/dt = ρ·(V − v) − e·v·x", "dx/dt = ε·e·v·x − μ·x − b·x·y", "dy/dt = c·b·x·y − g·y"],
    "meanfield": ["well-mixed map of the ABM rules:",
                  "kill P = 1−(1−x)^m, fed P = P(Bin(n,v) ≥ K), hunger-class ageing, delayed regrowth"],
}


# --------------------------------------------------------------------------- data
def series(history: list[dict], cells: int) -> dict[str, np.ndarray]:
    def col(k):
        return np.array([h.get(k, 0) for h in history], dtype=float)
    return {
        "tick": col("tick"),
        "x": col("prey") / cells, "y": col("predators") / cells, "v": col("vegetation") / cells,
        "eaten": col("prey_eaten") / cells,
        "prey_births": col("prey_births") / cells,
        "prey_deaths": (col("prey_starved") + col("prey_old_age")) / cells,
        "pred_births": col("predator_births") / cells,
        "pred_deaths": (col("predator_starved") + col("predator_old_age")) / cells,
        "veg_eaten": col("veg_eaten") / cells,
        "veg_regrown": col("veg_regrown") / cells,
    }


def _r2(obs, pred):
    obs, pred = np.asarray(obs), np.asarray(pred)
    ss = ((obs - obs.mean()) ** 2).sum()
    return float(1 - ((obs - pred) ** 2).sum() / ss) if ss > 0 else float("nan")


def _ls1(target, feature):
    """least squares through origin: target ~ k * feature"""
    d = (feature ** 2).sum()
    return float((target * feature).sum() / d) if d > 0 else 0.0


# --------------------------------------------------------------------------- fitting
def fit_rates(history: list[dict], cells: int, model: str, start: int = 0,
              fertile: float = 1.0) -> dict:
    """Estimate model rates from ABM events. Events of tick t use the state at t-1."""
    s = series(history, cells)
    sl = slice(max(start, 0), None)
    x, y = s["x"][:-1][sl], s["y"][:-1][sl]          # pre-tick state
    E = s["eaten"][1:][sl]
    G = (s["prey_births"] - s["prey_deaths"])[1:][sl]
    By, Dy = s["pred_births"][1:][sl], s["pred_deaths"][1:][sl]
    if len(x) < 5:
        raise ValueError("need at least 5 ticks of history to fit")

    p: dict[str, float] = {}
    r2: dict[str, float] = {}
    if model == "chain":
        v = s["v"][:-1][sl]
        Ve, Vr = s["veg_eaten"][1:][sl], s["veg_regrown"][1:][sl]
        Bx, Dx = s["prey_births"][1:][sl], s["prey_deaths"][1:][sl]
        p["V"] = float(fertile)
        p["rho"] = _ls1(Vr, fertile - v)
        r2["veg_regrowth"] = _r2(Vr, p["rho"] * (fertile - v))
        p["e"] = _ls1(Ve, x * v)
        r2["grazing"] = _r2(Ve, p["e"] * x * v)
        p["eps"] = _ls1(Bx, Ve)
        r2["prey_births"] = _r2(Bx, p["eps"] * Ve)
        p["mu"] = _ls1(Dx, x)
        r2["prey_deaths"] = _r2(Dx, p["mu"] * x)
    # prey growth (without predation)
    elif model == "classic":
        p["a"] = _ls1(G, x)
        p["K"] = inf
        r2["prey_growth"] = _r2(G, p["a"] * x)
    else:
        A = np.column_stack([x, -x * x])
        (r, rk), *_ = np.linalg.lstsq(A, G, rcond=None)
        p["r"] = float(r)
        p["K"] = float(r / rk) if rk > 0 and r > 0 else inf
        r2["prey_growth"] = _r2(G, A @ np.array([r, rk]))

    # functional response F(x): kills per predator per tick
    xy = x * y
    if model in ("classic", "logistic", "chain"):
        p["b"] = _ls1(E, xy)
        r2["predation"] = _r2(E, p["b"] * xy)
    else:
        m = (E > 0) & (y > 0)
        # linearisation  x*y = E/a + h*E*x  -> start values, then Gauss-Newton
        if m.sum() >= 3:
            (u, h), *_ = np.linalg.lstsq(np.column_stack([E[m], E[m] * x[m]]), xy[m], rcond=None)
            a = 1 / u if u > 0 else _ls1(E, xy)
            h = max(float(h), 0.0)
        else:
            a, h = _ls1(E, xy), 0.0
        for _ in range(50):
            den = 1 + a * h * x
            f = a * xy / den
            res = E - f
            J = np.column_stack([xy / den ** 2, -a * a * x * xy / den ** 2])
            step, *_ = np.linalg.lstsq(J, res, rcond=None)
            a, h = max(a + step[0], 1e-9), max(h + step[1], 0.0)
            if np.abs(step).max() < 1e-10:
                break
        p["a"], p["h"] = float(a), float(h)
        r2["predation"] = _r2(E, a * xy / (1 + a * h * x))

    p["c"] = _ls1(By, E)
    r2["conversion"] = _r2(By, p["c"] * E)
    p["g"] = _ls1(Dy, y)
    r2["predator_death"] = _r2(Dy, p["g"] * y)
    return {"params": p, "rate_r2": r2}


# --------------------------------------------------------------------------- ODEs
def _F(model, p, x):
    if model == "rm":
        return p["a"] * x / (1 + p["a"] * p["h"] * x)
    return p["b"] * x


def _growth(model, p, x):
    if model == "classic":
        return p["a"] * x
    K = p["K"]
    return p["r"] * x * (1 - x / K) if isfinite(K) else p["r"] * x


def rhs(model, p, z):
    if model == "chain":
        v, x, y = z
        graze = p["e"] * v * x
        return np.array([p["rho"] * (p["V"] - v) - graze,
                         p["eps"] * graze - p["mu"] * x - p["b"] * x * y,
                         p["c"] * p["b"] * x * y - p["g"] * y])
    x, y = z
    F = _F(model, p, x)
    return np.array([_growth(model, p, x) - F * y, p["c"] * F * y - p["g"] * y])


def integrate(model, p, z0, T, dt=0.05):
    """RK4, sampled every tick, densities clipped at 0. z0 = (x, y) or (v, x, y) for chain."""
    n_sub = max(1, int(round(1 / dt)))
    h = 1.0 / n_sub
    z = np.array(z0, float)
    out = np.empty((T + 1, len(z)))
    out[0] = z
    for t in range(1, T + 1):
        for _ in range(n_sub):
            k1 = rhs(model, p, z)
            k2 = rhs(model, p, z + h / 2 * k1)
            k3 = rhs(model, p, z + h / 2 * k2)
            k4 = rhs(model, p, z + h * k3)
            z = np.maximum(z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4), 0.0)
            if not np.isfinite(z).all() or z.max() > 1e6:
                out[t:] = np.nan
                return out
        out[t] = z
    return out


def equilibrium(model, p) -> dict:
    """Interior equilibrium, its stability (Jacobian eigenvalues) and predicted period."""
    c, g = p["c"], p["g"]
    if c <= 0 or g <= 0:
        return {"exists": False, "reason": "no predator growth/death"}
    if model == "chain":
        if p["b"] <= 0 or p["e"] <= 0:
            return {"exists": False, "reason": "no predation/grazing"}
        xs = g / (c * p["b"])
        vs = p["rho"] * p["V"] / (p["rho"] + p["e"] * xs)
        ys = (p["eps"] * p["e"] * vs - p["mu"]) / p["b"]
        if ys <= 0:
            return {"exists": False, "x": xs, "reason": "prey can't support predators: predators go extinct"}
        return _classify(model, p, np.array([vs, xs, ys]), xs, ys)
    target = g / c  # F(x*) = g/c
    if model == "rm":
        a, h = p["a"], p["h"]
        if a <= 0 or h * target >= 1:
            return {"exists": False, "reason": "predators can't reach break-even (satiation too strong)"}
        xs = target / (a * (1 - h * target))
    else:
        if p["b"] <= 0:
            return {"exists": False, "reason": "no predation"}
        xs = target / p["b"]
    gr = _growth(model, p, xs)
    if gr <= 0:
        return {"exists": False, "x": xs, "reason": "x* above prey carrying capacity: predators go extinct"}
    ys = gr / _F(model, p, xs)
    return _classify(model, p, np.array([xs, ys]), xs, ys)


def _classify(model, p, z0, xs, ys) -> dict:
    eps = 1e-7
    d = len(z0)
    J = np.zeros((d, d))
    for j in range(d):
        dz = np.zeros(d)
        dz[j] = eps * max(1.0, abs(z0[j]))
        J[:, j] = (rhs(model, p, z0 + dz) - rhs(model, p, z0 - dz)) / (2 * dz[j])
    ev = np.linalg.eigvals(J)
    lead = ev[np.argmax(ev.real)]
    re = float(lead.real)
    im = float(np.abs(ev.imag).max())
    if abs(re) < 1e-9 * max(1, im):
        kind = "center (neutral cycles)"
    elif re < 0:
        kind = "stable focus (damped oscillations)" if im > 0 else "stable node"
    else:
        kind = "unstable focus (limit cycle / growing oscillations)" if im > 0 else "saddle/unstable"
    return {"exists": True, "x": float(xs), "y": float(ys),
            "eig_real": re, "eig_imag": im, "type": kind,
            "period": float(2 * pi / im) if im > 0 else None,
            "damping_time": float(-1 / re) if re < -1e-12 else None}


def nullclines(model, p, xmax, n=120):
    if model == "chain":
        eq = equilibrium(model, p)
        return {"prey": [], "predator_x": eq.get("x")}
    xs = np.linspace(xmax / n, xmax, n)
    pts = []
    for x in xs:
        F = _F(model, p, x)
        if F > 0:
            yv = float(_growth(model, p, x) / F)
            if yv >= 0:
                pts.append([float(x), yv])
    eq = equilibrium(model, p)
    return {"prey": pts,
            "predator_x": eq.get("x")}


# --------------------------------------------------------------------------- mean-field map
def _pmf(n, q):
    q = min(max(q, 0.0), 1.0)
    return np.array([comb(n, k) * q ** k * (1 - q) ** (n - k) for k in range(n + 1)])


def meanfield(p: Params, T: int, x0: float, y0: float, v0: float, fertile: float) -> np.ndarray:
    """Well-mixed, hunger-structured map of the ABM rules. Returns (T+1, 3) [x, y, v]."""
    S, N, R = p.prey_starve_time, p.predator_starve_time, p.veg_regrow_time
    m = (2 * p.predator_kill_radius + 1) ** 2 - 1
    n_eat = (2 * p.prey_eat_radius + 1) ** 2
    K, kmax = p.prey_food_need, p.predator_max_kills
    q = np.zeros(S); q[0] = x0          # prey by hunger 0..S-1
    w = np.zeros(N); w[0] = y0          # predators by hunger 0..N-1
    v = v0
    regrow = deque([(fertile - v0) / R] * R)  # initial timers are uniform in [0, R)
    out = np.empty((T + 1, 3))
    out[0] = (x0, y0, v0)
    jk = np.arange(m + 1)
    je = np.arange(n_eat + 1)
    for t in range(1, T + 1):
        v += regrow.popleft()
        regrow.append(0.0)
        x, y = q.sum(), w.sum()
        # predation (each predator: kills min(kmax, prey neighbours))
        pk = _pmf(m, x)
        kills_pp = float((np.minimum(jk, kmax) * pk).sum())
        f_pred = 1 - pk[0]
        killed = min(x, y * kills_pp)
        if x > 0:
            q *= 1 - killed / x
        x = q.sum()
        # grazing
        pe = _pmf(n_eat, v)
        eat_pp = float((np.minimum(je, K) * pe).sum())
        phi = float(pe[K:].sum())
        demand = x * eat_pp
        if demand > v and demand > 0:
            scale = v / demand
            phi *= scale
            demand = v
        v -= demand
        regrow[-1] += demand
        # hunger update
        fedq, unq = q * phi, q * (1 - phi)
        q = np.concatenate([[fedq.sum()], unq[:-1]])
        fedw, unw = w * f_pred, w * (1 - f_pred)
        w = np.concatenate([[fedw.sum()], unw[:-1]])
        # reproduction into free neighbours
        occ = min(q.sum() + w.sum(), 1.0)
        free = 1 - occ ** 8
        w[0] += fedw.sum() * p.predator_repro_prob * free
        parents = fedq.sum() if p.prey_repro_requires_fed else q.sum()
        q[0] += parents * p.prey_repro_prob * free
        if p.prey_max_age > 0:
            q *= 1 - 1 / p.prey_max_age
        if p.predator_max_age > 0:
            w *= 1 - 1 / p.predator_max_age
        out[t] = (q.sum(), w.sum(), v)
    return out


# --------------------------------------------------------------------------- comparison
def _tail_stats(a):
    from .batch import dominant_period
    t = a[len(a) // 2:]
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return {"mean": None, "cv": None, "period": None}
    mean = float(t.mean())
    per = dominant_period(t) if len(t) > 20 else float("nan")
    return {"mean": mean, "cv": float(t.std() / mean) if mean > 0 else None,
            "period": per if isfinite(per) else None}


def compare(sim, model: str = "rm", fit_start: int = 0, extend: int = 0, max_points: int = 2000) -> dict:
    """Fit/derive `model` from a Simulation's history and simulate it from the same start."""
    if model not in MODELS:
        raise ValueError(f"unknown model {model}; choose from {list(MODELS)}")
    cells = sim.W * sim.H
    s = series(sim.history, cells)
    T = len(sim.history) - 1 + extend
    x0, y0, v0 = s["x"][0], s["y"][0], s["v"][0]
    res: dict = {"model": model, "name": MODELS[model], "equations": EQUATIONS[model], "cells": cells}

    if model == "meanfield":
        traj = meanfield(sim.p, T, x0, y0, v0, float(sim.fertile.mean()))
        res["params"] = {"m_kill_neigh": (2 * sim.p.predator_kill_radius + 1) ** 2 - 1,
                         "n_eat_cells": (2 * sim.p.prey_eat_radius + 1) ** 2,
                         "S": sim.p.prey_starve_time, "N": sim.p.predator_starve_time,
                         "R": sim.p.veg_regrow_time}
        res["rate_r2"] = {}
        res["equilibrium"] = None
        res["nullclines"] = None
    else:
        f = fit_rates(sim.history, cells, model, fit_start, float(sim.fertile.mean()))
        p = f["params"]
        res["params"], res["rate_r2"] = p, f["rate_r2"]
        if model == "chain":
            vxy = integrate(model, p, (v0, x0, y0), T)
            xy = vxy[:, 1:]
            traj = np.column_stack([xy, vxy[:, 0]])
        else:
            xy = integrate(model, p, (x0, y0), T)
            traj = np.column_stack([xy, np.full(T + 1, np.nan)])
        eq = equilibrium(model, p)
        if eq.get("exists"):
            eq["prey_count"], eq["predator_count"] = eq["x"] * cells, eq["y"] * cells
        res["equilibrium"] = eq
        xmax = max(float(np.nanmax(s["x"])), float(np.nanmax(xy[:, 0])) if np.isfinite(xy).any() else 0) * 1.1
        nc = nullclines(model, p, max(xmax, 1e-6))
        res["nullclines"] = {"prey": [[a * cells, b * cells] for a, b in nc["prey"]],
                             "predator_x": nc["predator_x"] * cells if nc["predator_x"] else None}

    n_abm = len(sim.history)
    res["fit"] = {
        "prey_r2": _r2(s["x"], traj[:n_abm, 0]) if np.isfinite(traj[:n_abm, 0]).all() else None,
        "predator_r2": _r2(s["y"], traj[:n_abm, 1]) if np.isfinite(traj[:n_abm, 1]).all() else None,
    }
    res["abm_stats"] = {"prey": _tail_stats(s["x"] * cells), "predators": _tail_stats(s["y"] * cells)}
    res["model_stats"] = {"prey": _tail_stats(traj[:n_abm, 0] * cells),
                          "predators": _tail_stats(traj[:n_abm, 1] * cells)}
    stride = max(1, (T + 1) // max_points)
    idx = np.arange(0, T + 1, stride)

    def clean(a):
        return [None if not np.isfinite(v) else round(float(v), 2) for v in a]

    res["traj"] = {"tick": idx.tolist(),
                   "prey": clean(traj[idx, 0] * cells),
                   "predators": clean(traj[idx, 1] * cells),
                   "vegetation": clean(traj[idx, 2] * cells)}
    return res
