"""Experiments for the paper and the guide.

    python -m experiments.exp all            # everything (several hours on 8 cores... see estimates)
    python -m experiments.exp E2 E6a         # selected experiments
    python -m experiments.exp E2 --quick     # tiny smoke version

Each experiment writes raw results to paper/data/<id>*.{csv,npz,json}. Figures are made by
experiments/figures.py from these files, so plots can be changed without re-running.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from lotka import lv
from lotka.batch import dominant_period, summarize
from lotka.presets import PRESETS
from lotka.sim import Params, Simulation

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "data"
QUICK = False
WORKERS = os.cpu_count() or 1


# ----------------------------------------------------------------------------- helpers
def run(params: dict, ticks: int, snapshots=(), fit_models=(), fit_start=300, keep_series=True) -> dict:
    """Run one simulation; return summary (+ optional series, snapshots, LV fits)."""
    p = Params.from_dict(params)
    sim = Simulation(p)
    snaps = {}
    for t in range(1, ticks + 1):
        sim.step()
        if t in snapshots:
            snaps[t] = sim.render_codes()
        if sim.extinct:
            break
    h = sim.history
    out = {"params": p.to_dict(), "summary": summarize(h, ticks)}
    cells = p.width * p.height
    half = h[len(h) // 2:]
    out["summary"]["yield_prey"] = float(np.mean([r["harvested_prey"] for r in half]))
    out["summary"]["yield_predators"] = float(np.mean([r["harvested_predators"] for r in half]))
    out["summary"]["cells"] = cells
    if keep_series:
        out["series"] = {k: np.array([r[k] for r in h], dtype=np.float32)
                         for k in ("prey", "predators", "vegetation", "prey_eaten", "harvested_prey",
                                   "harvested_predators")}
    if snaps:
        out["snapshots"] = snaps
    fits = {}
    for m in fit_models:
        if out["summary"]["outcome"] != "coexist":
            continue
        try:
            r = lv.compare(sim, m, fit_start=min(fit_start, sim.tick // 4))
            fits[m] = {k: r[k] for k in ("params", "rate_r2", "equilibrium", "abm_stats", "model_stats", "fit")}
            fits[m]["traj"] = {k: r["traj"][k] for k in ("tick", "prey", "predators")}
        except Exception as e:  # noqa: BLE001
            fits[m] = {"error": str(e)}
    if fits:
        out["fits"] = fits
    return out


def _job(args):
    params, ticks, kw = args
    return run(params, ticks, **kw)


def pmap(jobs, desc=""):
    t0 = time.time()
    res = [None] * len(jobs)
    with ProcessPoolExecutor(WORKERS) as ex:
        for i, r in enumerate(ex.map(_job, jobs, chunksize=1)):
            res[i] = r
            if (i + 1) % max(1, len(jobs) // 10) == 0:
                print(f"  {desc}: {i + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    return res


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def write_rows(name, rows):
    DATA.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with (DATA / name).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"  -> {DATA / name} ({len(rows)} rows)")


def flat(tag: dict, r: dict) -> dict:
    row = dict(tag)
    row.update({k: v for k, v in r["summary"].items()})
    return row


def seeds(n):
    return list(range(2 if QUICK else n))


def T(t):
    return 200 if QUICK else t


def L(size):
    return 60 if QUICK else size


def preset(name, **kw):
    return {**PRESETS[name]["params"], **kw}


# ----------------------------------------------------------------------------- experiments
def E1():
    """Regimes: the presets on 200x200, 3000 ticks, 6 seeds; series + spatial snapshots of seed 0."""
    names = ["stable_coexistence", "lv_cycles", "paradox_of_enrichment", "predator_starvation",
             "overexploitation", "prey_overpopulation", "vegetation_scarcity"]
    jobs, tags = [], []
    for n in names:
        for s in seeds(6):
            kw = {"snapshots": (100, 1000, 2000, 2050, 2100, 2150) if s == 0 else ()}
            jobs.append((preset(n, width=L(200), height=L(200), seed=s), T(3000), kw))
            tags.append({"preset": n, "seed": s})
    res = pmap(jobs, "E1")
    write_rows("E1_regimes.csv", [flat(t, r) for t, r in zip(tags, res)])
    arr = {}
    for t, r in zip(tags, res):
        if t["seed"] == 0:
            for k, v in r["series"].items():
                arr[f"{t['preset']}__{k}"] = v
            for tick, g in r.get("snapshots", {}).items():
                arr[f"{t['preset']}__snap{tick}"] = g
    np.savez_compressed(DATA / "E1_series.npz", **arr)


def E2():
    """Sweep predator starvation time N (the key 'predator death rate' parameter)."""
    Ns = [2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 21, 25, 30, 35, 40, 50, 60]
    if QUICK:
        Ns = [3, 12, 30]
    jobs, tags = [], []
    for N in Ns:
        for s in seeds(6):
            fit = ("logistic", "rm", "chain", "meanfield") if s == 0 else ()
            jobs.append(({"width": L(200), "height": L(200), "seed": s, "predator_starve_time": N}, T(3000),
                         {"fit_models": fit, "keep_series": s == 0}))
            tags.append({"N": N, "seed": s})
    res = pmap(jobs, "E2")
    write_rows("E2_starve_sweep.csv", [flat(t, r) for t, r in zip(tags, res)])
    fits = {str(t["N"]): r.get("fits", {}) for t, r in zip(tags, res) if t["seed"] == 0}
    (DATA / "E2_fits.json").write_text(json.dumps(clean(fits)))
    arr = {f"N{t['N']}__{k}": v for t, r in zip(tags, res) if t["seed"] == 0 for k, v in r["series"].items()}
    np.savez_compressed(DATA / "E2_series.npz", **arr)


def E3():
    """Enrichment: vegetation regrowth time R for two predator regimes."""
    Rs = [4, 6, 8, 10, 14, 20, 28, 40, 60]
    if QUICK:
        Rs = [6, 20]
    jobs, tags = [], []
    for N in (12, 30):
        for R in Rs:
            for s in seeds(6):
                jobs.append(({"width": L(200), "height": L(200), "seed": s, "predator_starve_time": N,
                              "veg_regrow_time": R}, T(3000), {"keep_series": False}))
                tags.append({"N": N, "R": R, "seed": s})
    res = pmap(jobs, "E3")
    write_rows("E3_enrichment.csv", [flat(t, r) for t, r in zip(tags, res)])


def E4():
    """System size: demographic noise and extinction risk vs grid size."""
    sizes = [40, 60, 80, 100, 140, 200, 280]
    if QUICK:
        sizes = [40, 80]
    jobs, tags = [], []
    for name in ("stable_coexistence", "lv_cycles"):
        for Lx in sizes:
            for s in seeds(10):
                jobs.append((preset(name, width=Lx, height=Lx, seed=s), T(3000), {"keep_series": False}))
                tags.append({"preset": name, "L": Lx, "seed": s})
    res = pmap(jobs, "E4")
    write_rows("E4_size.csv", [flat(t, r) for t, r in zip(tags, res)])


def E5():
    """Copy the optimisation results (MAP-Elites v2) into the data folder."""
    src = ROOT / "runs" / "evolve" / "qd_v2"
    for f in ("archive.json", "generations.csv", "best.json", "config.json"):
        if (src / f).exists():
            (DATA / f"E5_{f}").write_bytes((src / f).read_bytes())
            print(f"  -> E5_{f}")
    src = ROOT / "runs" / "evolve" / "yield_v1"
    for f in ("best.json", "generations.csv", "config.json"):
        if (src / f).exists():
            (DATA / f"E5_yield_{f}").write_bytes((src / f).read_bytes())
            print(f"  -> E5_yield_{f}")


CTRL_BASE = {"width": 200, "height": 200}


def E6a():
    """Open-loop constant-effort harvesting of prey and of predators (yield-effort curves)."""
    effs = [0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
    if QUICK:
        effs = [0, 0.02]
    jobs, tags = [], []
    for sp, sname in ((1, "prey"), (2, "predators"), (3, "both")):
        for e in effs:
            for s in seeds(5):
                jobs.append(({**CTRL_BASE, "width": L(200), "height": L(200), "seed": s, "harvest_mode": 1,
                              "harvest_species": sp, "harvest_effort": e}, T(3000), {"keep_series": False}))
                tags.append({"species": sname, "effort": e, "seed": s})
    res = pmap(jobs, "E6a")
    write_rows("E6a_effort.csv", [flat(t, r) for t, r in zip(tags, res)])


def E6b():
    """Open-loop constant-quota harvesting of prey (and predators)."""
    qs_prey = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100]
    qs_pred = [0, 2, 4, 6, 8, 10, 12, 15, 20]
    if QUICK:
        qs_prey, qs_pred = [0, 40], [0, 8]
    jobs, tags = [], []
    for sp, sname, qs in ((1, "prey", qs_prey), (2, "predators", qs_pred)):
        for q in qs:
            for s in seeds(5):
                jobs.append(({**CTRL_BASE, "width": L(200), "height": L(200), "seed": s, "harvest_mode": 2,
                              "harvest_species": sp, "harvest_quota": q, "harvest_start": 500}, T(3000),
                             {"keep_series": False}))
                tags.append({"species": sname, "quota": q, "seed": s})
    res = pmap(jobs, "E6b")
    write_rows("E6b_quota.csv", [flat(t, r) for t, r in zip(tags, res)])


def E6c():
    """Feedback (P) control of predators in the oscillating regime, switched on at t=1500."""
    gains = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    setpoints = [400, 600]
    if QUICK:
        gains, setpoints = [0, 0.2], [600]
    jobs, tags = [], []
    for sp_name, sp in (("predators", 2), ("prey", 1)):
        for sp_val in (setpoints if sp == 2 else [800, 1200]):
            for g in gains:
                for s in seeds(5):
                    jobs.append((preset("lv_cycles", width=L(200), height=L(200), seed=s, harvest_mode=3,
                                        harvest_species=sp, harvest_setpoint=sp_val, harvest_gain=g,
                                        harvest_start=1500), T(4000), {"keep_series": s == 0}))
                    tags.append({"controlled": sp_name, "setpoint": sp_val, "gain": g, "seed": s})
    res = pmap(jobs, "E6c")
    rows = []
    arr = {}
    for t, r in zip(tags, res):
        row = flat(t, r)
        # before/after statistics around the switch-on time
        if "series" in r:
            x, y = r["series"]["prey"], r["series"]["predators"]
            arr[f"{t['controlled']}_{t['setpoint']}_{t['gain']}__prey"] = x
            arr[f"{t['controlled']}_{t['setpoint']}_{t['gain']}__predators"] = y
            arr[f"{t['controlled']}_{t['setpoint']}_{t['gain']}__harvest"] = (
                r["series"]["harvested_predators"] + r["series"]["harvested_prey"])
        rows.append(row)
    write_rows("E6c_feedback.csv", rows)
    np.savez_compressed(DATA / "E6c_series.npz", **arr)


def E6d():
    """Same mean yield, open loop vs closed loop: constant effort vs feedback on predators (lv_cycles)."""
    effs = [0.002, 0.004, 0.006, 0.008, 0.01, 0.015, 0.02]
    sps = [200, 300, 400, 500, 600, 700]
    if QUICK:
        effs, sps = [0.005], [500]
    jobs, tags = [], []
    for e in effs:
        for s in seeds(5):
            jobs.append((preset("lv_cycles", width=L(200), height=L(200), seed=s, harvest_mode=1, harvest_species=2,
                                harvest_effort=e), T(3000), {"keep_series": False}))
            tags.append({"controller": "effort", "value": e, "seed": s})
    for spv in sps:
        for s in seeds(5):
            jobs.append((preset("lv_cycles", width=L(200), height=L(200), seed=s, harvest_mode=3, harvest_species=2,
                                harvest_setpoint=spv, harvest_gain=0.2), T(3000), {"keep_series": False}))
            tags.append({"controller": "feedback", "value": spv, "seed": s})
    res = pmap(jobs, "E6d")
    write_rows("E6d_open_vs_closed.csv", [flat(t, r) for t, r in zip(tags, res)])


def E7():
    """LV comparison reports for two regimes (full fits, kept for figures/tables)."""
    out = {}
    for name in ("stable_coexistence", "lv_cycles"):
        r = run(preset(name, width=L(200), height=L(200), seed=42), T(3000),
                fit_models=tuple(lv.MODELS), fit_start=100)
        out[name] = {"summary": r["summary"], "fits": r.get("fits", {}),
                     "series": {k: r["series"][k].tolist() for k in ("prey", "predators", "vegetation")}}
    (DATA / "E7_lv_reports.json").write_text(json.dumps(clean(out)))
    print("  -> E7_lv_reports.json")


def _shock_run(args):
    params, kind, seed = args
    p = Params.from_dict({**params, "seed": seed})
    sim = Simulation(p)
    sim.step(1500)
    if kind == "pred_minus50":
        sim.perturb(2, -0.5)
    elif kind == "prey_plus100":
        sim.perturb(1, 1.0)
    sim.step(900)
    x = np.array([h["prey"] for h in sim.history[1300:]], np.float32)
    y = np.array([h["predators"] for h in sim.history[1300:]], np.float32)
    return x, y


def E8():
    """Stability probe: shock the stable regime at t=1500 and average the response over seeds."""
    base = {"width": L(200), "height": L(200)}
    out = {}
    for regime, prm in (("stable", base), ("cycles", {**base, "predator_starve_time": 30})):
        for kind in ("none", "pred_minus50", "prey_plus100"):
            jobs = [(prm, kind, s) for s in seeds(16)]
            with ProcessPoolExecutor(WORKERS) as ex:
                res = list(ex.map(_shock_run, jobs))
            out[f"{regime}__{kind}__prey"] = np.stack([r[0] for r in res])
            out[f"{regime}__{kind}__predators"] = np.stack([r[1] for r in res])
            print(f"  E8 {regime} {kind} done", flush=True)
    np.savez_compressed(DATA / "E8_shocks.npz", **out)


EXPERIMENTS = {"E1": E1, "E2": E2, "E3": E3, "E4": E4, "E5": E5, "E6a": E6a, "E6b": E6b,
               "E6c": E6c, "E6d": E6d, "E7": E7, "E8": E8}


def main():
    global QUICK, DATA, WORKERS
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="+")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    QUICK, WORKERS = a.quick, a.workers
    if QUICK:
        DATA = ROOT / "paper" / "data_quick"
    DATA.mkdir(parents=True, exist_ok=True)
    globals()["DATA"] = DATA
    todo = list(EXPERIMENTS) if a.which == ["all"] else a.which
    for k in todo:
        t0 = time.time()
        print(f"== {k}: {EXPERIMENTS[k].__doc__.strip().splitlines()[0]}", flush=True)
        EXPERIMENTS[k]()
        print(f"== {k} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
