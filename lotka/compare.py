"""Run the ABM and compare it with every Lotka-Volterra variant; write a report.

    python -m lotka.compare --preset lv_cycles --ticks 3000 --out runs/cmp_cycles
    python -m lotka.compare --set predator_starve_time=20 --fit-start 200 --models rm,chain,meanfield

Outputs in --out:
  comparison.png   time series (ABM solid, model dashed) + phase portraits, one row per model
  summary.json     fitted parameters, rate-fit R², equilibria/stability, ABM vs model statistics
  trajectories.csv tick, ABM prey/predators/vegetation and each model's prey/predators
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np

from . import lv
from .presets import PRESETS, load_params_file, preset_params
from .sim import Params, Simulation


def _clean(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def plot(sim, results: dict, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array([h["tick"] for h in sim.history])
    X = np.array([h["prey"] for h in sim.history])
    Y = np.array([h["predators"] for h in sim.history])
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(15, 3.3 * n), gridspec_kw={"width_ratios": [2.2, 1]}, squeeze=False)
    for row, (m, r) in enumerate(results.items()):
        tr = r["traj"]
        tt = np.array(tr["tick"])
        mp = np.array([np.nan if v is None else v for v in tr["prey"]])
        my = np.array([np.nan if v is None else v for v in tr["predators"]])
        ylim = max(X.max(), Y.max()) * 1.6
        ax = axes[row, 0]
        ax.plot(t, X, color="#d4b800", lw=0.9, label="ABM prey")
        ax.plot(t, Y, color="#d43a30", lw=0.9, label="ABM predators")
        ax.plot(tt, np.minimum(mp, ylim), color="#8a6f00", lw=1.6, ls="--", label="model prey")
        ax.plot(tt, np.minimum(my, ylim), color="#7a1a14", lw=1.6, ls="--", label="model predators")
        ax.set_ylim(0, ylim)
        eq = r.get("equilibrium") or {}
        sub = eq.get("type", "") if eq.get("exists") else (eq.get("reason", "") if eq else "rule-derived, no fitting")
        ax.set_title(f"{r['name']} — {sub}", fontsize=10, loc="left")
        ax.set_xlabel("tick")
        if row == 0:
            ax.legend(fontsize=8, ncol=4, loc="upper right")
        A, M = r["abm_stats"], r["model_stats"]

        def f(v):
            return "—" if v is None else f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}"
        txt = (f"mean prey ABM {f(A['prey']['mean'])} / model {f(M['prey']['mean'])}   "
               f"pred {f(A['predators']['mean'])} / {f(M['predators']['mean'])}\n"
               f"CV prey {f(A['prey']['cv'])} / {f(M['prey']['cv'])}   "
               f"period {f(A['prey']['period'])} / {f(M['prey']['period'])}")
        ax.text(0.01, 0.97, txt, transform=ax.transAxes, fontsize=8, va="top", family="monospace",
                bbox=dict(fc="white", alpha=0.7, lw=0))
        ax2 = axes[row, 1]
        ax2.plot(X, Y, color="#5a78d6", lw=0.5, alpha=0.6, label="ABM")
        ok = np.isfinite(mp) & np.isfinite(my)
        ax2.plot(np.minimum(mp[ok], X.max() * 1.5), np.minimum(my[ok], Y.max() * 1.5), color="k", lw=1.2, ls="--", label="model")
        nc = r.get("nullclines")
        if nc:
            if nc["prey"]:
                a = np.array(nc["prey"])
                ax2.plot(a[:, 0], a[:, 1], color="#d4b800", lw=0.8, ls=":", label="prey nullcline")
            if nc["predator_x"]:
                ax2.axvline(nc["predator_x"], color="#d43a30", lw=0.8, ls=":", label="predator nullcline")
        if eq.get("exists"):
            ax2.plot(eq["prey_count"], eq["predator_count"], "ko", ms=5)
        ax2.set_xlim(0, X.max() * 1.5)
        ax2.set_ylim(0, Y.max() * 1.5 if Y.max() > 0 else 1)
        ax2.set_xlabel("prey")
        ax2.set_ylabel("predators")
        if row == 0:
            ax2.legend(fontsize=7)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare the ABM with Lotka-Volterra models",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--preset", choices=list(PRESETS))
    ap.add_argument("--params-file", help="JSON params, e.g. runs/evolve/<run>/best.json")
    ap.add_argument("--set", action="append", default=[], metavar="PARAM=VALUE")
    ap.add_argument("--ticks", type=int, default=3000)
    ap.add_argument("--models", default="all", help=f"comma list from {list(lv.MODELS)} or 'all'")
    ap.add_argument("--fit-start", type=int, default=100, help="ignore ticks before this when fitting (transient)")
    ap.add_argument("--extend", type=int, default=0, help="integrate models this many ticks beyond the ABM")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    d = preset_params(a.preset) if a.preset else {}
    if a.params_file:
        d.update(load_params_file(a.params_file))
    for s in a.set:
        k, v = s.split("=", 1)
        d[k] = v
    p = Params.from_dict(d)
    models = list(lv.MODELS) if a.models == "all" else a.models.split(",")
    out = Path(a.out or f"runs/compare_{a.preset or 'custom'}_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)

    sim = Simulation(p)
    for _ in range(a.ticks):
        sim.step()
        if sim.extinct:
            break
    print(f"ABM ran {sim.tick} ticks: final prey={sim.history[-1]['prey']} predators={sim.history[-1]['predators']}")

    results = {}
    for m in models:
        try:
            results[m] = lv.compare(sim, m, min(a.fit_start, max(sim.tick - 10, 0)), a.extend, max_points=100000)
        except Exception as e:  # noqa: BLE001
            print(f"  {m}: failed ({e})")
            continue
        r = results[m]
        eq = r.get("equilibrium") or {}
        print(f"  {m:10s} mean prey {r['abm_stats']['prey']['mean']} vs {r['model_stats']['prey']['mean']}"
              f" | {eq.get('type', eq.get('reason', '-'))}")

    title = f"Lotka ABM vs Lotka-Volterra — preset={a.preset or 'custom'} {' '.join(a.set)} seed={p.seed}"
    plot(sim, results, out / "comparison.png", title)
    summary = {"params": p.to_dict(), "preset": a.preset, "ticks": sim.tick,
               "models": {m: {k: v for k, v in r.items() if k != "traj"} for m, r in results.items()}}
    (out / "summary.json").write_text(json.dumps(_clean(summary), indent=2))
    with (out / "trajectories.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tick", "abm_prey", "abm_predators", "abm_vegetation"]
                   + [f"{m}_{k}" for m in results for k in ("prey", "predators")])
        for i, h in enumerate(sim.history):
            row = [h["tick"], h["prey"], h["predators"], h["vegetation"]]
            for m, r in results.items():
                row += [r["traj"]["prey"][i], r["traj"]["predators"][i]]
            w.writerow(row)
    print(f"report -> {out}/comparison.png, summary.json, trajectories.csv")


if __name__ == "__main__":
    main()
