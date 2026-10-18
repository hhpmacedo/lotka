"""Headless batch runner / parameter sweeps.

Examples
  # single run, 2000 ticks
  python -m lotka.batch --ticks 2000 --out runs/baseline

  # sweep predator starvation time x prey reproduction, 3 seeds each, 4 workers
  python -m lotka.batch --ticks 3000 --seeds 0,1,2 \
      --set width=150 --set height=150 \
      --sweep predator_starve_time=6,12,24 --sweep prey_repro_prob=0.04,0.08,0.16 \
      --workers 4 --out runs/starve_vs_repro

Sweep values can be a comma list (a,b,c) or a range lo:hi:n (n evenly spaced values).

Outputs in --out:
  summary.csv            one row per run: params + outcome statistics
  series/run_XXXX.csv    full per-tick metrics for every run (disable with --no-series)
  config.json            the exact command configuration
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .sim import METRIC_KEYS, Params, Simulation


def parse_values(spec: str) -> list[str]:
    if ":" in spec:
        lo, hi, n = spec.split(":")
        return [repr(round(v, 10)) for v in np.linspace(float(lo), float(hi), int(n))]
    return [v.strip() for v in spec.split(",") if v.strip()]


def dominant_period(x: np.ndarray) -> float:
    """Dominant oscillation period (ticks) via autocorrelation; nan if none."""
    x = np.asarray(x, dtype=float)
    if len(x) < 20 or x.std() == 0:
        return float("nan")
    x = x - x.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    ac /= ac[0]
    # first local max after first zero crossing
    neg = np.nonzero(ac < 0)[0]
    if len(neg) == 0:
        return float("nan")
    start = neg[0]
    seg = ac[start:len(x) // 2]
    if len(seg) < 3:
        return float("nan")
    k = int(np.argmax(seg))
    return float(start + k) if seg[k] > 0.1 else float("nan")


def summarize(hist: list[dict], ticks_requested: int) -> dict:
    prey = np.array([h["prey"] for h in hist], dtype=float)
    pred = np.array([h["predators"] for h in hist], dtype=float)
    veg = np.array([h["vegetation"] for h in hist], dtype=float)
    ticks = len(hist) - 1

    def ext_tick(a):
        z = np.nonzero(a == 0)[0]
        return int(z[0]) if len(z) else -1

    pe, de = ext_tick(prey), ext_tick(pred)
    if pe >= 0 and de >= 0:
        outcome = "all_extinct"
    elif de >= 0:
        outcome = "predators_extinct"
    elif pe >= 0:
        outcome = "prey_extinct"  # predators will follow shortly
    else:
        outcome = "coexist"

    tail = slice(len(hist) // 2, None)  # second half = after transient
    out = {
        "ticks_run": ticks,
        "outcome": outcome,
        "prey_extinct_tick": pe,
        "predator_extinct_tick": de,
        "final_prey": int(prey[-1]),
        "final_predators": int(pred[-1]),
        "final_vegetation": int(veg[-1]),
        "max_prey": int(prey.max()),
        "max_predators": int(pred.max()),
        "min_prey_after_transient": int(prey[tail].min()),
        "min_predators_after_transient": int(pred[tail].min()),
    }
    for name, a in (("prey", prey), ("predators", pred), ("vegetation", veg)):
        t = a[tail]
        out[f"mean_{name}"] = round(float(t.mean()), 2)
        out[f"std_{name}"] = round(float(t.std()), 2)
        out[f"cv_{name}"] = round(float(t.std() / t.mean()), 4) if t.mean() > 0 else float("nan")
    out["period_prey"] = dominant_period(prey[tail])
    out["period_predators"] = dominant_period(pred[tail])
    # lag of predator peak behind prey (ticks) via cross-correlation
    if outcome == "coexist" and prey[tail].std() > 0 and pred[tail].std() > 0:
        a = prey[tail] - prey[tail].mean()
        b = pred[tail] - pred[tail].mean()
        maxlag = min(len(a) // 3, 500)
        cc = [np.dot(a[: len(a) - L], b[L:]) for L in range(maxlag)]
        out["predator_lag"] = int(np.argmax(cc))
    else:
        out["predator_lag"] = -1
    return out


def run_one(job: dict) -> dict:
    p = Params.from_dict(job["params"])
    sim = Simulation(p)
    t0 = time.time()
    for _ in range(job["ticks"]):
        sim.step()
        if job["stop_on"] == "all" and sim.extinct:
            break
        if job["stop_on"] == "any":
            last = sim.history[-1]
            if last["prey"] == 0 or last["predators"] == 0:
                break
    row = {"run": job["run"], **job["swept"], "seed": p.seed}
    row.update(summarize(sim.history, job["ticks"]))
    row["seconds"] = round(time.time() - t0, 2)
    if job["series_dir"]:
        path = Path(job["series_dir"]) / f"run_{job['run']:04d}.csv"
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=METRIC_KEYS)
            w.writeheader()
            w.writerows(sim.history)
    return row


def build_jobs(args) -> list[dict]:
    base: dict = {}
    for s in args.set:
        k, v = s.split("=", 1)
        base[k] = v
    sweeps = []
    for s in args.sweep:
        k, v = s.split("=", 1)
        sweeps.append((k, parse_values(v)))
    valid = {f["name"] for f in Params.schema()}
    for k in list(base) + [k for k, _ in sweeps]:
        if k not in valid:
            raise SystemExit(f"unknown parameter: {k}\nvalid: {sorted(valid)}")
    seeds = [int(s) for s in parse_values(args.seeds)]
    keys = [k for k, _ in sweeps]
    combos = list(itertools.product(*[v for _, v in sweeps])) or [()]
    jobs, run = [], 0
    for combo in combos:
        for seed in seeds:
            params = dict(base)
            params.update(dict(zip(keys, combo)))
            params["seed"] = seed
            typed = Params.from_dict(params).to_dict()
            jobs.append({"run": run, "params": typed, "swept": {k: typed[k] for k in keys},
                         "ticks": args.ticks, "stop_on": args.stop_on,
                         "series_dir": None if args.no_series else str(Path(args.out) / "series")})
            run += 1
    return jobs


def main(argv=None):
    ap = argparse.ArgumentParser(description="Lotka batch runner",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--seeds", default="0", help="comma list or lo:hi:n")
    ap.add_argument("--set", action="append", default=[], metavar="PARAM=VALUE", help="fixed parameter")
    ap.add_argument("--sweep", action="append", default=[], metavar="PARAM=V1,V2|lo:hi:n", help="swept parameter")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--stop-on", choices=["all", "any", "never"], default="all",
                    help="stop a run early when all / any species go extinct")
    ap.add_argument("--no-series", action="store_true", help="don't write per-tick CSVs")
    ap.add_argument("--out", default=f"runs/{time.strftime('%Y%m%d_%H%M%S')}")
    args = ap.parse_args(argv)

    out = Path(args.out)
    (out / "series").mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    print(f"{len(jobs)} runs x {args.ticks} ticks -> {out}  (workers={args.workers})")

    rows = []
    t0 = time.time()
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(args.workers) as ex:
            futs = [ex.submit(run_one, j) for j in jobs]
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result()
                rows.append(r)
                print(f"  [{i}/{len(jobs)}] run {r['run']}: {r['outcome']} "
                      f"prey~{r['mean_prey']} pred~{r['mean_predators']} ({r['seconds']}s)")
    else:
        for i, j in enumerate(jobs, 1):
            r = run_one(j)
            rows.append(r)
            print(f"  [{i}/{len(jobs)}] run {r['run']}: {r['outcome']} "
                  f"prey~{r['mean_prey']} pred~{r['mean_predators']} ({r['seconds']}s)")
    rows.sort(key=lambda r: r["run"])
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"done in {time.time() - t0:.1f}s -> {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
