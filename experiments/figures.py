"""Figures for the paper/guide from paper/data (python -m experiments.figures [--quick] [names...])."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "data"
FIG = ROOT / "paper" / "figures"

C_PREY, C_PRED, C_VEG = "#c9a400", "#c0392b", "#3e8e41"
GRID_CMAP = ListedColormap(["#5b4126", "#3e9a3a", "#f4e04d", "#e8443a", "#22252a"])
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 150, "savefig.bbox": "tight", "axes.titlesize": 9.5,
                     "legend.fontsize": 8, "legend.frameon": False})


def rows(name):
    with (DATA / name).open() as f:
        out = []
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                try:
                    d[k] = float(v)
                except (TypeError, ValueError):
                    d[k] = v
            out.append(d)
        return out


def group(rs, *keys):
    g = defaultdict(list)
    for r in rs:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def stat(rs, key, f=np.nanmean):
    v = np.array([r[key] if isinstance(r[key], float) else np.nan for r in rs], float)
    return f(v) if np.isfinite(v).any() else np.nan


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=130)
    plt.close(fig)
    print("  ->", FIG / f"{name}.pdf")


def sem(rs, key):
    v = np.array([r[key] for r in rs if isinstance(r[key], float) and math.isfinite(r[key])])
    return v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0


# ----------------------------------------------------------------------------- figures
def fig_snapshots():
    z = np.load(DATA / "E1_series.npz")
    ticks = [2000, 2050, 2100, 2150]
    fig, axs = plt.subplots(1, len(ticks) + 1, figsize=(10.5, 2.6),
                            gridspec_kw={"width_ratios": [1] * len(ticks) + [1.7]})
    for ax, t in zip(axs, ticks):
        ax.imshow(z[f"lv_cycles__snap{t}"], cmap=GRID_CMAP, vmin=0, vmax=4, interpolation="nearest")
        ax.set_title(f"t = {t}")
        ax.set_xticks([])
        ax.set_yticks([])
    ax = axs[-1]
    x, y = z["lv_cycles__prey"], z["lv_cycles__predators"]
    tt = np.arange(len(x))
    sl = slice(1900, 2250)
    ax.plot(tt[sl], x[sl], color=C_PREY, label="prey")
    ax.plot(tt[sl], y[sl], color=C_PRED, label="predators")
    for t in ticks:
        ax.axvline(t, color="k", lw=0.5, ls=":")
    ax.set_xlabel("tick")
    ax.legend(loc="upper right")
    ax.set_title("populations around the snapshots")
    save(fig, "snapshots")


def fig_regimes():
    z = np.load(DATA / "E1_series.npz")
    names = [("stable_coexistence", "Stable coexistence"), ("lv_cycles", "Sustained cycles (N=30)"),
             ("paradox_of_enrichment", "Enrichment (N=30, R=10)"),
             ("predator_starvation", "Predator starvation (N=3)"), ("overexploitation", "Overexploitation")]
    fig, axs = plt.subplots(len(names), 2, figsize=(9, 1.75 * len(names)), gridspec_kw={"width_ratios": [2.6, 1]})
    for i, (k, lab) in enumerate(names):
        x, y, v = z[f"{k}__prey"], z[f"{k}__predators"], z[f"{k}__vegetation"]
        a = axs[i, 0]
        a.plot(x, color=C_PREY, lw=0.9, label="prey")
        a.plot(y, color=C_PRED, lw=0.9, label="predators")
        a2 = a.twinx()
        a2.plot(v, color=C_VEG, lw=0.7, alpha=0.5)
        a2.set_ylabel("veg.", color=C_VEG, fontsize=7)
        a2.tick_params(labelsize=7, colors=C_VEG)
        a2.spines["top"].set_visible(False)
        a.set_title(lab, loc="left")
        if i == 0:
            a.legend(loc="upper right", ncol=2)
        if i == len(names) - 1:
            a.set_xlabel("tick")
        b = axs[i, 1]
        n = len(x)
        b.plot(x[n // 3:], y[n // 3:], color="#4a6fd0", lw=0.4) if n > 30 else None
        b.plot(x[:n // 3 + 1], y[:n // 3 + 1], color="#aaa", lw=0.4)
        b.set_xlabel("prey", fontsize=7)
        b.set_ylabel("predators", fontsize=7)
    fig.tight_layout()
    save(fig, "regimes")


def _fit_period(f, model):
    m = f.get(model, {})
    eq = m.get("equilibrium") or {}
    return eq.get("period"), eq.get("prey_count"), eq.get("predator_count"), eq.get("type")


def fig_starve_sweep():
    rs = rows("E2_starve_sweep.csv")
    fits = json.loads((DATA / "E2_fits.json").read_text())
    g = group(rs, "N")
    Ns = sorted(k[0] for k in g)
    out = {o: [np.mean([r["outcome"] == o for r in g[(N,)]]) for N in Ns]
           for o in ("coexist", "predators_extinct", "all_extinct")}
    fig, axs = plt.subplots(2, 2, figsize=(9, 6))
    a = axs[0, 0]
    a.stackplot(Ns, out["coexist"], out["predators_extinct"], out["all_extinct"],
                colors=["#6aa84f", "#e6b800", "#999"], labels=["coexistence", "predators extinct", "both extinct"])
    a.set_xlabel("predator starvation time N (ticks)")
    a.set_ylabel("fraction of seeds")
    a.set_title("(a) outcome")
    a.legend(loc="lower center", frameon=True, facecolor="white", framealpha=0.9)
    a.set_xlim(min(Ns), max(Ns))
    a = axs[0, 1]
    for key, col, lab in (("mean_prey", C_PREY, "prey"), ("mean_predators", C_PRED, "predators")):
        m = [stat(g[(N,)], key) for N in Ns]
        lo = [stat(g[(N,)], key.replace("mean", "min") + "_after_transient") if False else np.nan for N in Ns]
        a.plot(Ns, m, "o-", color=col, ms=3, label=f"ABM {lab}")
    eqx = [(_fit_period(fits.get(str(int(N)), {}), "chain")[1]) for N in Ns]
    eqy = [(_fit_period(fits.get(str(int(N)), {}), "chain")[2]) for N in Ns]
    a.plot(Ns, [v if v else np.nan for v in eqx], "--", color=C_PREY, lw=1, label="chain-LV equilibrium")
    a.plot(Ns, [v if v else np.nan for v in eqy], "--", color=C_PRED, lw=1)
    a.set_xlabel("N")
    a.set_ylabel("mean population (2nd half)")
    a.set_title("(b) mean levels: ABM vs fitted equilibrium")
    a.legend()
    a = axs[1, 0]
    a.plot(Ns, [stat(g[(N,)], "cv_prey") for N in Ns], "o-", color=C_PREY, ms=3, label="prey CV")
    a.plot(Ns, [stat(g[(N,)], "cv_predators") for N in Ns], "o-", color=C_PRED, ms=3, label="predator CV")
    a.set_xlabel("N")
    a.set_ylabel("coefficient of variation")
    a.set_title("(c) oscillation amplitude")
    a.legend()
    a = axs[1, 1]
    # period is only meaningful for coexisting runs with visible oscillations (CV >= 0.05)
    per = []
    for N in Ns:
        ok = [r for r in g[(N,)] if r["outcome"] == "coexist" and isinstance(r["cv_prey"], float) and r["cv_prey"] >= 0.05]
        per.append(stat(ok, "period_prey", np.nanmedian) if len(ok) >= len(g[(N,)]) / 2 else np.nan)
    a.plot(Ns, per, "o-", color="k", ms=3, label="ABM (autocorrelation, coexisting, CV≥0.05)")
    for model, ls in (("rm", "--"), ("chain", ":")):
        a.plot(Ns, [(_fit_period(fits.get(str(int(N)), {}), model)[0] or np.nan) for N in Ns], ls,
               label=f"{model.upper() if model == 'rm' else 'chain'} LV prediction 2π/Im λ")
    a.set_xlabel("N")
    a.set_ylabel("period (ticks)")
    a.set_title("(d) cycle period")
    a.legend()
    fig.tight_layout()
    save(fig, "starve_sweep")


def fig_lv_compare():
    d = json.loads((DATA / "E7_lv_reports.json").read_text())
    r = d["lv_cycles"]
    x, y = np.array(r["series"]["prey"]), np.array(r["series"]["predators"])
    models = [("classic", "Classic LV"), ("rm", "Rosenzweig–MacArthur"), ("chain", "Veg–prey–predator chain"),
              ("meanfield", "Mean-field of the rules (no fitting)")]
    fig, axs = plt.subplots(len(models), 2, figsize=(9.5, 1.9 * len(models)), gridspec_kw={"width_ratios": [2.4, 1]})
    for i, (m, lab) in enumerate(models):
        f = r["fits"].get(m)
        a, b = axs[i, 0], axs[i, 1]
        a.plot(x, color=C_PREY, lw=0.8)
        a.plot(y, color=C_PRED, lw=0.8)
        if f and "traj" in f:
            tt = np.array(f["traj"]["tick"])
            mx = np.array([np.nan if v is None else v for v in f["traj"]["prey"]], float)
            my = np.array([np.nan if v is None else v for v in f["traj"]["predators"]], float)
            cap = 1.6 * max(x.max(), y.max())
            a.plot(tt, np.minimum(mx, cap), "--", color="#6b5600", lw=1.4)
            a.plot(tt, np.minimum(my, cap), "--", color="#6e1a12", lw=1.4)
            a.set_ylim(0, cap)
            b.plot(x, y, color="#4a6fd0", lw=0.3, alpha=0.7)
            ok = np.isfinite(mx) & np.isfinite(my)
            b.plot(np.minimum(mx[ok], 1.5 * x.max()), np.minimum(my[ok], 1.5 * y.max()), "k--", lw=1)
            b.set_xlim(0, 1.5 * x.max())
            b.set_ylim(0, 1.5 * y.max())
            eq = f.get("equilibrium") or {}
            if eq.get("exists"):
                b.plot(eq["prey_count"], eq["predator_count"], "ko", ms=3)
            A, M = f["abm_stats"], f["model_stats"]

            def fm(v):
                return "–" if v is None else f"{v:.0f}" if abs(v) >= 10 else f"{v:.2f}"
            a.text(0.01, 0.95, f"mean prey {fm(A['prey']['mean'])} / {fm(M['prey']['mean'])}   "
                   f"CV {fm(A['prey']['cv'])} / {fm(M['prey']['cv'])}   "
                   f"period {fm(A['prey']['period'])} / {fm(M['prey']['period'])}   (ABM / model)",
                   transform=a.transAxes, fontsize=7, va="top", family="monospace")
        a.set_title(lab + ((" — " + (f.get("equilibrium") or {}).get("type", "")) if f and f.get("equilibrium") else ""),
                    loc="left")
        b.set_xlabel("prey", fontsize=7)
        b.set_ylabel("predators", fontsize=7)
    axs[-1, 0].set_xlabel("tick")
    fig.tight_layout()
    save(fig, "lv_compare")


def fig_spatial_rate():
    fits = json.loads((DATA / "E2_fits.json").read_text())
    Ns, bs, r2 = [], [], []
    for N, f in sorted(fits.items(), key=lambda kv: float(kv[0])):
        c = f.get("chain", {})
        if "params" in c:
            Ns.append(float(N))
            bs.append(c["params"]["b"])
            r2.append(c["rate_r2"]["predation"])
    fig, ax = plt.subplots(figsize=(4.6, 3))
    ax.axhline(8, color="k", ls="--", lw=1, label="well-mixed prediction (8 neighbours)")
    ax.plot(Ns, bs, "o-", color=C_PRED, ms=4, label="fitted from ABM events")
    ax.set_xlabel("predator starvation time N")
    ax.set_ylabel("predation rate b (per predator per prey density per tick)")
    ax.set_ylim(0, 9)
    ax.legend(loc="center right")
    save(fig, "spatial_rate")


def fig_enrichment():
    rs = rows("E3_enrichment.csv")
    g = group(rs, "N", "R")
    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    for N, ls in ((12.0, "o-"), (30.0, "s--")):
        Rs = sorted(k[1] for k in g if k[0] == N)
        axs[0].plot(Rs, [stat(g[(N, R)], "cv_prey") for R in Rs], ls, ms=3, label=f"N={N:.0f}")
        axs[1].plot(Rs, [stat(g[(N, R)], "max_prey") for R in Rs], ls, ms=3, label=f"N={N:.0f}")
        axs[2].plot(Rs, [np.mean([r["outcome"] == "coexist" for r in g[(N, R)]]) for R in Rs], ls, ms=3,
                    label=f"N={N:.0f}")
    for a, t in zip(axs, ("(a) prey CV", "(b) peak prey", "(c) P(coexistence)")):
        a.set_xlabel("vegetation regrowth time R  (← richer)")
        a.set_title(t)
        a.invert_xaxis()
        a.legend()
    axs[2].set_ylim(-0.05, 1.05)
    fig.tight_layout()
    save(fig, "enrichment")


def fig_size():
    rs = rows("E4_size.csv")
    g = group(rs, "preset", "L")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for name, mk, lab in (("stable_coexistence", "o-", "stable regime"), ("lv_cycles", "s-", "cycling regime (N=30)")):
        Ls = sorted(k[1] for k in g if k[0] == name)
        cells = np.array(Ls) ** 2
        cv = [stat([r for r in g[(name, Lx)] if r["outcome"] == "coexist"], "cv_prey")
              if any(r["outcome"] == "coexist" for r in g[(name, Lx)]) else np.nan for Lx in Ls]
        axs[0].loglog(cells, cv, mk, ms=4, label=lab)
        axs[1].plot(cells, [np.mean([r["outcome"] == "coexist" for r in g[(name, Lx)]]) for Lx in Ls], mk, ms=4,
                    label=lab)
        if name == "stable_coexistence":
            ok = np.isfinite(cv)
            c0, v0 = cells[ok][-1], np.array(cv)[ok][-1]
            axs[0].loglog(cells, v0 * np.sqrt(c0 / cells), "k:", lw=1, label=r"$\propto 1/\sqrt{\mathrm{cells}}$")
    axs[0].set_xlabel("grid cells")
    axs[0].set_ylabel("prey CV (coexisting runs)")
    axs[0].set_title("(a) fluctuation size")
    axs[0].legend()
    axs[1].set_xscale("log")
    axs[1].set_xlabel("grid cells")
    axs[1].set_ylabel("P(both survive 3000 ticks)")
    axs[1].set_title("(b) persistence")
    axs[1].set_ylim(-0.05, 1.05)
    axs[1].legend()
    Ls_all = sorted({k[1] for k in g})
    for a in axs:
        a.minorticks_off()
        a.set_xticks([L * L for L in Ls_all])
        a.set_xticklabels([f"{int(L)}²" for L in Ls_all], fontsize=7)
        a.set_xlabel("grid size (cells)")
    fig.tight_layout()
    save(fig, "size")


def fig_qd():
    p = DATA / "E5_archive.json"
    if not p.exists():
        print("  (no E5 archive yet)")
        return
    ar = json.loads(p.read_text())
    (xn, xl, xh, xb), (yn, yl, yh, yb) = ar["axes"]["x"], ar["axes"]["y"]
    M = np.full((int(yb), int(xb)), np.nan)
    for k, e in ar["cells"].items():
        i, j = map(int, k.split(","))
        M[j, i] = e["fitness"]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1, 1.25]})
    im = axs[0].imshow(M, origin="lower", extent=[xl, xh, yl, yh], aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=axs[0], label="robust-coexistence fitness")
    axs[0].set_xlabel("prey CV (oscillation amplitude)")
    axs[0].set_ylabel("prey period (ticks)")
    axs[0].set_title(f"(a) MAP-Elites map: {len(ar['cells'])}/{int(xb * yb)} behaviours found")
    # three illustrative elites: low, medium, high CV
    from lotka.sim import Params, Simulation
    cells = sorted(ar["cells"].values(), key=lambda e: e["metrics"]["cv_prey"] or 0)
    picks = [cells[len(cells) // 10], cells[len(cells) // 2], cells[int(len(cells) * 0.9)]]
    for e, col in zip(picks, ("#2c7bb6", "#7b3294", "#d7191c")):
        prm = dict(e["params"])
        prm["seed"] = 1
        s = Simulation(Params.from_dict(prm))
        s.step(1500)
        x = np.array([h["prey"] for h in s.history], float)
        cv = e["metrics"]["cv_prey"]
        axs[0].plot(cv, e["metrics"]["period_prey"], "o", mfc="none", mec=col, ms=9, mew=2)
        axs[1].plot(x / max(x.mean(), 1), color=col, lw=0.8, label=f"CV≈{cv:.2f}, period≈{e['metrics']['period_prey'] or 0:.0f}")
    axs[1].set_xlabel("tick")
    axs[1].set_ylabel("prey / mean prey")
    axs[1].set_title("(b) three evolved parameter sets (circled in a)")
    axs[1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "qd_map")


def _lv_harvest_pred():
    d = json.loads((DATA / "E7_lv_reports.json").read_text())
    return d["stable_coexistence"]["fits"]["logistic"]["params"], d["stable_coexistence"]["summary"]["cells"]


def fig_effort():
    rs = rows("E6a_effort.csv")
    g = group(rs, "species", "effort")
    p, cells = _lv_harvest_pred()
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.2))
    for sp, col, lab in (("prey", C_PREY, "harvest prey"), ("predators", C_PRED, "harvest predators"),
                         ("both", "#555", "harvest both")):
        E = sorted(k[1] for k in g if k[0] == sp)
        yk = "yield_prey" if sp == "prey" else "yield_predators" if sp == "predators" else None
        yv = [stat(g[(sp, e)], yk) if yk else stat(g[(sp, e)], "yield_prey") + stat(g[(sp, e)], "yield_predators")
              for e in E]
        axs[0].plot(E, yv, "o-", color=col, ms=3, label=lab)
    axs[0].set_xlabel("effort E (fraction removed per tick)")
    axs[0].set_ylabel("yield (individuals / tick)")
    axs[0].set_title("(a) yield–effort curves")
    axs[0].legend()
    for ax, sp, ttl in ((axs[1], "prey", "(b) harvesting prey"), (axs[2], "both", "(c) harvesting both (Volterra)")):
        E = np.array(sorted(k[1] for k in g if k[0] == sp))
        ax.plot(E, [stat(g[(sp, e)], "mean_prey") for e in E], "o-", color=C_PREY, ms=3, label="ABM prey")
        ax.plot(E, [stat(g[(sp, e)], "mean_predators") for e in E], "o-", color=C_PRED, ms=3, label="ABM predators")
        # LV (logistic, fitted at E=0) prediction: x* = (g+Ey)/(c b), y* = (r(1-x*/K) - Ex)/b
        Ex = E
        Ey = E if sp == "both" else 0 * E
        xs = (p["g"] + Ey) / (p["c"] * p["b"])
        ys = (p["r"] * (1 - xs / p["K"]) - Ex) / p["b"]
        ax.plot(E, xs * cells, "--", color=C_PREY, lw=1, label="logistic-LV prediction")
        ax.plot(E, np.maximum(ys, 0) * cells, "--", color=C_PRED, lw=1)
        ax.set_xlabel("effort E")
        ax.set_ylabel("mean population (2nd half)")
        ax.set_title(ttl)
        ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "harvest_effort")


def fig_quota():
    rs = rows("E6b_quota.csv")
    g = group(rs, "species", "quota")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, sp, col in ((axs[0], "prey", C_PREY), (axs[1], "predators", C_PRED)):
        Q = sorted(k[1] for k in g if k[0] == sp)
        ax.plot(Q, [stat(g[(sp, q)], "yield_" + ("prey" if sp == "prey" else "predators")) for q in Q], "o-",
                color=col, ms=3, label="realised yield")
        ax.plot(Q, Q, "k:", lw=0.8, label="quota")
        ax2 = ax.twinx()
        ax2.plot(Q, [np.mean([r["outcome"] == "coexist" for r in g[(sp, q)]]) for q in Q], "s--", color="#555",
                 ms=3, label="P(coexistence)")
        ax2.set_ylim(-0.05, 1.05)
        ax2.set_ylabel("P(coexistence)")
        ax2.spines["top"].set_visible(False)
        ax.set_xlabel(f"quota Q ({sp} removed per tick)")
        ax.set_ylabel("yield (individuals / tick)")
        ax.set_title(f"constant-quota harvesting of {sp}")
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=7, framealpha=0.9)
        ax.set_ylim(bottom=0, top=max(Q) * 1.05)
    fig.tight_layout()
    save(fig, "harvest_quota")


def fig_feedback():
    rs = rows("E6c_feedback.csv")
    z = np.load(DATA / "E6c_series.npz")
    sp_val = 400.0
    gains = sorted({r["gain"] for r in rs if r["controlled"] == "predators" and r["setpoint"] == sp_val})
    fig, axs = plt.subplots(2, 2, figsize=(10, 5.6))
    show = [g for g in gains if g in (0.0, 0.05, 0.2, 1.0)]
    for gval, col in zip(show, ("#999", "#5aa0d0", "#1f5fa0", "#0b2d55")):
        cands = [f"predators_{int(sp_val)}_{gval}", f"predators_{int(sp_val)}_{int(gval)}" if gval == int(gval) else ""]
        key = next((k for k in cands if k and f"{k}__predators" in z), None)
        if key is None:
            continue
        y = z[f"{key}__predators"]
        axs[0, 0].plot(y, color=col, lw=0.8, label=f"Kp = {gval}")
        axs[1, 0].plot(z[f"{key}__prey"], color=col, lw=0.8)
    for a in (axs[0, 0], axs[1, 0]):
        a.axvline(1500, color="k", lw=0.8, ls=":")
        a.set_xlim(1000, 3000)
    axs[0, 0].axhline(sp_val, color="k", lw=0.8, ls="--")
    axs[0, 0].set_ylabel("predators")
    axs[0, 0].set_title("(a) controlled output (predators), controller on at t=1500")
    axs[0, 0].legend(ncol=4)
    axs[1, 0].set_ylabel("prey")
    axs[1, 0].set_xlabel("tick")
    axs[1, 0].set_title("(b) uncontrolled state (prey)")
    for spv, ls in ((400.0, "o-"), (600.0, "s--")):
        sub = group([r for r in rs if r["controlled"] == "predators" and r["setpoint"] == spv], "gain")
        G = sorted(k[0] for k in sub)
        axs[0, 1].plot(G, [stat(sub[(gg,)], "cv_predators") for gg in G], ls, color=C_PRED, ms=3,
                       label=f"predator CV, target {spv:.0f}")
        axs[0, 1].plot(G, [stat(sub[(gg,)], "cv_prey") for gg in G], ls, color=C_PREY, ms=3,
                       label=f"prey CV, target {spv:.0f}")
        axs[1, 1].plot(G, [stat(sub[(gg,)], "yield_predators") for gg in G], ls, color="k", ms=3,
                       label=f"target {spv:.0f}")
    for a in (axs[0, 1], axs[1, 1]):
        a.set_xscale("symlog", linthresh=0.02)
        a.set_xlabel("feedback gain Kp")
    axs[0, 1].set_ylabel("CV after switch-on (2nd half)")
    axs[0, 1].set_title("(c) oscillation suppression")
    axs[0, 1].legend(fontsize=7)
    axs[1, 1].set_ylabel("predator yield / tick")
    axs[1, 1].set_title("(d) control effort (harvest)")
    axs[1, 1].legend()
    fig.tight_layout()
    save(fig, "feedback")


def fig_open_closed():
    rs = rows("E6d_open_vs_closed.csv")
    g = group(rs, "controller", "value")
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for ctrl, mk, col, lab in (("effort", "o-", "#999", "open loop: constant effort"),
                               ("feedback", "s-", "#1f5fa0", "closed loop: P-control (Kp=0.2)")):
        vals = sorted(k[1] for k in g if k[0] == ctrl)
        xs = [stat(g[(ctrl, v)], "yield_predators") for v in vals]
        ys = [stat(g[(ctrl, v)], "cv_prey") for v in vals]
        ax.plot(xs, ys, mk, color=col, ms=4, label=lab)
    ax.set_xlabel("predator yield (individuals / tick)")
    ax.set_ylabel("prey CV (lower = steadier)")
    ax.set_title("same harvest, open vs closed loop (cycling regime)")
    ax.legend()
    save(fig, "open_vs_closed")


def fig_shocks():
    z = np.load(DATA / "E8_shocks.npz")
    d = json.loads((DATA / "E7_lv_reports.json").read_text())
    eq = d["stable_coexistence"]["fits"]["chain"]["equilibrium"]
    t = np.arange(z["stable__none__prey"].shape[1]) + 1300 - 1500
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.1))
    for ax, regime, ttl in ((axs[0], "stable", "(a) stable regime: −50% predators at t=0"),
                            (axs[1], "cycles", "(b) cycling regime: −50% predators at t=0")):
        for k, col in (("prey", C_PREY), ("predators", C_PRED)):
            base = z[f"{regime}__none__{k}"].mean(0)
            sh = z[f"{regime}__pred_minus50__{k}"]
            ax.plot(t, sh.mean(0), color=col, lw=1.2, label=f"{k} (mean of {sh.shape[0]} seeds)")
            ax.plot(t, base, color=col, lw=0.8, ls=":", alpha=0.8)
            ax.fill_between(t, np.percentile(sh, 25, 0), np.percentile(sh, 75, 0), color=col, alpha=0.15, lw=0)
        ax.axvline(0, color="k", lw=0.7)
        ax.set_xlabel("ticks after shock")
        ax.set_title(ttl)
        ax.legend(fontsize=7)
    # deviation of predators from unshocked baseline, stable regime, vs exponential envelope from the fitted model
    dev = z["stable__pred_minus50__predators"].mean(0) - z["stable__none__predators"].mean(0)
    ax = axs[2]
    ax.plot(t, dev, color=C_PRED, lw=1.2, label="ABM: predator deviation")
    if eq.get("damping_time"):
        tt = t[t >= 0]
        a0 = dev[t == 0][0] if (t == 0).any() else dev[t >= 0][0]
        env = abs(a0) * np.exp(-tt / eq["damping_time"])
        ax.plot(tt, env, "k--", lw=1, label=f"chain-LV envelope, τ = {eq['damping_time']:.0f}")
        ax.plot(tt, -env, "k--", lw=1)
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_xlabel("ticks after shock")
    ax.set_title("(c) return to equilibrium vs eigenvalue prediction")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "shocks")


FIGS = {"snapshots": fig_snapshots, "regimes": fig_regimes, "starve_sweep": fig_starve_sweep,
        "lv_compare": fig_lv_compare, "spatial_rate": fig_spatial_rate, "enrichment": fig_enrichment,
        "size": fig_size, "qd": fig_qd, "effort": fig_effort, "quota": fig_quota, "feedback": fig_feedback,
        "open_closed": fig_open_closed, "shocks": fig_shocks}


def main():
    global DATA, FIG
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    if a.quick:
        DATA, FIG = ROOT / "paper" / "data_quick", ROOT / "paper" / "figures_quick"
    for n in a.names or FIGS:
        try:
            FIGS[n]()
        except FileNotFoundError as e:
            print(f"  {n}: missing data ({e.filename})")
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  {n}: FAILED {e}")


if __name__ == "__main__":
    sys.exit(main())
