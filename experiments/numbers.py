"""Compute every number quoted in the paper from paper/data and write paper/numbers.tex
(+ generated tables). Run: python -m experiments.numbers"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "paper" / "data"
OUT = ROOT / "paper"
M: dict[str, str] = {}
J: dict = {}


def rows(name):
    out = []
    with (DATA / name).open() as f:
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


def mean(rs, k, f=np.nanmean):
    v = np.array([r[k] if isinstance(r[k], float) else np.nan for r in rs], float)
    return float(f(v)) if np.isfinite(v).any() else float("nan")


def pcoex(rs):
    return float(np.mean([r["outcome"] == "coexist" for r in rs]))


def put(name, value, fmt="{:.0f}"):
    J[name] = value
    if isinstance(value, str):
        s = value
    elif value is None or (isinstance(value, float) and not math.isfinite(value)):
        s = "n/a"
    else:
        s = fmt.format(value)
    M[name] = s


def safe(fn):
    try:
        fn()
    except FileNotFoundError as e:
        print(f"  {fn.__name__}: missing {Path(e.filename).name}")
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"  {fn.__name__}: FAILED {e}")


# ----------------------------------------------------------------------------- sections
def e1():
    rs = rows("E1_regimes.csv")
    g = group(rs, "preset")
    labels = {"stable_coexistence": "Stable coexistence", "lv_cycles": "Sustained cycles",
              "paradox_of_enrichment": "Enrichment", "predator_starvation": "Predator starvation",
              "overexploitation": "Overexploitation", "prey_overpopulation": "Prey only",
              "vegetation_scarcity": "Vegetation scarcity"}
    changes = {"stable_coexistence": "defaults ($N{=}12$, $R{=}20$)", "lv_cycles": "$N{=}30$",
               "paradox_of_enrichment": "$N{=}30$, $R{=}10$", "predator_starvation": "$N{=}3$",
               "overexploitation": "kill radius 2, 8 kills, $N{=}40$",
               "prey_overpopulation": "no predators, fast breeding",
               "vegetation_scarcity": "30\\% fertile, $R{=}30$"}
    lines = []
    for k, lab in labels.items():
        r = g.get((k,), [])
        if not r:
            continue
        def f(v, d=0):
            return "--" if not math.isfinite(v) else (f"{v:.{d}f}")
        coex = pcoex(r)
        outc = max({o: sum(x["outcome"] == o for x in r) for o in {x["outcome"] for x in r}}.items(), key=lambda kv: kv[1])[0]
        dead = outc == "all_extinct"
        osc = [x for x in r if x["outcome"] != "all_extinct" and isinstance(x["cv_prey"], float) and x["cv_prey"] >= 0.1]
        per = mean(osc, "period_prey", np.nanmedian) if len(osc) >= len(r) / 2 else float("nan")
        ext = ""
        if dead:
            ext = f" (t$\\approx${mean(r, 'predator_extinct_tick'):.0f})"
        lines.append(f"{lab} & {changes[k]} & {outc.replace('_', ' ')}{ext} & "
                     f"{'--' if dead else f(mean(r, 'mean_prey'))} & {'--' if dead else f(mean(r, 'mean_predators'))} & "
                     f"{'--' if dead else f(mean(r, 'cv_prey'), 2)} & {f(per)} \\\\")
    (OUT / "table_regimes.tex").write_text("\n".join(lines) + "\n")
    s = g[("stable_coexistence",)]
    c = g[("lv_cycles",)]
    put("StablePrey", mean(s, "mean_prey"))
    put("StablePred", mean(s, "mean_predators"))
    put("StableCV", mean(s, "cv_prey"), "{:.2f}")
    put("CyclePrey", mean(c, "mean_prey"))
    put("CyclePred", mean(c, "mean_predators"))
    put("CycleCV", mean(c, "cv_prey"), "{:.2f}")
    put("CyclePeriod", mean(c, "period_prey", np.nanmedian))
    put("CycleLag", mean(c, "predator_lag", np.nanmedian))
    put("EnrichCV", mean(g[("paradox_of_enrichment",)], "cv_prey"), "{:.2f}")
    put("EnrichMax", mean(g[("paradox_of_enrichment",)], "max_prey"))
    put("CycleMax", mean(c, "max_prey"))
    put("ScarcePrey", mean(g[("vegetation_scarcity",)], "mean_prey"))
    put("PreyOnlyK", mean(g[("predator_starvation",)], "mean_prey"))


def e2():
    rs = rows("E2_starve_sweep.csv")
    g = group(rs, "N")
    Ns = sorted(k[0] for k in g)
    coex = [N for N in Ns if pcoex(g[(N,)]) == 1.0]
    put("NcoexMin", min(coex))
    put("NcoexMax", max(coex))
    put("NpredExt", max(N for N in Ns if N < min(coex)))
    over = [N for N in Ns if N > max(coex)]
    put("NoverMin", min(over) if over else float("nan"))
    put("NoverPcoex", pcoex(g[(min(over),)]) * 100 if over else float("nan"))
    cvs = {N: mean(g[(N,)], "cv_prey") for N in coex}
    put("CVlow", cvs[min(coex)], "{:.2f}")
    put("CVhigh", cvs[max(coex)], "{:.2f}")
    put("PredPlateau", np.mean([mean(g[(N,)], "mean_predators") for N in coex if N >= 18]))
    put("PreyNtwelve", mean(g[(12.0,)], "mean_prey"))
    put("PreyNforty", mean(g[(40.0,)], "mean_prey"))
    per = [mean([r for r in g[(N,)] if r["cv_prey"] >= 0.05], "period_prey", np.nanmedian)
           for N in coex if N >= 15]
    put("ABMperLo", np.nanmin(per))
    put("ABMperHi", np.nanmax(per))
    lag = [mean(g[(N,)], "predator_lag", np.nanmedian) for N in coex if N >= 15]
    put("LagLo", np.nanmin(lag))
    put("LagHi", np.nanmax(lag))
    fits = json.loads((DATA / "E2_fits.json").read_text())
    def eqs(model, key):
        out = []
        for N in coex:
            e = (fits.get(str(int(N)), {}).get(model, {}) or {}).get("equilibrium") or {}
            if e.get(key) is not None:
                out.append(e[key])
        return out
    for model, tag in (("chain", "Chain"), ("rm", "RM")):
        p = [v for v, N in zip(eqs(model, "period"), coex) if v]
        put(f"{tag}perLo", min(p))
        put(f"{tag}perHi", max(p))
        types = {(fits.get(str(int(N)), {}).get(model, {}).get("equilibrium") or {}).get("type", "") for N in coex}
        J[f"{tag}types"] = sorted(types)
    bs = [fits[str(int(N))]["chain"]["params"]["b"] for N in coex if "params" in fits.get(str(int(N)), {}).get("chain", {})]
    put("bLo", min(bs), "{:.1f}")
    put("bHi", max(bs), "{:.1f}")
    put("bRatio", 8 / np.mean(bs), "{:.1f}")
    # equilibrium accuracy: chain x* vs ABM mean prey
    errs = []
    for N in coex:
        e = (fits.get(str(int(N)), {}).get("chain", {}) or {}).get("equilibrium") or {}
        if e.get("prey_count"):
            errs.append(abs(e["prey_count"] - mean(g[(N,)], "mean_prey")) / mean(g[(N,)], "mean_prey"))
            errs.append(abs(e["predator_count"] - mean(g[(N,)], "mean_predators")) / mean(g[(N,)], "mean_predators"))
    put("EqErrMax", 100 * max(errs))
    put("EqErrMean", 100 * np.mean(errs), "{:.1f}")
    mf = [fits[str(int(N))]["meanfield"]["model_stats"]["prey"]["cv"] for N in coex
          if fits.get(str(int(N)), {}).get("meanfield", {}).get("model_stats")]
    put("MFcvLo", min(mf), "{:.2f}")


def e7():
    d = json.loads((DATA / "E7_lv_reports.json").read_text())
    for reg, tag in (("stable_coexistence", "S"), ("lv_cycles", "C")):
        f = d[reg]["fits"]
        put(f"{tag}classicPer", f["classic"]["equilibrium"]["period"])
        put(f"{tag}chainPer", f["chain"]["equilibrium"]["period"])
        put(f"{tag}chainTau", f["chain"]["equilibrium"]["damping_time"])
        put(f"{tag}rmTau", f["rm"]["equilibrium"]["damping_time"])
        put(f"{tag}classicCV", f["classic"]["model_stats"]["prey"]["cv"], "{:.2f}")
        put(f"{tag}mfPrey", f["meanfield"]["model_stats"]["prey"]["mean"])
        put(f"{tag}mfPred", f["meanfield"]["model_stats"]["predators"]["mean"])
        put(f"{tag}mfCV", f["meanfield"]["model_stats"]["prey"]["cv"], "{:.2f}")
        put(f"{tag}logR2", f["logistic"]["rate_r2"]["prey_growth"], "{:.2f}")
        r2 = f["chain"]["rate_r2"]
        put(f"{tag}chainR2lo", min(r2["veg_regrowth"], r2["prey_births"], r2["predation"], r2["conversion"]), "{:.2f}")
        put(f"{tag}chainR2hi", max(r2.values()), "{:.2f}")
        put(f"{tag}gR2", r2["predator_death"], "{:.2f}")
        p = f["chain"]["params"]
        for k in ("b", "c", "g", "rho", "e", "eps", "mu"):
            put(f"{tag}p{k}", p[k], "{:.3g}")
    # table of LV fits for the cycling regime
    f = d["lv_cycles"]["fits"]
    names = {"classic": "Classic LV", "logistic": "Logistic LV", "rm": "Rosenzw.--MacArthur",
             "chain": "Vegetation chain", "meanfield": "Mean-field"}
    s = d["lv_cycles"]["summary"]
    lines = [f"ABM (spatial) & -- & {s['mean_prey']:.0f} & {s['mean_predators']:.0f} & {s['cv_prey']:.2f} & {s['period_prey']:.0f} \\\\ \\midrule"]
    for k, lab in names.items():
        m = f[k]
        eq = m.get("equilibrium") or {}
        st = m["model_stats"]
        typ = {"center (neutral cycles)": "centre", "stable focus (damped oscillations)": "stable focus"}.get(eq.get("type"), eq.get("type", "--"))
        per = eq.get("period") or st["prey"]["period"]
        lines.append(f"{lab} & {typ} & {st['prey']['mean']:.0f} & {st['predators']['mean']:.0f} & "
                     f"{st['prey']['cv']:.2f} & {per:.0f} \\\\")
    (OUT / "table_lv.tex").write_text("\n".join(lines) + "\n")


def e3():
    rs = rows("E3_enrichment.csv")
    g = group(rs, "N", "R")
    for N, tag in ((12.0, "A"), (30.0, "B")):
        Rs = sorted(k[1] for k in g if k[0] == N)
        put(f"Enr{tag}cvRich", mean(g[(N, min(Rs))], "cv_prey"), "{:.2f}")
        put(f"Enr{tag}cvPoor", mean(g[(N, max(Rs))], "cv_prey"), "{:.2f}")
        put(f"Enr{tag}maxRich", mean(g[(N, min(Rs))], "max_prey"))
        put(f"Enr{tag}pRich", 100 * pcoex(g[(N, min(Rs))]))
        put(f"Enr{tag}pMin", 100 * min(pcoex(g[(N, R)]) for R in Rs))
        put(f"Enr{tag}Rworst", min(Rs, key=lambda R: pcoex(g[(N, R)])))
    put("EnrRrich", min(k[1] for k in g))
    put("EnrRpoor", max(k[1] for k in g))


def e4():
    rs = rows("E4_size.csv")
    g = group(rs, "preset", "L")
    for name, tag in (("stable_coexistence", "S"), ("lv_cycles", "C")):
        Ls = sorted(k[1] for k in g if k[0] == name)
        cv = np.array([mean([r for r in g[(name, L)] if r["outcome"] == "coexist"], "cv_prey") for L in Ls])
        ok = np.isfinite(cv)
        slope = np.polyfit(np.log(np.array(Ls)[ok] ** 2), np.log(cv[ok]), 1)[0]
        put(f"Size{tag}slope", slope, "{:.2f}")
        put(f"Size{tag}Lmin", min(Ls))
        put(f"Size{tag}Lmax", max(Ls))
        put(f"Size{tag}pSmall", 100 * pcoex(g[(name, min(Ls))]))
        put(f"Size{tag}pLarge", 100 * pcoex(g[(name, max(Ls))]))
        put(f"Size{tag}cvSmall", cv[0], "{:.2f}")
        put(f"Size{tag}cvLarge", cv[-1], "{:.2f}")
        allok = [L for L in Ls if pcoex(g[(name, L)]) == 1.0]
        put(f"Size{tag}Lsafe", min(allok) if allok else float("nan"))


def e5():
    ar = json.loads((DATA / "E5_archive.json").read_text())
    best = json.loads((DATA / "E5_best.json").read_text())
    cfg = json.loads((DATA / "E5_config.json").read_text())
    gens = list(csv.DictReader((DATA / "E5_generations.csv").open()))
    (xn, xl, xh, xb), (yn, yl, yh, yb) = ar["axes"]["x"], ar["axes"]["y"]
    put("QDcells", len(ar["cells"]))
    put("QDtotal", int(xb * yb))
    put("QDevals", int(gens[-1]["evals"]))
    put("QDgens", int(gens[-1]["gen"]))
    put("QDpop", cfg["pop"])
    put("QDbest", best["validated_fitness"], "{:.3f}")
    put("QDbestCV", best["metrics"]["cv_prey"], "{:.3f}")
    put("QDbestPrey", best["metrics"]["mean_prey"])
    put("QDbestPred", best["metrics"]["mean_predators"])
    put("QDbestKills", best["params"]["predator_max_kills"])
    cells = list(ar["cells"].values())
    lo = [e["fitness"] for e in cells if (e["metrics"]["cv_prey"] or 0) < 0.125]
    hi = [e["fitness"] for e in cells if (e["metrics"]["cv_prey"] or 0) >= 1.25]
    put("QDfitLowCV", max(lo), "{:.2f}")
    put("QDfitHighCV", max(hi) if hi else float("nan"), "{:.2f}")
    put("QDminRepro", min(e["params"]["predator_repro_prob"] for e in cells), "{:.3f}")
    osc = ar["cells"].get("10,6")
    if osc:
        m = osc["metrics"]
        put("OscCV", m["cv_prey"], "{:.2f}")
        put("OscPer", m["period_prey"])
        put("OscMax", m["max_prey"])
        put("OscMin", m["min_prey_after_transient"])
        put("OscFit", osc["fitness"], "{:.2f}")
    y = DATA / "E5_yield_best.json"
    if y.exists():
        yb_ = json.loads(y.read_text())
        p, m = yb_["params"], yb_["metrics"]
        mode = {1: "constant effort", 2: "constant quota", 3: "proportional feedback"}[int(p["harvest_mode"])]
        put("YmodeName", mode)
        put("Yyield", m["yield_prey"], "{:.2f}")
        put("Yval", yb_["validated_fitness"], "{:.2f}")
        put("YmeanPred", m["mean_predators"])
        yg = list(csv.DictReader((DATA / "E5_yield_generations.csv").open()))
        put("Ytrain", max(float(r["best"]) for r in yg), "{:.2f}")
        put("Ygens", int(yg[-1]["gen"]))
        put("Ycoex", 100 * m["coexist_frac"])
        put("YmeanPrey", m["mean_prey"])
        put("Yeffort", p["harvest_effort"], "{:.3f}")
        put("Yquota", p["harvest_quota"])
        put("Ysetpoint", p["harvest_setpoint"])
        put("Ygain", p["harvest_gain"], "{:.2f}")
        modes = defaultdict(list)
        for t in yb_["top"]:
            modes[int(t["params"]["harvest_mode"])].append(t["validated_fitness"])
        J["yield_top_modes"] = {k: v for k, v in modes.items()}
        # all evaluations by mode: best mean yield among coexisting
        ev = list(csv.DictReader((ROOT / "runs/evolve/yield_v1/evals.csv").open()))
        bym = defaultdict(float)
        for r in ev:
            if float(r["coexist_frac"] or 0) == 1.0:
                md = int(float(r["harvest_mode"]))
                bym[md] = max(bym[md], float(r["yield_prey"] or 0))
        for md, nm in ((1, "Eff"), (2, "Quo"), (3, "Fb")):
            put(f"Ybest{nm}", bym.get(md, float("nan")), "{:.2f}")
        put("Yevals", len(ev) * 3)


def e6a():
    rs = rows("E6a_effort.csv")
    g = group(rs, "species", "effort")
    for sp, tag, yk in (("prey", "X", "yield_prey"), ("predators", "Y", "yield_predators")):
        E = sorted(k[1] for k in g if k[0] == sp)
        ok = [e for e in E if pcoex(g[(sp, e)]) == 1.0]
        ys = {e: mean(g[(sp, e)], yk) for e in ok}
        emsy = max(ys, key=ys.get)
        put(f"MSY{tag}effort", emsy, "{:.3f}")
        put(f"MSY{tag}yield", ys[emsy], "{:.1f}")
        bad = [e for e in E if pcoex(g[(sp, e)]) < 1.0]
        put(f"Ecoll{tag}", min(bad) if bad else float("nan"), "{:.3f}")
        yall = {e: mean(g[(sp, e)], yk) for e in E}
        efree = max(yall, key=yall.get)
        put(f"Free{tag}effort", efree, "{:.3f}")
        put(f"Free{tag}yield", yall[efree], "{:.0f}")
        put(f"Free{tag}collapse", min([e for e in E if yall[e] < 0.2 * yall[efree] and e > efree] or [float("nan")]), "{:.2f}")
        put(f"MSY{tag}prey", mean(g[(sp, emsy)], "mean_prey"))
        put(f"MSY{tag}pred", mean(g[(sp, emsy)], "mean_predators"))
    base = g[("prey", 0.0)]
    put("HzeroPrey", mean(base, "mean_prey"))
    put("HzeroPred", mean(base, "mean_predators"))
    E = sorted(k[1] for k in g if k[0] == "both")
    ok = [e for e in E if pcoex(g[("both", e)]) == 1.0 and e > 0]
    e1 = 0.01 if 0.01 in ok else ok[len(ok) // 2]
    put("VoltE", e1, "{:.3f}")
    put("VoltPrey", mean(g[("both", e1)], "mean_prey"))
    put("VoltPred", mean(g[("both", e1)], "mean_predators"))
    ep = 0.01
    put("PreyHprey", mean(g[("prey", ep)], "mean_prey"))
    put("PreyHpred", mean(g[("prey", ep)], "mean_predators"))
    put("PredHprey", mean(g[("predators", ep)], "mean_prey"))
    put("PredHpred", mean(g[("predators", ep)], "mean_predators"))


def e6b():
    rs = rows("E6b_quota.csv")
    g = group(rs, "species", "quota")
    for sp, tag in (("prey", "X"), ("predators", "Y")):
        Q = sorted(k[1] for k in g if k[0] == sp)
        ok = [q for q in Q if pcoex(g[(sp, q)]) == 1.0]
        put(f"Qmax{tag}", max(ok))
        bad = [q for q in Q if q > max(ok)]
        put(f"Qcoll{tag}", min(bad) if bad else float("nan"))
        put(f"Qcoll{tag}p", 100 * pcoex(g[(sp, min(bad))]) if bad else float("nan"))
        put(f"Qmax{tag}prey", mean(g[(sp, max(ok))], "mean_prey"))
        put(f"Qmax{tag}pred", mean(g[(sp, max(ok))], "mean_predators"))


def e6c():
    rs = rows("E6c_feedback.csv")
    g = group([r for r in rs if r["controlled"] == "predators"], "setpoint", "gain")
    for spv, tag in ((400.0, "A"), (600.0, "B")):
        put(f"FB{tag}cvZero", mean(g[(spv, 0.0)], "cv_predators"), "{:.2f}")
        put(f"FB{tag}cvpZero", mean(g[(spv, 0.0)], "cv_prey"), "{:.2f}")
        put(f"FB{tag}cvTwo", mean(g[(spv, 0.2)], "cv_predators"), "{:.2f}")
        put(f"FB{tag}cvpTwo", mean(g[(spv, 0.2)], "cv_prey"), "{:.2f}")
        put(f"FB{tag}cvOne", mean(g[(spv, 1.0)], "cv_predators"), "{:.2f}")
        put(f"FB{tag}cvpOne", mean(g[(spv, 1.0)], "cv_prey"), "{:.2f}")
        put(f"FB{tag}meanOne", mean(g[(spv, 1.0)], "mean_predators"))
        put(f"FB{tag}meanTwo", mean(g[(spv, 0.2)], "mean_predators"))
        put(f"FB{tag}meanSmall", mean(g[(spv, 0.02)], "mean_predators"))
        put(f"FB{tag}yieldTwo", mean(g[(spv, 0.2)], "yield_predators"), "{:.1f}")
        put(f"FB{tag}p", 100 * min(pcoex(g[(spv, k)]) for k in {kk[1] for kk in g if kk[0] == spv}))
    put("FBsetA", 400)
    put("FBsetB", 600)
    gx = group([r for r in rs if r["controlled"] == "prey"], "setpoint", "gain")
    put("FBpreyCVzero", mean(gx[(800.0, 0.0)], "cv_prey"), "{:.2f}")
    put("FBpreyCVTwo", mean(gx[(800.0, 0.2)], "cv_prey"), "{:.2f}")
    put("FBpreyPredTwo", mean(gx[(800.0, 0.2)], "cv_predators"), "{:.2f}")
    put("FBpreyP", 100 * min(pcoex(v) for v in gx.values()))


def e6d():
    rs = rows("E6d_open_vs_closed.csv")
    g = group(rs, "controller", "value")
    eff = sorted([(mean(g[k], "yield_predators"), mean(g[k], "cv_prey"), k[1]) for k in g if k[0] == "effort"])
    fb = sorted([(mean(g[k], "yield_predators"), mean(g[k], "cv_prey"), k[1]) for k in g if k[0] == "feedback"])
    # compare at matched yields by interpolation over the common range
    ex, ey = np.array([e[0] for e in eff]), np.array([e[1] for e in eff])
    fx, fy = np.array([f[0] for f in fb]), np.array([f[1] for f in fb])
    lo, hi = max(ex.min(), fx.min()), min(ex.max(), fx.max())
    if hi > lo:
        grid = np.linspace(lo, hi, 20)
        r = np.interp(grid, fx, fy) / np.interp(grid, ex, ey)
        put("ODratio", float(np.mean(r)), "{:.2f}")
        put("ODyLo", lo, "{:.1f}")
        put("ODyHi", hi, "{:.1f}")
    put("ODeffCVmin", min(e[1] for e in eff), "{:.2f}")
    put("ODfbCVmin", min(f[1] for f in fb), "{:.2f}")


def e8():
    z = np.load(DATA / "E8_shocks.npz")
    d = json.loads((DATA / "E7_lv_reports.json").read_text())
    t0 = 200  # index of shock (series starts at tick 1300, shock at 1500)
    for regime, tag in (("stable", "S"), ("cycles", "C")):
        dev = z[f"{regime}__pred_minus50__predators"].mean(0) - z[f"{regime}__none__predators"].mean(0)
        devx = z[f"{regime}__pred_minus50__prey"].mean(0) - z[f"{regime}__none__prey"].mean(0)
        a = dev[t0:]
        # e-folding: time until the envelope (running max of |dev| over future) falls below |a0|/e
        env = np.array([np.abs(a[i:]).max() for i in range(len(a))])
        k = np.nonzero(env < abs(a[0]) / math.e)[0]
        put(f"Shock{tag}tau", float(k[0]) if len(k) else float("nan"))
        put(f"Shock{tag}drop", -a[0])
        put(f"Shock{tag}preyPeak", devx[t0:t0 + 150].max())
        put(f"Shock{tag}preyPeakT", float(np.argmax(devx[t0:t0 + 150])))
        put(f"Shock{tag}n", z[f"{regime}__pred_minus50__predators"].shape[0])
    put("ShockModelTau", d["stable_coexistence"]["fits"]["chain"]["equilibrium"]["damping_time"])


def main():
    for fn in (e1, e2, e7, e3, e4, e5, e6a, e6b, e6c, e6d, e8):
        safe(fn)
    tex = ["% generated by experiments/numbers.py -- do not edit"]
    for k, v in sorted(M.items()):
        name = k.replace("R2", "Rsq")
        assert name.isalpha(), name
        tex.append(f"\\newcommand{{\\{name}}}{{{v}}}")
    (OUT / "numbers.tex").write_text("\n".join(tex) + "\n")
    (OUT / "data" / "numbers.json").write_text(json.dumps(J, indent=1, default=str))
    print(f"{len(M)} numbers -> paper/numbers.tex")
    for k, v in sorted(M.items()):
        print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
