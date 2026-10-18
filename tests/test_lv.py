import math

import numpy as np
import pytest

from lotka import lv
from lotka.presets import PRESETS
from lotka.sim import Params, Simulation


def synthetic_history(model, p, T=400, cells=10_000, x0=0.05, y0=0.01):
    """Exact Euler-map history with event counts produced by the model itself."""
    hist = [dict(tick=0, prey=x0 * cells, predators=y0 * cells, vegetation=0)]
    x, y = x0, y0
    for t in range(1, T + 1):
        F = lv._F(model, p, x)
        G = lv._growth(model, p, x)
        eaten, births_y, deaths_y = F * y, p["c"] * F * y, p["g"] * y
        hist.append(dict(tick=t, prey=0, predators=0, vegetation=0,
                         prey_eaten=eaten * cells, prey_births=max(G, 0) * cells,
                         prey_starved=max(-G, 0) * cells,
                         predator_births=births_y * cells, predator_starved=deaths_y * cells))
        x, y = x + G - eaten, y + births_y - deaths_y
        hist[-1]["prey"], hist[-1]["predators"] = x * cells, y * cells
    return hist


@pytest.mark.parametrize("model,p", [
    ("classic", dict(a=0.05, b=2.0, c=0.3, g=0.02)),
    ("logistic", dict(r=0.06, K=0.1, b=2.0, c=0.3, g=0.02)),
    ("rm", dict(r=0.06, K=0.1, a=3.0, h=2.0, c=0.3, g=0.02)),
])
def test_fit_recovers_known_rates(model, p):
    f = lv.fit_rates(synthetic_history(model, p), 10_000, model)
    for k, v in p.items():
        assert f["params"][k] == pytest.approx(v, rel=1e-3), k


def test_classic_equilibrium_is_center_with_known_period():
    p = dict(a=0.05, b=2.0, c=0.3, g=0.02)
    eq = lv.equilibrium("classic", p)
    assert eq["x"] == pytest.approx(0.02 / 0.6)
    assert eq["y"] == pytest.approx(0.05 / 2.0)
    assert eq["type"].startswith("center")
    assert eq["period"] == pytest.approx(2 * math.pi / math.sqrt(0.05 * 0.02), rel=1e-4)


def test_rm_equilibrium_is_fixed_point():
    p = dict(r=0.065, K=0.08, a=3.2, h=3.7, c=0.3, g=0.017)
    eq = lv.equilibrium("rm", p)
    assert eq["exists"]
    assert np.allclose(lv.rhs("rm", p, np.array([eq["x"], eq["y"]])), 0, atol=1e-12)


def test_chain_equilibrium_is_fixed_point():
    p = dict(V=1.0, rho=0.05, e=1.4, eps=0.08, mu=0.02, b=2.4, c=0.3, g=0.017)
    eq = lv.equilibrium("chain", p)
    v = p["rho"] * p["V"] / (p["rho"] + p["e"] * eq["x"])
    assert np.allclose(lv.rhs("chain", p, np.array([v, eq["x"], eq["y"]])), 0, atol=1e-12)


def test_classic_integration_conserves_lv_invariant():
    p = dict(a=0.05, b=2.0, c=0.3, g=0.02)
    tr = lv.integrate("classic", p, (0.05, 0.01), 500, dt=0.01)

    def V(x, y):  # conserved quantity of classic LV
        return p["c"] * p["b"] * x - p["g"] * math.log(x) + p["b"] * y - p["a"] * math.log(y)
    assert V(*tr[-1]) == pytest.approx(V(*tr[0]), rel=1e-6)


def test_meanfield_prey_starve_after_S():
    p = Params.from_dict(dict(prey_starve_time=4, init_predator=0))
    tr = lv.meanfield(p, 6, 0.05, 0.0, 0.0, 0.0)
    assert tr[3, 0] > 0 and tr[4, 0] == pytest.approx(0)


@pytest.mark.parametrize("model", list(lv.MODELS))
def test_compare_runs_on_sim(model):
    s = Simulation(Params.from_dict(dict(width=40, height=40, seed=3)))
    s.step(150)
    r = lv.compare(s, model, fit_start=20, extend=10)
    assert len(r["traj"]["prey"]) == 161
    assert r["abm_stats"]["prey"]["mean"] > 0


def test_presets_use_valid_params():
    valid = {f["name"] for f in Params.schema()}
    for name, pr in PRESETS.items():
        assert set(pr["params"]) <= valid, name
        Simulation(Params.from_dict({**pr["params"], "width": 20, "height": 20})).step(3)
