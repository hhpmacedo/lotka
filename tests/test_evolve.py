import json

import numpy as np
import pytest

from lotka import evolve
from lotka.evolve import DEFAULT_GENES, Genome, safe_eval


def test_genome_roundtrip_and_types():
    g = Genome.from_spec(DEFAULT_GENES)
    rng = np.random.default_rng(0)
    for _ in range(20):
        d = g.decode(rng.random(len(g.names)))
        assert isinstance(d["predator_starve_time"], int)
        assert isinstance(d["prey_repro_requires_fed"], bool)
        for k, (lo, hi) in DEFAULT_GENES.items():
            assert lo <= float(d[k]) <= hi
        assert g.decode(g.encode(d)) == d


def test_safe_eval():
    m = {"mean_predators": 500, "cells": 10000, "cv_prey": 0.2}
    assert safe_eval("mean_predators/cells*100 - 5*cv_prey", m) == pytest.approx(4.0)
    assert safe_eval("max(cv_prey, 1) + abs(-2)", m) == 3
    for bad in ("__import__('os')", "cells.__class__", "[1,2]", "open('x')"):
        with pytest.raises(ValueError):
            safe_eval(bad, m)


def test_viability_gate_orders_failures_below_successes():
    f = evolve.OBJECTIVES["msy"].per_seed
    ok = {"coexist": 1.0, "survival": 1.0, "density_predators": 0.0}
    dead = {"coexist": 0.0, "survival": 0.99}
    assert f(ok, {}) >= 0 > f(dead, {})


@pytest.mark.parametrize("objective,extra", [
    ("coexistence", []), ("qd", []), ("custom", ["--arg", "expr=mean_prey/cells"]),
])
def test_small_evolution_run_and_resume(tmp_path, objective, extra):
    out = tmp_path / objective
    base = ["--objective", objective, "--pop", "6", "--gens", "1", "--ticks", "60", "--size", "24",
            "--seeds", "2", "--workers", "1", "--validate-seeds", "2", "--validate-top", "2", "--out", str(out)]
    evolve.main(base + extra)
    for f in ("config.json", "generations.csv", "evals.csv", "checkpoint.json", "best.json"):
        assert (out / f).exists(), f
    best = json.loads((out / "best.json").read_text())
    assert best["params"]["width"] == 24 and "validated_fitness" in best
    evolve.main(["--resume", str(out), "--gens", "2", "--workers", "1"])
    gens = (out / "generations.csv").read_text().strip().splitlines()
    assert len(gens) == 1 + 3  # header + gen 0,1,2


def test_immortal_non_breeding_predators_are_not_coexistence():
    params = dict(width=40, height=40, predator_repro_prob=0.0, predator_starve_time=1000,
                  prey_max_age=0, predator_max_age=0)
    job = {"params": params, "seed": 0, "ticks": 300, "stop_on_pred": True, "lv_model": None}
    m = evolve.evaluate({**job, "require_turnover": True})
    assert m["outcome"] == "coexist" and m["static"] == 1.0 and m["coexist"] == 0.0
    assert evolve.OBJECTIVES["msy"].per_seed(m, {}) == -0.5
    m2 = evolve.evaluate({**job, "require_turnover": False})
    assert m2["coexist"] == 1.0


def test_default_lifespans_applied(tmp_path):
    out = tmp_path / "r"
    evolve.main(["--pop", "4", "--gens", "0", "--ticks", "30", "--size", "20", "--seeds", "1",
                 "--workers", "1", "--validate-seeds", "1", "--validate-top", "1", "--out", str(out)])
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["fixed"]["prey_max_age"] == 150 and cfg["fixed"]["predator_max_age"] == 300
    assert cfg["require_turnover"] is True
    out2 = tmp_path / "r2"
    evolve.main(["--pop", "4", "--gens", "0", "--ticks", "30", "--size", "20", "--seeds", "1", "--immortal",
                 "--workers", "1", "--validate-seeds", "1", "--validate-top", "1", "--out", str(out2)])
    assert "prey_max_age" not in json.loads((out2 / "config.json").read_text())["fixed"]
