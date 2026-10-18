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
