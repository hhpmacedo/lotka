import numpy as np
import pytest

from lotka.batch import dominant_period, main as batch_main
from lotka.sim import EMPTY, PREDATOR, PREY, Params, Simulation, box_sum


def small(**kw):
    base = dict(width=30, height=30, seed=1)
    base.update(kw)
    return Params.from_dict(base)


def test_deterministic_with_seed():
    a, b = Simulation(small()), Simulation(small())
    a.step(50)
    b.step(50)
    assert np.array_equal(a.animal, b.animal)
    assert np.array_equal(a.veg, b.veg)
    assert a.history == b.history


def test_initial_counts():
    s = Simulation(small(init_prey=0.1, init_predator=0.05))
    assert s.history[0]["prey"] == 90
    assert s.history[0]["predators"] == 45


def test_counts_consistent_with_events():
    s = Simulation(small(width=60, height=60))
    for _ in range(100):
        before = s.history[-1]
        s.step()
        h = s.history[-1]
        assert h["prey"] == (before["prey"] + h["prey_births"] - h["prey_eaten"] - h["prey_starved"]
                             - h["prey_old_age"] - h["harvested_prey"])
        assert h["predators"] == (before["predators"] + h["predator_births"] - h["predator_starved"]
                                  - h["predator_old_age"] - h["harvested_predators"])
        # state arrays stay clean on empty cells
        empty = s.animal == EMPTY
        assert (s.hunger[empty] == 0).all() and (s.age[empty] == 0).all()


def test_counts_consistent_with_harvest():
    s = Simulation(small(width=60, height=60, harvest_mode=1, harvest_species=3, harvest_effort=0.05))
    for _ in range(60):
        before = s.history[-1]
        s.step()
        h = s.history[-1]
        assert h["prey"] == (before["prey"] + h["prey_births"] - h["prey_eaten"] - h["prey_starved"]
                             - h["prey_old_age"] - h["harvested_prey"])
        assert h["predators"] == (before["predators"] + h["predator_births"] - h["predator_starved"]
                                  - h["predator_old_age"] - h["harvested_predators"])


def test_harvest_modes():
    # quota: exactly Q per tick while available
    s = Simulation(small(width=60, height=60, harvest_mode=2, harvest_species=1, harvest_quota=7))
    s.step()
    assert s.history[-1]["harvested_prey"] == 7 and s.history[-1]["harvested_predators"] == 0
    # feedback: removes Kp*(N - setpoint), never negative
    s = Simulation(small(width=60, height=60, harvest_mode=3, harvest_species=2, harvest_gain=1.0,
                         harvest_setpoint=10))
    s.step()
    assert s.history[-1]["predators"] <= 10 + s.history[-1]["predator_births"]
    # start tick
    s = Simulation(small(harvest_mode=1, harvest_effort=1.0, harvest_start=5))
    s.step(4)
    assert sum(h["harvested_prey"] for h in s.history) == 0
    s.step(1)
    assert s.history[-1]["prey"] == 0


def test_predators_starve_exactly_after_N_ticks():
    for N in (1, 3, 7):
        s = Simulation(small(init_prey=0, init_predator=0.05, predator_starve_time=N))
        s.step(N - 1)
        assert s.history[-1]["predators"] > 0
        s.step()
        assert s.history[-1]["predators"] == 0
        assert s.history[-1]["predator_starved"] == 45


def test_prey_starve_after_S_ticks_without_food():
    S = 4
    s = Simulation(small(init_prey=0.05, init_predator=0, fertile_fraction=0, prey_starve_time=S))
    s.step(S - 1)
    assert s.history[-1]["prey"] == 45
    s.step()
    assert s.history[-1]["prey"] == 0


def test_prey_need_K_vegetation_to_be_fed():
    # one isolated prey, exactly 3 veg cells in reach, needs 4 -> never fed
    p = small(init_prey=0, init_predator=0, init_vegetation=0, veg_regrow_time=1000,
              prey_food_need=4, prey_directed_prob=0, prey_repro_prob=0)
    s = Simulation(p)
    s.veg[:] = False
    s.veg_timer[:] = 0
    s.animal[10, 10] = PREY
    s._move = lambda *a, **k: None  # keep it in place
    s.veg[9, 9] = s.veg[10, 10] = s.veg[11, 11] = True
    s.step()
    assert s.history[-1]["prey_fed_frac"] == 0
    assert s.veg.sum() == 0  # it still ate what it could
    assert s.hunger[10, 10] == 1


def test_predator_kills_prey_in_vicinity():
    p = small(init_prey=0, init_predator=0, init_vegetation=1, predator_repro_prob=0,
              prey_repro_prob=0, predator_max_kills=8)
    s = Simulation(p)
    s._move = lambda *a, **k: None
    s.animal[10, 10] = PREDATOR
    for dy, dx in [(-1, 0), (1, 1), (0, -1)]:
        s.animal[10 + dy, 10 + dx] = PREY
    s.animal[20, 20] = PREY  # far away, survives
    s.step()
    h = s.history[-1]
    assert h["prey_eaten"] == 3 and h["prey"] == 1
    assert s.hunger[10, 10] == 0


def test_max_kills_satiation():
    p = small(init_prey=0, init_predator=0, init_vegetation=1, predator_repro_prob=0,
              prey_repro_prob=0, predator_max_kills=1)
    s = Simulation(p)
    s._move = lambda *a, **k: None
    s.animal[10, 10] = PREDATOR
    s.animal[9, 10] = s.animal[11, 10] = PREY
    s.step()
    assert s.history[-1]["prey_eaten"] == 1


def test_vegetation_regrows_after_R():
    R = 5
    s = Simulation(small(init_prey=0, init_predator=0, init_vegetation=0, veg_regrow_time=R))
    s.veg_timer[:] = 0
    s.veg[:] = False
    s.step(R - 1)
    assert s.veg.sum() == 0
    s.step()
    assert s.veg.all()


def test_box_sum_wraps():
    a = np.zeros((5, 5))
    a[0, 0] = 1
    b = box_sum(a, 1)
    assert b[4, 4] == 1 and b[1, 1] == 1 and b[2, 2] == 0
    assert b.sum() == 9


def test_dominant_period():
    t = np.arange(1000)
    assert abs(dominant_period(np.sin(2 * np.pi * t / 50)) - 50) <= 1


def test_params_from_dict_types():
    p = Params.from_dict({"width": "50", "prey_repro_prob": "0.2", "prey_repro_requires_fed": "false"})
    assert p.width == 50 and p.prey_repro_prob == 0.2 and p.prey_repro_requires_fed is False


def test_batch_runs(tmp_path):
    batch_main(["--ticks", "30", "--seeds", "0,1", "--set", "width=30", "--set", "height=30",
                "--sweep", "predator_starve_time=3,10", "--workers", "1", "--out", str(tmp_path)])
    rows = (tmp_path / "summary.csv").read_text().strip().splitlines()
    assert len(rows) == 5
    assert len(list((tmp_path / "series").glob("*.csv"))) == 4


def test_batch_rejects_unknown_param(tmp_path):
    with pytest.raises(SystemExit):
        batch_main(["--set", "nope=1", "--out", str(tmp_path)])


def test_perturb():
    s = Simulation(small(width=50, height=50))
    n0 = s.history[-1]["prey"]
    s.perturb(PREY, -0.5)
    assert s.history[-1]["prey"] == n0 - round(0.5 * n0)
    n1 = s.history[-1]["predators"]
    s.perturb(PREDATOR, 1.0)
    assert s.history[-1]["predators"] == 2 * n1
    assert (s.hunger[s.animal == EMPTY] == 0).all()
