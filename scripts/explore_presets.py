"""Explore candidate presets: run each across seeds, print outcomes."""
import sys
from concurrent.futures import ProcessPoolExecutor

from lotka.batch import summarize
from lotka.sim import Params, Simulation

CANDS = {
    "stable_a": dict(),
    "stable_b": dict(prey_fear=10, prey_directed_prob=0.9, predator_starve_time=10, predator_repro_prob=0.2),
    "cycles_a": dict(predator_starve_time=30),
    "cycles_b": dict(predator_starve_time=40, predator_repro_prob=0.5),
    "pred_starve_a": dict(predator_starve_time=3),
    "pred_starve_b": dict(predator_starve_time=5, predator_directed_prob=0.0),
    "overexploit_a": dict(predator_max_kills=8, predator_kill_radius=2, predator_repro_prob=0.8,
                          predator_starve_time=40, predator_directed_prob=1.0, prey_fear=0, prey_directed_prob=0.3),
    "overexploit_b": dict(predator_max_kills=3, predator_repro_prob=0.9, predator_starve_time=60,
                          predator_directed_prob=1.0, predator_sense_radius=8, prey_fear=0),
    "overpop_a": dict(init_predator=0, prey_repro_prob=0.3, prey_food_need=2, veg_regrow_time=60, prey_starve_time=3),
    "overpop_b": dict(prey_repro_prob=0.5, prey_repro_requires_fed=False, prey_food_need=2, veg_regrow_time=50,
                      prey_starve_time=4, predator_starve_time=8),
    "enrich_a": dict(veg_regrow_time=5, predator_starve_time=30),
    "overpop_c": dict(init_predator=0, prey_repro_prob=0.3, prey_starve_time=8, veg_regrow_time=40),
    "overpop_d": dict(init_predator=0.005, prey_repro_prob=0.4, prey_starve_time=10, veg_regrow_time=60),
    "overpop_e": dict(init_predator=0, prey_repro_prob=0.25, prey_starve_time=12, veg_regrow_time=80, prey_food_need=2),
    "enrich_b": dict(veg_regrow_time=10, predator_starve_time=30),
    "enrich_c": dict(veg_regrow_time=10, predator_starve_time=20),
    "scarce_b": dict(fertile_fraction=0.3, veg_regrow_time=30),
    "scarce_c": dict(fertile_fraction=0.4, veg_regrow_time=40),
    "scarce_a": dict(fertile_fraction=0.3, veg_regrow_time=60, prey_food_need=2, prey_starve_time=3),
}


def run(args):
    name, kw, seed, ticks = args
    p = Params.from_dict({**kw, "seed": seed})
    s = Simulation(p)
    for _ in range(ticks):
        s.step()
        if s.extinct:
            break
    r = summarize(s.history, ticks)
    return name, seed, r


if __name__ == "__main__":
    names = sys.argv[1:] or list(CANDS)
    jobs = [(n, CANDS[n], seed, 2000) for n in names for seed in (0, 1, 2)]
    with ProcessPoolExecutor(8) as ex:
        for name, seed, r in ex.map(run, jobs):
            print(f"{name:15s} s{seed} {r['outcome']:18s} ext(prey={r['prey_extinct_tick']},pred={r['predator_extinct_tick']}) "
                  f"mean prey={r['mean_prey']} pred={r['mean_predators']} cv={r['cv_prey']},{r['cv_predators']} "
                  f"max prey={r['max_prey']} per={r['period_prey']}")
