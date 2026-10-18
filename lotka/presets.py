"""Named parameter presets ("templates"). Each preset only lists differences from Params defaults.

Expected outcomes were checked on the default 200x200 grid, 2000 ticks, seeds 0-2
(scripts/explore_presets.py); stochastic runs can occasionally differ.
"""

PRESETS: dict[str, dict] = {
    "stable_coexistence": {
        "label": "Stable coexistence",
        "desc": "Default rules. Satiated predators (1 kill/tick) and fleeing prey settle into a "
                "noisy equilibrium: ~2,100 prey, ~500 predators, small fluctuations (CV ~6-10%).",
        "params": {},
    },
    "lv_cycles": {
        "label": "Lotka-Volterra cycles",
        "desc": "Predators survive 30 ticks without food, so they overshoot: sustained prey/predator "
                "cycles of ~140 ticks with predator peaks lagging prey peaks (CV ~25-35%).",
        "params": {"predator_starve_time": 30},
    },
    "paradox_of_enrichment": {
        "label": "Paradox of enrichment",
        "desc": "Same as LV cycles but vegetation regrows twice as fast (R=10). More food makes the "
                "system LESS stable: prey booms to 6-8k, oscillations double (CV ~70%).",
        "params": {"predator_starve_time": 30, "veg_regrow_time": 10},
    },
    "predator_starvation": {
        "label": "Predator starvation",
        "desc": "Predators die after 3 ticks without food and can't find prey fast enough: extinct "
                "by tick ~25, then prey grow to vegetation carrying capacity (~3,500).",
        "params": {"predator_starve_time": 3},
    },
    "overexploitation": {
        "label": "Overexploitation (predators eat everything)",
        "desc": "Super-efficient hunters (radius 2, up to 8 kills, always chase, prey don't flee): "
                "prey wiped out by tick ~25, predators starve by tick ~65. Total collapse.",
        "params": {"predator_max_kills": 8, "predator_kill_radius": 2, "predator_repro_prob": 0.8,
                   "predator_starve_time": 40, "predator_directed_prob": 1.0,
                   "prey_fear": 0, "prey_directed_prob": 0.3},
    },
    "prey_overpopulation": {
        "label": "Prey overpopulation (no predators)",
        "desc": "No predators, fast-breeding prey: boom to ~11k, vegetation is grazed out, mass "
                "starvation, then boom-bust cycles of ~40 ticks around ~3,500.",
        "params": {"init_predator": 0, "prey_repro_prob": 0.3, "prey_starve_time": 8,
                   "veg_regrow_time": 40},
    },
    "overpopulation_collapse": {
        "label": "Overpopulation collapse",
        "desc": "Very fertile prey and slow-regrowing vegetation: prey explode to ~13k, predators "
                "can't keep up and die out in the famine (~tick 170); later the prey crash to extinction.",
        "params": {"init_predator": 0.005, "prey_repro_prob": 0.4, "prey_starve_time": 10,
                   "veg_regrow_time": 60},
    },
    "vegetation_scarcity": {
        "label": "Vegetation scarcity (bottom-up control)",
        "desc": "Only 30% of land is fertile and regrowth is slow: prey stay sparse (~420), too "
                "sparse to feed predators, which go extinct by tick ~50.",
        "params": {"fertile_fraction": 0.3, "veg_regrow_time": 30},
    },
}


def preset_params(name: str) -> dict:
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {list(PRESETS)}")
    return dict(PRESETS[name]["params"])


def load_params_file(path: str) -> dict:
    """Parameters from a JSON file: an evolve best.json / archive cell, or a plain {param: value} dict."""
    import json
    from pathlib import Path
    d = json.loads(Path(path).read_text())
    return dict(d.get("params", d))
