# Lotka — agent-based predator / prey simulation

A spatial, agent-based version of the Lotka–Volterra predator–prey system.
Every pixel of a toroidal grid has a **vegetation layer** (grass or regrowing
soil) and an **animal layer** (empty, prey, or predator).

## Quick start

    cd ~/projects/lotka
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once
    .venv/bin/python -m lotka serve           # dashboard on http://127.0.0.1:8765
    .venv/bin/python -m pytest -q             # tests

Two levels of use (`python -m lotka` prints the full list):

| level | command | what it does |
|---|---|---|
| simulation | `lotka serve` | live dashboard: grid, populations, phase portrait, presets, LV comparison |
| simulation | `lotka batch` | headless runs and parameter sweeps → CSV |
| simulation | `lotka compare` | ABM vs five Lotka–Volterra variants → PNG + JSON report |
| optimisation | `lotka evolve` | genetic algorithm / MAP-Elites over parameter sets, objective chosen per run |

Dashboard: Space = play/pause. Pick a **Preset** (template) or edit parameters on the left, then
"Apply & reset". "Export CSV" downloads the per-tick metrics of the current run. The bottom panel
fits a Lotka–Volterra model to the running simulation and plots it side by side.

## Presets (templates)

Defined in `lotka/presets.py`; usable in the dashboard, `batch --preset` and `compare --preset`.
Outcomes checked on 200×200, 2000 ticks, 3 seeds.

| preset | what happens |
|---|---|
| `stable_coexistence` | defaults: noisy equilibrium ~2,100 prey / ~500 predators, CV 6–10% |
| `lv_cycles` | predators survive 30 ticks unfed → sustained ~140-tick cycles |
| `paradox_of_enrichment` | as cycles but vegetation regrows 2× faster → oscillations double |
| `predator_starvation` | predators starve (N=3), extinct by ~tick 25; prey hit carrying capacity |
| `overexploitation` | super-hunters eat all prey (~tick 25) then starve (~tick 65) |
| `prey_overpopulation` | no predators, fast breeding: boom to 11k, famine, boom-bust cycles |
| `overpopulation_collapse` | prey explode, predators die in the famine, prey later go extinct |
| `vegetation_scarcity` | 30% fertile land: sparse prey can't feed predators |

## Lotka–Volterra comparison

`lotka/lv.py` fits classic LV, logistic LV, Rosenzweig–MacArthur, a vegetation–prey–predator chain,
and a rule-derived mean-field model to the ABM, using the per-tick event counts. Theory, the
parameter mapping and findings are in [docs/lotka_volterra.md](docs/lotka_volterra.md).

    .venv/bin/python -m lotka compare --preset lv_cycles --ticks 3000 --out runs/cmp_cycles

## Evolution (optimisation level)

Each "DNA" is a parameter set (18 genes by default: vegetation, prey and predator traits, initial
densities). Each genome is simulated on several seeds (same seeds for all genomes), scored by the
chosen objective, and the best are re-validated on 10 fresh seeds.

| objective | goal |
|---|---|
| `coexistence` | robust persistence of both species + resilience (highest lowest-population) |
| `target_cycles` | sustained cycles with a given period/amplitude (`--arg period=150 --arg cv=0.3`) |
| `msy` | maximum sustainable density of a species while both coexist (`--arg species=predators\|prey\|total`) |
| `amplitude` | largest prey oscillations without extinction (edge of collapse) |
| `lv_match` | parameter sets where the ABM behaves like a fitted LV model (`--arg model=chain`) |
| `custom` | any expression over metrics, e.g. `--arg "expr=mean_predators/cells*100 - 5*cv_prey"` |
| `qd` | MAP-Elites: a map of distinct behaviours (prey CV × period), best genome per cell |

    .venv/bin/python -m lotka evolve --list                       # objectives, genes, metrics
    .venv/bin/python -m lotka evolve --objective coexistence --pop 48 --gens 30
    .venv/bin/python -m lotka evolve --objective qd --gens 40 --out runs/evolve/qd
    .venv/bin/python -m lotka evolve --resume runs/evolve/qd --gens 60          # continue
    .venv/bin/python -m lotka evolve --objective msy --genes predator_repro_prob,predator_starve_time \
        --fix prey_repro_prob=0.1                                  # evolve a subset only

Defaults: 100×100 grid, 1,500 ticks, 3 seeds, population 48, all CPU cores. Outputs in the run
directory: `generations.csv`, `evals.csv` (every evaluation — a dataset), `best.json`,
`checkpoint.json` (resume), `evolution.png`, and `archive.json` for `qd`.
Finished runs appear in the dashboard's Preset list as "Evolved: …", and can be re-run with
`lotka batch --params-file runs/evolve/<run>/best.json` or `lotka compare --params-file ...`.

## Rules (one tick = one period)

1. **Vegetation regrowth** – an eaten fertile cell regrows after `veg_regrow_time` (R) ticks.
2. **Movement** – each animal moves to a free neighbouring cell (8-neighbourhood).
   With probability `*_directed_prob` the move is "smart": prey pick the cell
   that maximises `vegetation nearby − prey_fear × predators nearby`; predators
   pick the cell with most prey within `predator_sense_radius`. Otherwise random.
3. **Predation** – any prey within `predator_kill_radius` of a predator dies
   (each predator kills at most `predator_max_kills` per tick — satiation).
4. **Grazing** – each prey eats up to `prey_food_need` (K) vegetation cells within
   `prey_eat_radius` (own cell first). It is *fed* only if it ate K.
5. **Hunger** – a prey that is underfed for `prey_starve_time` (S) consecutive
   ticks dies. A predator that goes `predator_starve_time` (N) ticks without a kill dies.
   Optional max lifespans (`*_max_age`, 0 = off).
6. **Reproduction** – a fed prey reproduces with `prey_repro_prob`
   (or any prey, if `prey_repro_requires_fed` is off); a predator that killed this
   tick reproduces with `predator_repro_prob`. Offspring go into a free neighbouring cell.

Conflicts (two agents wanting the same cell / the same prey) are resolved in
random order; a run is fully reproducible from its `seed`.

## Theory: what to expect

The classic (non-spatial) Lotka–Volterra equations

    dPrey/dt = a·Prey − b·Prey·Pred
    dPred/dt = c·b·Prey·Pred − d·Pred

produce endless neutral cycles: prey boom → predators boom (with a lag of
about a quarter period) → prey crash → predators starve → repeat. The phase
portrait (prey vs predators) is a closed loop rotating counter-clockwise.

The parameters here map onto those terms:

| ODE term                     | in this model |
|------------------------------|---------------|
| a  prey growth               | `prey_repro_prob`, limited by vegetation (K, R) → logistic-like carrying capacity |
| b  predation rate            | `predator_kill_radius`, `predator_sense_radius`, `predator_directed_prob`, prey fleeing |
| c  conversion efficiency     | `predator_repro_prob`, `predator_max_kills` |
| d  predator death rate       | ~1 / `predator_starve_time`, `predator_max_age` |

Behaviours the spatial ABM adds, worth exploring:

* **Carrying capacity / bottom-up control** – prey are capped by vegetation
  (fertile area / (K · R) roughly). Fewer, slower-growing plants damp the system.
* **Damped vs sustained oscillations** – short N or strong satiation damps cycles
  toward a stable point; long N lets predators overshoot, giving large cycles
  (try N = 30 vs the default 12).
* **Extinction** – if predators can't find prey fast enough (N very small) they die out
  and prey go to carrying capacity; if predators are too efficient they eat all prey and
  then starve (*overexploitation*).
* **Paradox of enrichment** – more food for prey (bigger fertile area, faster regrowth,
  smaller K) can *destabilise* the system: bigger prey peaks → bigger predator
  booms → deeper crashes → extinction risk.
* **Spatial refuges & waves** – local extinctions are recolonised from neighbours;
  travelling waves and patchiness let coexistence persist where the ODE would crash.
  Larger grids are generally more stable than small ones.
* **Prey behaviour** – fleeing (`prey_fear`, `prey_directed_prob`) lowers effective
  predation and stabilises; random prey are easier to catch.

## Batch experiments

    .venv/bin/python -m lotka.batch --ticks 3000 --seeds 0,1,2 \
        --set width=150 --set height=150 \
        --sweep predator_starve_time=6,12,24 --sweep prey_repro_prob=0.04:0.16:4 \
        --workers 7 --out runs/starve_vs_repro

* `--set P=V` fixes a parameter, `--sweep P=a,b,c` or `P=lo:hi:n` sweeps it (Cartesian product × seeds).
* `--stop-on all|any|never` – end a run early when all / any species are extinct (default `all`).
* Output: `summary.csv` (one row per run) and `series/run_XXXX.csv` (per-tick metrics).

Summary columns: `outcome` (coexist / predators_extinct / prey_extinct / all_extinct),
extinction ticks, final/max counts, and over the second half of the run (after the
transient): mean, std, coefficient of variation (`cv_*` — oscillation strength),
dominant `period_*` (autocorrelation) and `predator_lag` (ticks predator peaks trail prey).

Per-tick metrics: `prey, predators, vegetation, prey_births, predator_births,
prey_eaten, prey_starved, predator_starved, prey_old_age, predator_old_age, prey_fed_frac`.

## Layout

    lotka/__main__.py      `python -m lotka serve|batch|compare|evolve`
    lotka/sim.py           model (Params dataclass + vectorised NumPy Simulation)
    lotka/presets.py       named parameter templates
    lotka/lv.py            Lotka-Volterra models, rate fitting, equilibria, mean-field map
    lotka/compare.py       ABM vs LV report (matplotlib)
    lotka/evolve.py        GA / MAP-Elites optimiser with pluggable objectives
    docs/lotka_volterra.md theory, ABM↔LV mapping, findings
    lotka/server.py        stdlib HTTP server + JSON API for the dashboard
    lotka/static/index.html dashboard (canvas grid, time series, phase portrait, events)
    lotka/batch.py         headless sweeps, parallel, CSV output
    tests/                 rule-level, LV-fitting and evolution tests
