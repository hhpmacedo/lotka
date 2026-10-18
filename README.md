# Lotka — agent-based predator / prey simulation

A spatial, agent-based version of the Lotka–Volterra predator–prey system.
Every pixel of a toroidal grid has a **vegetation layer** (grass or regrowing
soil) and an **animal layer** (empty, prey, or predator).

## Quick start

    cd ~/projects/lotka
    python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # once
    .venv/bin/python -m lotka.server          # dashboard on http://127.0.0.1:8765
    .venv/bin/python -m pytest -q             # tests

Dashboard: Space = play/pause. Edit parameters on the left, then "Apply & reset".
"Export CSV" downloads the per-tick metrics of the current run.

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

    lotka/sim.py           model (Params dataclass + vectorised NumPy Simulation)
    lotka/server.py        stdlib HTTP server + JSON API for the dashboard
    lotka/static/index.html dashboard (canvas grid, time series, phase portrait, events)
    lotka/batch.py         headless sweeps, parallel, CSV output
    tests/test_sim.py      rule-level tests (starvation timing, K feeding, kills, regrowth, determinism)
