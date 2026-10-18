# Lotka–Volterra theory and how it compares with the ABM

## 1. The equations

Let x = prey density and y = predator density (here: fraction of grid cells), t in ticks.

Classic Lotka–Volterra (Lotka 1925, Volterra 1926):

    dx/dt = a·x − b·x·y          prey grow exponentially, eaten at rate b per encounter
    dy/dt = c·b·x·y − g·y        predators convert eaten prey with efficiency c, die at rate g

* Equilibrium: x* = g / (c·b), y* = a / b. Prey density at equilibrium is set by the *predator*
  parameters, predator density by the *prey* parameters.
* Jacobian at (x*, y*) has eigenvalues ±i·√(a·g): a neutral **center**. Every initial condition
  lies on its own closed orbit (the quantity V = c·b·x − g·ln x + b·y − a·ln y is conserved).
* Period of small oscillations: T = 2π / √(a·g). Predator peaks lag prey peaks by ~T/4.
* Structurally unstable: any small change to the equations (self-limitation, satiation,
  noise, delays) turns the center into a stable spiral or a limit cycle.

Logistic prey (prey self-limitation, carrying capacity K):

    dx/dt = r·x·(1 − x/K) − b·x·y

The center becomes a **stable focus**: oscillations damp to the equilibrium. If x* = g/(c·b) ≥ K,
predators can't persist.

Rosenzweig–MacArthur (Holling type II functional response, h = handling/satiation time):

    dx/dt = r·x·(1 − x/K) − a·x·y/(1 + a·h·x)
    dy/dt = c·a·x·y/(1 + a·h·x) − g·y

x* = (g/c) / (a·(1 − h·g/c)). The prey nullcline is a hump peaking at x = (K − 1/(a·h))/2.
If x* lies to the right of the hump the equilibrium is stable; to the left it is unstable and a
**limit cycle** appears (Hopf bifurcation). Raising K moves the hump right → destabilises:
the **paradox of enrichment** (Rosenzweig 1971).

Vegetation–prey–predator chain (the ABM has three levels):

    dv/dt = ρ·(V − v) − e·v·x
    dx/dt = ε·e·v·x − μ·x − b·x·y
    dy/dt = c·b·x·y − g·y

## 2. Mapping ABM rules to LV terms

| LV term | ABM rule | Rule-derived (well-mixed) value |
|---|---|---|
| b (or a) predation rate | prey within `predator_kill_radius` die | P(kill) = 1 − (1 − x)^m, m = 8 neighbours → b ≈ 8 at low x |
| h satiation | `predator_max_kills` = 1 | kills per predator ≤ 1 per tick → type-II-like cap |
| c conversion | `predator_repro_prob` after a kill | c = `predator_repro_prob` |
| g predator death | die after N ticks unfed | stationary hazard f(1−f)^N / (1−(1−f)^N), → 1/N when food is rare |
| r, K prey growth | reproduce when fed, starve after S unfed ticks, food = vegetation | K ≈ set by fertile area, R and K-need |
| ρ regrowth | eaten cell regrows after exactly R ticks | a *delay*, not an exponential rate |

`lotka/lv.py` offers two ways to get the parameters:

1. **Fitted from events**: every tick the ABM records `prey_eaten`, `predator_births`,
   `predator_starved`, prey births/starvations and vegetation eaten/regrown. Each LV term is a
   regression of the matching event count on the state at the start of the tick
   (e.g. `prey_eaten ≈ b·x·y`). The model is then integrated from the ABM's initial densities.
2. **Mean-field of the rules** (`meanfield`): a discrete-time map that applies the exact ABM rules
   to a well-mixed population (binomial neighbourhoods, hunger-class structure, delayed regrowth),
   with no fitting at all.

## 3. What the comparison shows (200×200, seed 42, 3000 ticks)

Reproduce with `python -m lotka.compare --preset lv_cycles` and `--preset stable_coexistence`.

| | ABM | classic | logistic | RM | chain | mean-field |
|---|---|---|---|---|---|---|
| **lv_cycles** mean prey / pred | 975 / 756 | 1004 / 612 | 979 / 781 | 918 / 751 | 979 / 747 | 2476 / 194 |
| CV prey | 0.24 | 0.84 | ~0 | ~0 | ~0 | 0.96 |
| period | 138 | 268 | — | 253 (damped) | — | 398 |
| **stable** mean prey | 2135 | 2126 | 2126 | 2123 | 2126 | 2444 |

Findings:

* **Mean levels match very well** for all fitted models (within ~5%). The equilibrium
  x* = g/(c·b) is right: prey abundance is controlled by predator parameters, as LV predicts.
* **The fitted deterministic models damp, the ABM keeps cycling.** Logistic, RM and chain fits
  are stable foci, but the ABM sustains oscillations (CV 24%). These are *noise-driven
  quasi-cycles*: demographic noise keeps re-exciting the damped oscillation (McKane & Newman 2005).
  Classic LV oscillates, but only because it is neutral; its amplitude is too large.
* **The ABM cycles about twice as fast as the ODEs predict** (138 vs 250–270 ticks). The ABM has
  fixed delays (die after exactly N ticks, regrow after exactly R) and local, fast feedbacks
  that the ODEs smear into exponential rates.
* **Space stabilises.** The well-mixed mean-field of the same rules produces violent
  boom-bust cycles (prey up to 5k, predators near extinction). The fitted predation rate in the
  ABM (b ≈ 2.4) is ~3× below the well-mixed value (≈ 8): predators deplete prey locally and
  prey clusters escape. This is the classic Huffaker (1958) result that spatial structure
  lets predator-prey systems persist.
* **Prey growth is poorly described by logistic x(1 − x/K)** (rate-fit R² < 0.25): prey growth
  depends on the vegetation level, which lags. The chain model, which includes vegetation,
  fits the prey processes far better (R² 0.6–0.93 for the cycling case). Predator death stays
  poorly fitted by a constant g (R² ≈ 0.1): starvation after exactly N unfed ticks is a delayed,
  prey-dependent hazard, not a constant rate.
* Trajectory R² against the ABM is low or negative even when means and periods match: noisy
  cycles drift out of phase. Compare means, CV (amplitude), period and phase-portrait shape,
  not point-by-point trajectories.

## 4. Where to go next

* Add delays to the ODE (delay differential equations with lag N for predator death, R for
  regrowth): should recover the shorter period.
* Stochastic LV (chemical Langevin / Gillespie) to reproduce the quasi-cycle amplitude.
* Pair approximation (track neighbour correlations) to capture the spatial reduction of b.
* Use the GA (next step) to search for parameter regions where the ABM matches each LV regime.

References: Lotka (1925) *Elements of Physical Biology*; Volterra (1926) *Nature* 118;
Rosenzweig & MacArthur (1963) *Am. Nat.* 97; Rosenzweig (1971) *Science* 171;
Huffaker (1958) *Hilgardia* 27; McKane & Newman (2005) *PRL* 94; Wilensky (1997) NetLogo Wolf Sheep Predation.
