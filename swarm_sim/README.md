# Step 1: scalar consensus with delay, information weights and neighbour normalization

Eight drones sit at fixed random positions and agree on one number `x` (think "each drone's
estimate of something"). Nothing moves yet and there is no sensing or vehicle dynamics. The point
is to check that the consensus law from `CLAUDE.md` behaves as theory predicts before we build on it.

## How to run

```bash
pip install numpy matplotlib
python -m swarm_sim.main                     # from the repo root; prints table, saves + shows plots
python -m swarm_sim.main --no-show           # headless: only save PNGs to outputs/
python -m swarm_sim.main --heavy 0 3         # drones 0 and 3 get gamma = 2.0 (--heavy_gamma to change)
python -m swarm_sim.main --tau 0.5 --r_c 35 --seed 4
```

All parameters live in `config.py` (`Config` dataclass). CLI flags override the common ones.

| file | what it does |
|---|---|
| `config.py` | every parameter (N, area, r_c, sigma, gamma, dt, tau, T, seed, ...) |
| `graph.py` | positions (redrawn until connected), unit-disk links, weights, A, D, L, Lhat, eigenvalues |
| `consensus.py` | per-drone delayed Euler update, plus a vectorized copy used only as a cross-check |
| `plots.py` | the four figures |
| `main.py` | runs everything, prints the summary table, saves figures |

## Equations

**Graph.** Link iff `d_ij = ||p_i - p_j|| <= r_c`. Weight `a_ij = exp(-d_ij^2 / sigma^2)` inside r_c,
otherwise 0, so closer drones count more. Links are stored as two directed edges (i->j, j->i), which
keeps the door open for asymmetric links later.

    A = [a_ij],   D = diag(sum_j a_ij),   L = D - A,   DeltaN = diag(|N_i|),   Gamma = diag(gamma_i)

**Consensus law (per drone i, only local information):**

    gamma_i * dx_i/dt = (1/|N_i|) * sum_{j in N_i} a_ij * ( x_j(t - tau) - x_i(t - tau) )

Matrix form: `dx/dt = -Lhat x(t - tau)`, with `Lhat = Gamma^-1 DeltaN^-1 L`.

**Discretization.** Forward Euler with step `dt`, delay rounded to `d = round(tau/dt)` steps,
constant history `x(t) = x(0)` for `t in [-tau, 0]`:

    x_i[k+1] = x_i[k] + dt / (gamma_i |N_i|) * sum_j a_ij * ( x_j[k-d] - x_i[k-d] )

`consensus.drone_update()` is that line for a single drone. It only reads drone i's own value and
the delayed values of its neighbours. `simulate_vectorized()` is the same thing written as
`x[k+1] = x[k] - dt * Lhat x[k-d]`, and `main.py` checks that the two agree to round-off.

**Theory checked numerically**

| prediction | formula |
|---|---|
| consensus value | `alpha = sum_i w_i x_i(0) / sum_i w_i`, `w_i = gamma_i |N_i|` (w is the left null vector of Lhat, so `w.x` is conserved even with delay) |
| eigenvalues | Lhat is similar to symmetric `S = W^-1/2 L W^-1/2`, `W = Gamma DeltaN`, so its eigenvalues `0 = mu_1 < mu_2 <= ... <= mu_n` are real |
| convergence rate | `||x(t) - alpha 1|| ~ exp(-mu_2 t)` for tau = 0 |
| delay bound | consensus iff `tau < tau_max = pi / (2 mu_n)` |
| Euler stability | for tau = 0, need `dt < 2 / mu_n`; we use dt far below that |

## Figures (`outputs/`)

1. **`1_graph.png`**: the comms graph. Edge width is proportional to `a_ij`, node colour is `x_i(0)`,
   each label gives the drone id, `|N_i|` and `gamma_i`, and the dashed circle is the r_c disk around drone 0.
2. **`2_states.png`**: every `x_i(t)`. They should all meet at the dashed line (predicted alpha),
   which is generally not the plain average.
3. **`3_disagreement.png`**: `||x(t) - alpha 1||` on a log axis. After a short transient the curve is a
   straight line whose slope matches the dashed `exp(-mu_2 t)` reference.
4. **`4_delay_comparison.png`**: disagreement for tau = 0, 0.5 tau_max and 1.2 tau_max. Below the bound
   it decays, above it it oscillates and grows.

## Notes on the results (default seed)

- A moderate delay can make convergence *faster*. Take the slow mode `y' = -mu_2 y(t - tau)`. When
  `mu_2 tau < 1/e` it has a real root `-r` with `r = mu_2 exp(r tau) > mu_2`. `main.py` prints the
  predicted and measured rates for the 0.5 tau_max run. The fast modes (`mu_n tau` near pi/2) are the
  ones that become unstable.
- The bound is sharp. `main.py` runs 0.95 and 1.05 tau_max for 200 s at both `dt` and `dt/10`:
  the first decays and the second grows at both step sizes, so the instability is not an Euler artefact.
- If a drone has no neighbours (`|N_i| = 0`) it holds its value. This cannot happen yet because
  the layout is redrawn until the graph is connected, but it will matter once the topology switches.
