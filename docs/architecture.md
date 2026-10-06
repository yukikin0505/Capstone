# Architecture

The simulation is built in layers. Information flows downward from operator intent to agent
decisions, and **sideways only through the comms graph**.

```
operator.py   (planned)  objective  ->  initial priors / task weights
      |
agent.py      (planned)  one drone: belief, decision, motion.  Sees: own sensor + its inbox
      |                                         ^
comms.py      who can talk to whom, delivery  --+   (range, weights, delay; later dropouts, LOS)
      |
consensus.py  pure math: local update rule, Laplacians, spectra, bounds
      |
world.py      (planned)  ground truth, target, sensor model
sim.py        loop, events, metrics, scenario runners
viz.py        figures (optional)
```

## Module responsibilities

| module | status | owns | must not |
|---|---|---|---|
| `config.py` | done | the `Config` dataclass and YAML loading | contain logic |
| `comms.py` | done | `CommsGraph`, unit-disk links, `a_ij`, `deliver()` | know about beliefs or tasks |
| `consensus.py` | done | `consensus_step()` and analysis (L, Lhat, spectrum, alpha, tau_max) | hold state or draw random numbers |
| `sim.py` | done (step 1) | the time loop, delay buffer, metrics, `SCENARIOS` registry | let an agent read another agent's state |
| `viz.py` | done (step 1) | plots | be imported by core modules |
| `operator.py` | planned | turning an objective into priors | |
| `world.py` | planned | ground truth and noisy observations | |
| `agent.py` | planned | per-drone belief update and policy | |

## Data flow in one step (step 1)

1. Every drone broadcasts its current value. The broadcast history is the delay buffer.
2. `comms.deliver(g, i, sent)` gives drone i the list `[(a_ij, x_j(t - tau)) for j in N_i]`.
3. `consensus.consensus_step()` updates x_i from that inbox and the drone's own memory.

## Adding a scenario

1. Write `run_<name>(cfg) -> Result` in `sim.py` (or a new module) and register it in `SCENARIOS`.
2. Add any new parameters to `Config` with defaults, then add `configs/<name>.yaml`.
3. Add tests, and an ADR if the design choice is significant.
