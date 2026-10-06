# CLAUDE.md

Context for Claude Code (and any other AI assistant) working in this repo.

## What this project is

UC Berkeley MEng capstone (Project 212, sponsor: U.S. Army Research Laboratory, Sep 2026 to May 2027): a **minimal simulation proof of concept** for objective-driven shared autonomy in UAS swarms. An operator states an objective; agents coordinate in a fully decentralized way to achieve it. Emphasis is on **architecture**, not on optimal algorithms or real hardware.

Advisors: Mark Mueller, Christian Brommer. Team: Yuki Kin (multi-agent autonomy / consensus), Maria Vlahos (autonomy hierarchy: goal -> tasks -> success metrics), Leonard "Leo" Wang (communications).

## Advisor guidance (keep this in mind when proposing changes)

- Start from a concrete scenario (current one: 2D search for a missing person after a disaster).
- Stay at the right level of abstraction: not low-level control (no MPC, no vehicle dynamics), not high-level LLM agents chatting with each other.
- Prefer no "judge" or leader nodes. Prefer probabilistic, gradient-based or learning-based coordination.
- Aim for fully decentralized operation with information propagating through the comms graph. Agents only observe the world partially.

## Core principles

- No leader drone, central node, judging node or ground station in the decision loop.
- Every drone runs the same algorithm and uses only its neighbours N_i, never the whole network.
- Decisions propagate hop by hop. Continuous-time model, simulated with a consistent discretization.

## Code conventions

- Python >= 3.10, dependencies kept minimal: `numpy`, `pyyaml`, optional `matplotlib`.
- Package lives in `src/swarm/`. Each module owns one layer:
  - `config.py`: scenario parameters (one dataclass), loaded from YAML
  - `comms.py`: who can talk to whom at each step, and message delivery
  - `consensus.py`: pure math functions, no side effects
  - `sim.py`: loop, events, metrics, scenario runners (`SCENARIOS` registry)
  - `viz.py`: figures, only module that imports matplotlib
  - planned: `operator.py` (objectives to initial priors), `world.py` (ground truth, sensor model), `agent.py` (one agent's belief and decisions, using only local info plus messages)
- An agent must never read another agent's state directly. Everything goes through the comms graph (`comms.deliver()`).
- All randomness goes through a seeded `numpy.random.Generator` passed in from the sim. Runs must be reproducible from the config seed.
- Scenarios are YAML in `configs/`. New behavior should be switchable from config, not hard-coded. Unknown config keys are an error.
- Type hints and short docstrings on public functions. Comment code with the equation each block implements; keep it readable for teammates.

## Consensus model (step 1, implemented; reference: Olfati-Saber, Fax, Murray, Proc. IEEE 2007)

- Drones = nodes, comms links = edges. Unit-disk graph: link iff ||p_i - p_j|| <= r_c (later: plus line-of-sight, buildings).
- Adjacency A = [a_ij], degree D = diag(sum_j a_ij), Laplacian L = D - A. Undirected for now (a_ij = a_ji), but each link is stored as two directed edges so asymmetric links can be added later.
- Edge weights: a_ij = exp(-d_ij^2 / sigma^2) inside r_c, 0 outside.
- Law, with information weight gamma_i, normalization by neighbour count and comms delay tau:
    gamma_i * dx_i/dt = (1/|N_i|) * sum_{j in N_i} a_ij * ( x_j(t - tau) - x_i(t - tau) )
  Matrix form: dx/dt = -Lhat x(t - tau), Lhat = Gamma^{-1} DeltaN^{-1} L, DeltaN = diag(|N_i|).
- Known results, all checked in `tests/test_sim.py`:
  * Undirected + connected => consensus on alpha = sum_i w_i x_i(0) / sum_i w_i, w_i = gamma_i * |N_i|.
  * Rate set by mu_2 of Lhat. Lhat is similar to symmetric W^{-1/2} L W^{-1/2}, W = Gamma*DeltaN, so its eigenvalues are real and >= 0.
  * Delay stability: consensus iff tau < pi / (2 * mu_n). A moderate delay (mu_2 tau < 1/e) speeds up the slow mode.
  * Euler: dt must be << 2 / mu_n.
- Isolated drones (|N_i| = 0) hold their value.
- Next stages: vector states x_i in R^m, switching/dynamic topology, sensor inputs that change x over time, temporal weighting of old info (x e^(-age/T)), link dropouts, node capacity limits, minimum separation d_min, 2D search-and-rescue with belief maps b(c) = P(person in cell).

## Commands

```bash
pip install -e ".[dev,viz]"
pytest                                   # run tests
ruff check src tests                     # lint
python -m swarm --config configs/consensus_static.yaml --plot
# planned: configs/search_rescue_2d.yaml, configs/stress_failures.yaml
```

## When making changes

- Add or update a test in `tests/` for any new behavior.
- Significant design choices go in `docs/decisions/` as a short ADR (copy `0000-template.md`).
- Keep `README.md` and `docs/architecture.md` in sync with the code layout.
- Generated output goes to `outputs/` (git-ignored). Figures meant for docs go in `docs/figures/`.
