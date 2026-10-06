# Roadmap

Fill in the status and dates in the team meeting.

| # | Phase | Status | Notes |
|---|---|---|---|
| 1 | Split research areas | | comms (Leo), autonomy hierarchy (Maria), multi-agent autonomy (Yuki) |
| 2 | Literature review | | see [literature.md](literature.md) |
| 3 | Scoping (scenario choice) | | 2D post-disaster search, see [scenarios.md](scenarios.md) |
| 4 | Functional requirements | | |
| 5 | Technical choices | | Python + numpy, Laplacian consensus ([ADR 0001](decisions/0001-weighted-delayed-consensus.md)) |
| 6 | System architecture | | see [architecture.md](architecture.md) |
| 7 | Demonstrator | in progress | step 1 (static scalar consensus) done |
| 8 | Scenario testing | | comms loss, drone failure, GPS loss, ... |
| 9 | Final evaluation | | |

## Demonstrator build-up (phase 7)

1. [x] Static graph, scalar consensus with delay, information weights, normalization ([details](consensus_step1.md))
2. [ ] Moving drones, switching topology (rebuild the graph each step, track mu_2(t))
3. [ ] Vector states x_i in R^m, then belief maps b(c)
4. [ ] Temporal weighting of old information, per-link delays, packet loss
5. [ ] World + sensor model, Bayesian belief update
6. [ ] Operator objective -> priors; local motion policy
7. [ ] Stress-test configs (failures, blackouts)
