# Capstone 212: Self-Coordinating Swarm (UC Berkeley ME, sponsor: U.S. Army Research Laboratory)

## Project
Proof-of-concept simulation of objective-driven shared autonomy for UAS swarms: a single operator gives mission
objectives (not per-drone commands) and the swarm decides task allocation, coordination and adaptation itself.
The focus is validating the architecture, not optimizing algorithms or flying real hardware. Sep 2026 to May 2027.
Advisors: Mark Mueller, Christian Brommer. Team: Yuki Kin (multi-agent autonomy / consensus model),
Maria Vlahos (autonomy hierarchy: goal -> tasks -> success metrics), Leonard "Leo" Wang (communications).
Advisor guidance: start from one scenario; not too low level (MPC) and not too high level (LLMs chatting);
prefer fully decentralized decisions with probability/gradients over "judging" nodes.

## Core principles
- Fully decentralized: no leader drone, central node, judging node or ground station in the decision loop.
- Every drone runs the same algorithm and uses only its neighbours N_i, never the whole network.
- Decisions propagate hop by hop. Continuous-time model, simulated with a consistent discretization.

## Model (reference: Olfati-Saber, Fax, Murray, "Consensus and Cooperation in Networked Multi-Agent Systems", Proc. IEEE 2007)
- Drones = nodes, comms links = edges. Unit-disk graph: link iff ||p_i - p_j|| <= r_c (later: plus line-of-sight, buildings).
- Adjacency A = [a_ij], degree D = diag(sum_j a_ij), Laplacian L = D - A. Start undirected (a_ij = a_ji),
  but store each link as two directed edges so asymmetric links can be added later.
- Edge weights: farther drones count less, e.g. a_ij = exp(-d_ij^2 / sigma^2) inside r_c, 0 outside.
- Core law, with information weight gamma_i, normalization by neighbour count and comms delay tau:
    gamma_i * dx_i/dt = (1/|N_i|) * sum_{j in N_i} a_ij * ( x_j(t - tau) - x_i(t - tau) )
  Matrix form: dx/dt = -Lhat x(t - tau), Lhat = Gamma^{-1} DeltaN^{-1} L, DeltaN = diag(|N_i|).
- Known results to check numerically:
  * Undirected + connected => consensus on alpha = sum_i w_i x_i(0) / sum_i w_i with w_i = gamma_i * |N_i|
    (gamma_i = 1 and no normalization -> plain average).
  * Rate set by mu_2, the second-smallest eigenvalue of Lhat (Fiedler value for plain L).
    Lhat is similar to the symmetric W^{-1/2} L W^{-1/2}, W = Gamma*DeltaN, so its eigenvalues are real and >= 0.
  * Delay stability: consensus iff tau < pi / (2 * mu_n), mu_n = largest eigenvalue of Lhat.
  * Disagreement decays roughly like exp(-mu_2 t) for tau = 0.
- Later stages, not yet: vector states x_i in R^m, switching/dynamic topology, sensor inputs that change x over time,
  vehicle dynamics (x' = Ax + Bu, y = C1 x, z_ij = C2(x_i - x_j)), temporal weighting of old info (x e^(-age/T)),
  Kalman filtering of sensor data, minimum separation d_min, node capacity limits,
  2D search-and-rescue scenario with belief maps b(c) = P(person in cell).

## Coding conventions
- Python 3, numpy + matplotlib (no heavy frameworks). Fixed random seeds. Parameters in one config object.
- Comment code with the equation each block implements. Keep it simple and readable for teammates.
