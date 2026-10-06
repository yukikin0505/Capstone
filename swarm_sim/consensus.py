"""Delayed, weighted, normalized consensus on one scalar x per drone.

Continuous-time law for drone i (CLAUDE.md):
    gamma_i * dx_i/dt = (1/|N_i|) * sum_{j in N_i} a_ij * ( x_j(t - tau) - x_i(t - tau) )
Matrix form:  dx/dt = -Lhat x(t - tau),  Lhat = Gamma^-1 DeltaN^-1 L.

Discretization: forward Euler with step dt, delay rounded to d = round(tau / dt) steps:
    x_i[k+1] = x_i[k] + dt/(gamma_i |N_i|) * sum_j a_ij * ( x_j[k-d] - x_i[k-d] )
History is constant before t = 0:  x[k] = x[0] for k < 0  (i.e. t in [-tau, 0]).
"""
import numpy as np

from swarm_sim.graph import Graph


def drone_update(i: int, x_i_now: float, x_delayed: np.ndarray, g: Graph, dt: float) -> float:
    """One Euler step for drone i, using ONLY local information:
    its own current value, its own delayed value, and its neighbours' delayed values
    (what it received over the radio tau seconds ago), plus its own gamma_i and |N_i|.
    """
    nbrs = g.neighbours[i]                       # N_i as list of (j, a_ij)
    if not nbrs:                                 # isolated drone: nothing to agree with -> hold
        return x_i_now
    # (1/|N_i|) * sum_j a_ij * ( x_j(t - tau) - x_i(t - tau) )
    u_i = sum(a_ij * (x_delayed[j] - x_delayed[i]) for j, a_ij in nbrs) / len(nbrs)
    # gamma_i * dx_i/dt = u_i   ->   x_i <- x_i + dt * u_i / gamma_i
    return x_i_now + dt * u_i / g.gamma[i]


def simulate(g: Graph, x0: np.ndarray, tau: float, dt: float, t_final: float):
    """Integrate all drones in parallel (synchronous update). Returns (t, X) with X[k] = x(t_k).

    The state history X doubles as the delay buffer: the 'radio message' drone i sees from j at
    step k is X[k - d, j].
    """
    n = len(x0)
    d = int(round(tau / dt))                     # delay in steps
    n_steps = int(round(t_final / dt))
    t = np.arange(n_steps + 1) * dt
    X = np.empty((n_steps + 1, n))
    X[0] = x0
    for k in range(n_steps):
        x_delayed = X[max(k - d, 0)]             # x(t - tau); constant history x(0) for t < tau
        X[k + 1] = [drone_update(i, X[k, i], x_delayed, g, dt) for i in range(n)]
        if not np.all(np.isfinite(X[k + 1])) or np.abs(X[k + 1]).max() > 1e12:
            X[k + 1:] = np.nan                   # diverged (tau above the bound); stop early
            break
    return t, X


def simulate_vectorized(g: Graph, x0: np.ndarray, tau: float, dt: float, t_final: float):
    """Same as simulate(), written as x[k+1] = x[k] - dt * Lhat x[k-d].
    Row i of Lhat only has non-zeros at i and j in N_i, so this is exactly the per-drone loop
    stacked into one matrix product. Used only to cross-check simulate()."""
    d = int(round(tau / dt))
    n_steps = int(round(t_final / dt))
    t = np.arange(n_steps + 1) * dt
    X = np.empty((n_steps + 1, len(x0)))
    X[0] = x0
    for k in range(n_steps):
        X[k + 1] = X[k] - dt * g.L_hat @ X[max(k - d, 0)]
    return t, X


def disagreement(X: np.ndarray, alpha: float) -> np.ndarray:
    """||x(t) - alpha * 1||_2 at every time step."""
    return np.linalg.norm(X - alpha, axis=1)


def time_to_converge(t: np.ndarray, X: np.ndarray, alpha: float, tol_rel: float):
    """First time after which |x_i - alpha| < tol_rel * |alpha| for ALL i and stays there."""
    inside = np.all(np.abs(X - alpha) < tol_rel * abs(alpha), axis=1)
    if not inside[-1]:
        return None
    outside = np.nonzero(~inside)[0]
    return t[0] if len(outside) == 0 else t[outside[-1] + 1]


def measured_rate(t: np.ndarray, e: np.ndarray, lo: float = 1e-8, hi: float = 1e-2) -> float:
    """Fit log e(t) = c - r t on the stretch where e(0)*lo < e < e(0)*hi (past the transient,
    above round-off). Returns the decay rate r, to compare against mu_2."""
    m = (e < e[0] * hi) & (e > e[0] * lo)
    if m.sum() < 10:
        return float("nan")
    slope, _ = np.polyfit(t[m], np.log(e[m]), 1)
    return -slope
