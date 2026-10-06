"""Simulation loop, metrics and scenario runners.

Discretization of the consensus law: forward Euler with step dt, delay rounded to d = round(tau/dt)
steps, constant history x(t) = x(0) for t in [-tau, 0]:
    x_i[k+1] = x_i[k] + dt / (gamma_i |N_i|) * sum_j a_ij * ( x_j[k-d] - x_i[k-d] )
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from swarm import comms, consensus
from swarm.config import Config


# --------------------------------------------------------------------------- simulation loops
def run_consensus(
    g: comms.CommsGraph, gamma: np.ndarray, x0: np.ndarray, tau: float, dt: float, t_final: float
) -> tuple[np.ndarray, np.ndarray]:
    """Message-passing simulation. Returns (t, X) with X[k] = x(t_k).

    Every step, each drone broadcasts its value; comms.deliver() hands drone i the values its
    neighbours broadcast d steps ago. The drone then updates from its inbox and its own memory only.
    Stops early (rest filled with NaN) if the run diverges, e.g. for tau above the delay bound.
    """
    d = int(round(tau / dt))
    n_steps = int(round(t_final / dt))
    t = np.arange(n_steps + 1) * dt
    X = np.empty((n_steps + 1, g.n))         # broadcast history = delay buffer
    X[0] = x0
    for k in range(n_steps):
        sent = X[max(k - d, 0)]              # what was on the air tau ago
        for i in range(g.n):
            inbox = comms.deliver(g, i, sent)
            X[k + 1, i] = consensus.consensus_step(X[k, i], sent[i], inbox, gamma[i], dt)
        if not np.all(np.isfinite(X[k + 1])) or np.abs(X[k + 1]).max() > 1e12:
            X[k + 1 :] = np.nan
            break
    return t, X


def run_consensus_vectorized(
    L_hat: np.ndarray, x0: np.ndarray, tau: float, dt: float, t_final: float
) -> tuple[np.ndarray, np.ndarray]:
    """Same dynamics as run_consensus() via x[k+1] = x[k] - dt Lhat x[k-d]. Fast; analysis/tests."""
    d = int(round(tau / dt))
    n_steps = int(round(t_final / dt))
    t = np.arange(n_steps + 1) * dt
    X = np.empty((n_steps + 1, len(x0)))
    X[0] = x0
    for k in range(n_steps):
        X[k + 1] = consensus.lhat_step(X[k], X[max(k - d, 0)], L_hat, dt)
    return t, X


# --------------------------------------------------------------------------- metrics
def disagreement(X: np.ndarray, alpha: float) -> np.ndarray:
    """||x(t) - alpha * 1||_2 at every time step."""
    return np.linalg.norm(X - alpha, axis=1)


def time_to_converge(t: np.ndarray, X: np.ndarray, alpha: float, tol_rel: float) -> float | None:
    """First time after which |x_i - alpha| < tol_rel * |alpha| for ALL i and stays there."""
    inside = np.all(np.abs(X - alpha) < tol_rel * abs(alpha), axis=1)
    if not inside[-1]:
        return None
    outside = np.nonzero(~inside)[0]
    return float(t[0]) if len(outside) == 0 else float(t[outside[-1] + 1])


def measured_rate(t: np.ndarray, e: np.ndarray, lo: float = 1e-8, hi: float = 1e-2) -> float:
    """Fit log e(t) = c - r t where e(0)*lo < e < e(0)*hi (past the transient, above round-off)."""
    m = (e < e[0] * hi) & (e > e[0] * lo)
    if m.sum() < 10:
        return float("nan")
    slope, _ = np.polyfit(t[m], np.log(e[m]), 1)
    return float(-slope)


# --------------------------------------------------------------------------- scenario
@dataclass
class DelayRun:
    factor: float              # tau / tau_max requested
    tau: float                 # delay actually simulated (whole steps)
    t: np.ndarray
    e: np.ndarray              # disagreement

    @property
    def converges(self) -> bool:
        """True if log ||x - alpha 1|| trends down over the second half of the run (a straight-line
        fit, so oscillations above the delay bound don't fool a first-vs-last comparison)."""
        if not np.all(np.isfinite(self.e)):
            return False                     # diverged and was stopped
        h = len(self.t) // 2
        e = np.maximum(self.e[h:], 1e-300)   # guard log(0) once it has fully converged
        return bool(np.polyfit(self.t[h:], np.log(e), 1)[0] < 0)


@dataclass
class ConsensusResult:
    cfg: Config
    graph: comms.CommsGraph
    gamma: np.ndarray
    x0: np.ndarray
    alpha: float
    spec: dict
    t: np.ndarray
    X: np.ndarray
    sweep: list[DelayRun] = field(default_factory=list)

    @property
    def e(self) -> np.ndarray:
        return disagreement(self.X, self.alpha)


def run_consensus_static(cfg: Config) -> ConsensusResult:
    """Step 1 scenario: static drones, one scalar, delay sweep around the stability bound."""
    rng = np.random.default_rng(cfg.seed)
    g = comms.random_connected_graph(
        cfg.n_drones, cfg.area, cfg.r_c, cfg.sigma, rng, cfg.max_redraws
    )
    x0 = rng.uniform(cfg.x0_low, cfg.x0_high, cfg.n_drones)
    gamma = np.full(cfg.n_drones, cfg.gamma_default)
    for i, gi in cfg.gamma_overrides.items():
        gamma[i] = gi

    A = g.A
    spec = consensus.spectrum(A, gamma)
    alpha = consensus.consensus_value(x0, A, gamma)
    t, X = run_consensus(g, gamma, x0, cfg.tau, cfg.dt, cfg.t_final)

    sweep = []
    for f in cfg.delay_sweep:
        tau = round(f * spec["tau_max"] / cfg.dt) * cfg.dt
        ts, Xs = run_consensus(g, gamma, x0, tau, cfg.dt, cfg.t_final)
        sweep.append(DelayRun(f, tau, ts, disagreement(Xs, alpha)))
    return ConsensusResult(cfg, g, gamma, x0, alpha, spec, t, X, sweep)


SCENARIOS = {"consensus_static": run_consensus_static}
