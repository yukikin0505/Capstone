"""Consensus math: pure functions, no side effects, no simulation state.

Continuous-time law for drone i (see docs/consensus_step1.md):
    gamma_i * dx_i/dt = (1/|N_i|) * sum_{j in N_i} a_ij * ( x_j(t - tau) - x_i(t - tau) )
Matrix form:  dx/dt = -Lhat x(t - tau),  Lhat = Gamma^-1 DeltaN^-1 L,  L = D - A.
"""

from __future__ import annotations

import numpy as np


def consensus_step(
    x_i: float, x_i_delayed: float, inbox: list[tuple[float, float]], gamma_i: float, dt: float
) -> float:
    """One forward-Euler step for a single drone, using only local information.

    x_i          its current value
    x_i_delayed  its own value tau ago (kept in its own memory, so delays line up with the inbox)
    inbox        [(a_ij, x_j(t - tau)), ...] received from its neighbours
    Implements x_i <- x_i + dt / gamma_i * (1/|N_i|) * sum_j a_ij * ( x_j(t - tau) - x_i(t - tau) ).
    """
    if not inbox:              # isolated drone: nothing to agree with -> hold its value
        return x_i
    u_i = sum(a_ij * (x_j - x_i_delayed) for a_ij, x_j in inbox) / len(inbox)
    return x_i + dt * u_i / gamma_i


def laplacian(A: np.ndarray) -> np.ndarray:
    """L = D - A, D = diag(sum_j a_ij)."""
    return np.diag(A.sum(axis=1)) - A


def info_weights(A: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """w_i = gamma_i * |N_i|: diagonal of W = Gamma * DeltaN and left null vector of Lhat."""
    return gamma * (A > 0).sum(axis=1)


def normalized_laplacian(A: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """Lhat = Gamma^-1 DeltaN^-1 L = W^-1 L  (row i of L divided by gamma_i * |N_i|)."""
    return laplacian(A) / info_weights(A, gamma)[:, None]


def lhat_step(x: np.ndarray, x_delayed: np.ndarray, L_hat: np.ndarray, dt: float) -> np.ndarray:
    """Vectorized Euler step x <- x - dt * Lhat x(t - tau). Row i of Lhat is non-zero only at i and
    j in N_i, so this is consensus_step() for every drone at once. Used for analysis and tests."""
    return x - dt * L_hat @ x_delayed


def consensus_value(x0: np.ndarray, A: np.ndarray, gamma: np.ndarray) -> float:
    """alpha = sum_i w_i x_i(0) / sum_i w_i.

    Conserved because w^T Lhat = 1^T L = 0, also with delay."""
    w = info_weights(A, gamma)
    return float(w @ x0 / w.sum())


def spectrum(A: np.ndarray, gamma: np.ndarray) -> dict[str, float | np.ndarray]:
    """Eigenvalues of Lhat and plain L, and the delay bound.

    Lhat = W^-1 L is not symmetric, but it is similar to S = W^-1/2 L W^-1/2
    (Lhat = W^-1/2 S W^1/2), so its eigenvalues are real and >= 0. eigvalsh(S) is the
    numerically robust way to get them.
    """
    L = laplacian(A)
    w_isqrt = 1.0 / np.sqrt(info_weights(A, gamma))
    S = w_isqrt[:, None] * L * w_isqrt[None, :]
    mu = np.sort(np.linalg.eigvalsh(S))
    lam = np.sort(np.linalg.eigvalsh(L))
    return {
        "mu": mu,                            # eigenvalues of Lhat, ascending, mu_1 = 0
        "mu_2": float(mu[1]),                # slowest non-zero mode -> convergence rate
        "mu_n": float(mu[-1]),               # fastest mode -> delay bound
        "lambda_2": float(lam[1]),           # Fiedler value of plain L
        "tau_max": float(np.pi / (2.0 * mu[-1])),   # consensus iff tau < pi / (2 mu_n)
    }


def delayed_mode_rate(mu: float, tau: float, iters: int = 200) -> float:
    """Decay rate r of the scalar mode y' = -mu y(t - tau) when it has a real root s = -r.

    r solves r = mu exp(r tau), which has a solution for mu tau < 1/e. Then r > mu, so a small
    delay speeds this mode up."""
    if mu * tau >= 1.0 / np.e:
        raise ValueError("no real root: mu * tau >= 1/e (the slow mode oscillates)")
    r = mu
    for _ in range(iters):                   # fixed-point iteration, converges for mu tau < 1/e
        r = mu * np.exp(r * tau)
    return float(r)
