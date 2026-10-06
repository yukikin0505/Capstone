"""Communication graph: positions, unit-disk adjacency, weights, Laplacians and their eigenvalues.

This module is the *analysis* side (it sees the whole network). The drones themselves never use
anything from here except their own row: their neighbour list N_i and weights a_ij.
"""
from dataclasses import dataclass

import numpy as np

from swarm_sim.config import Config


@dataclass
class Graph:
    pos: np.ndarray        # (N, 2) drone positions [m]
    A: np.ndarray          # (N, N) weighted adjacency a_ij
    D: np.ndarray          # degree matrix diag(sum_j a_ij)
    L: np.ndarray          # Laplacian L = D - A
    n_nbrs: np.ndarray     # |N_i| (neighbour COUNT, not weighted degree)
    gamma: np.ndarray      # information weights gamma_i
    L_hat: np.ndarray      # Lhat = Gamma^-1 DeltaN^-1 L
    edges: list            # directed edges (i, j, a_ij): each undirected link stored twice
    neighbours: list       # neighbours[i] = list of (j, a_ij) -> all drone i knows about the graph

    @property
    def w(self):
        """w_i = gamma_i * |N_i|: left null vector of Lhat, sets the consensus value."""
        return self.gamma * self.n_nbrs


def random_positions(cfg: Config, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0.0, cfg.area, size=(cfg.n_drones, 2))


def unit_disk_edges(pos: np.ndarray, r_c: float, sigma: float) -> list:
    """Directed edge list. Link iff d_ij <= r_c, weight a_ij = exp(-d_ij^2 / sigma^2).

    Each undirected link is stored as two directed edges (i->j and j->i) so that asymmetric links
    (e.g. different transmit power, line-of-sight blocked one way) can be added later.
    """
    n = len(pos)
    edges = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d_ij = np.linalg.norm(pos[i] - pos[j])
            if d_ij <= r_c:
                edges.append((i, j, float(np.exp(-d_ij**2 / sigma**2))))
    return edges


def adjacency_from_edges(n: int, edges: list) -> np.ndarray:
    A = np.zeros((n, n))
    for i, j, a in edges:
        A[i, j] = a
    return A


def is_connected(A: np.ndarray) -> bool:
    """Breadth-first search from drone 0 over links with a_ij > 0."""
    n = len(A)
    seen, frontier = {0}, [0]
    while frontier:
        i = frontier.pop()
        for j in np.nonzero(A[i])[0]:
            if j not in seen:
                seen.add(int(j))
                frontier.append(int(j))
    return len(seen) == n


def build_graph(cfg: Config, rng: np.random.Generator) -> Graph:
    """Draw positions until the unit-disk graph is connected, then build all matrices."""
    for _ in range(cfg.max_redraws):
        pos = random_positions(cfg, rng)
        edges = unit_disk_edges(pos, cfg.r_c, cfg.sigma)
        A = adjacency_from_edges(cfg.n_drones, edges)
        if is_connected(A):
            break
    else:
        raise RuntimeError("No connected layout found; increase r_c or max_redraws.")

    D = np.diag(A.sum(axis=1))                       # D = diag(sum_j a_ij)
    L = D - A                                        # L = D - A
    n_nbrs = (A > 0).sum(axis=1).astype(float)       # |N_i|
    gamma = np.full(cfg.n_drones, cfg.gamma_default)
    for i, g in cfg.gamma_overrides.items():
        gamma[i] = g
    # Lhat = Gamma^-1 DeltaN^-1 L   (row i of L divided by gamma_i * |N_i|)
    L_hat = L / (gamma * n_nbrs)[:, None]

    neighbours = [[(j, a) for (i2, j, a) in edges if i2 == i] for i in range(cfg.n_drones)]
    return Graph(pos, A, D, L, n_nbrs, gamma, L_hat, edges, neighbours)


def eigen_summary(g: Graph) -> dict:
    """Eigenvalues of Lhat and of plain L.

    Lhat = W^-1 L with W = Gamma * DeltaN is not symmetric, but it is similar to the symmetric
        S = W^-1/2 L W^-1/2       (Lhat = W^-1/2 S W^1/2),
    so its eigenvalues are real and >= 0. We use eigvalsh(S) for numerical robustness.
    """
    W = g.w
    W_isqrt = 1.0 / np.sqrt(W)
    S = W_isqrt[:, None] * g.L * W_isqrt[None, :]   # S = W^-1/2 L W^-1/2
    mu = np.sort(np.linalg.eigvalsh(S))
    lam = np.sort(np.linalg.eigvalsh(g.L))
    return {
        "mu": mu,              # eigenvalues of Lhat, ascending, mu_1 = 0
        "mu_2": mu[1],         # slowest non-zero mode -> convergence rate
        "mu_n": mu[-1],        # fastest mode -> delay bound
        "lambda_2": lam[1],    # Fiedler value of plain L (algebraic connectivity)
        "tau_max": np.pi / (2.0 * mu[-1]),   # consensus iff tau < pi / (2 mu_n)
    }
