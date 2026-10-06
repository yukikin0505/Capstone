"""Communication layer: who can talk to whom, and message delivery.

The comms graph is a unit-disk graph with distance-based link weights. It is the ONLY way
information moves between drones: `deliver()` hands each drone the delayed values of its
neighbours, and nothing else about the other drones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# A directed link i -> j with weight a_ij. Each undirected link is stored as two of these so that
# asymmetric links (blocked line of sight, different transmit power) can be added later.
Edge = tuple[int, int, float]


@dataclass
class CommsGraph:
    pos: np.ndarray            # (N, 2) drone positions [m]
    edges: list[Edge]          # directed edges (i, j, a_ij)
    neighbours: list[list[tuple[int, float]]]   # neighbours[i] = [(j, a_ij), ...] -> N_i

    @property
    def n(self) -> int:
        return len(self.pos)

    @property
    def A(self) -> np.ndarray:
        """Weighted adjacency A = [a_ij] (analysis only; drones never see the full matrix)."""
        A = np.zeros((self.n, self.n))
        for i, j, a in self.edges:
            A[i, j] = a
        return A

    @property
    def n_nbrs(self) -> np.ndarray:
        """|N_i|: neighbour COUNT, not weighted degree."""
        return np.array([len(nb) for nb in self.neighbours], dtype=float)


def unit_disk_edges(pos: np.ndarray, r_c: float, sigma: float) -> list[Edge]:
    """Link iff d_ij = ||p_i - p_j|| <= r_c, with weight a_ij = exp(-d_ij^2 / sigma^2)."""
    edges = []
    for i in range(len(pos)):
        for j in range(len(pos)):
            if i == j:
                continue
            d_ij = np.linalg.norm(pos[i] - pos[j])
            if d_ij <= r_c:
                edges.append((i, j, float(np.exp(-(d_ij**2) / sigma**2))))
    return edges


def make_graph(pos: np.ndarray, r_c: float, sigma: float) -> CommsGraph:
    edges = unit_disk_edges(pos, r_c, sigma)
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(len(pos))]
    for i, j, a in edges:
        neighbours[i].append((j, a))
    return CommsGraph(pos, edges, neighbours)


def is_connected(g: CommsGraph) -> bool:
    """Breadth-first search from drone 0 over the links."""
    seen, frontier = {0}, [0]
    while frontier:
        i = frontier.pop()
        for j, _ in g.neighbours[i]:
            if j not in seen:
                seen.add(j)
                frontier.append(j)
    return len(seen) == g.n


def random_connected_graph(
    n: int, area: float, r_c: float, sigma: float, rng: np.random.Generator, max_redraws: int
) -> CommsGraph:
    """Draw uniform positions in [0, area]^2 until the unit-disk graph is connected."""
    for _ in range(max_redraws):
        g = make_graph(rng.uniform(0.0, area, size=(n, 2)), r_c, sigma)
        if is_connected(g):
            return g
    raise RuntimeError("No connected layout found; increase r_c or max_redraws.")


def deliver(g: CommsGraph, i: int, broadcast: np.ndarray) -> list[tuple[float, float]]:
    """Inbox of drone i: [(a_ij, x_j), ...] for j in N_i, where `broadcast` holds the values the
    drones sent tau seconds ago. Drone i receives nothing from drones outside N_i."""
    return [(a_ij, float(broadcast[j])) for j, a_ij in g.neighbours[i]]
