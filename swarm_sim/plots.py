"""Figures for step 1. Each function saves a PNG into cfg.out_dir and returns the figure."""
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from swarm_sim.graph import Graph

# Fixed categorical order (one colour per drone id, never cycled) and a one-hue sequential ramp.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "font.size": 10,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2,
})


def _save(fig, cfg, name):
    os.makedirs(cfg.out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(cfg.out_dir, name), bbox_inches="tight")
    return fig


def plot_graph(g: Graph, x0: np.ndarray, cfg, disk_drone: int = 0):
    """Plot 1: comms graph. Edge width ~ a_ij, node colour = x_i(0), dashed r_c disk around one drone."""
    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    a_max = g.A.max()
    for i, j, a in g.edges:
        if i < j:   # each undirected link is stored twice; draw once
            ax.plot(*g.pos[[i, j]].T, color=INK_2, lw=0.5 + 4.0 * a / a_max, alpha=0.55,
                    solid_capstyle="round", zorder=1)
    c = g.pos[disk_drone]
    ax.add_patch(plt.Circle(c, cfg.r_c, fill=False, ls="--", lw=1.2, color=INK_2, zorder=0))
    ax.annotate(f"r_c = {cfg.r_c:.0f} m", c + [0, cfg.r_c], ha="center", va="bottom",
                color=INK_2, fontsize=9)
    sc = ax.scatter(*g.pos.T, c=x0, cmap=SEQ_BLUE, vmin=cfg.x0_low, vmax=cfg.x0_high, s=360,
                    edgecolors="white", linewidths=2, zorder=2)
    # Info labels: start each one radially outside the swarm, then push overlapping labels apart
    # (simple repulsion in data units) and connect each to its drone with a thin leader line.
    centre = g.pos.mean(axis=0)
    u = g.pos - centre
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-9
    lab = g.pos + 16.0 * u
    box = np.array([22.0, 6.0])                       # approx. label width/height [m]
    for _ in range(300):
        for i in range(len(lab)):
            for j in range(len(lab)):
                gap = lab[i] - lab[j]
                if i != j and np.all(np.abs(gap) < box):
                    lab[i] += 0.5 * np.sign(gap + 1e-6) * (box - np.abs(gap)) * [0.2, 1.0]
            for p_node in g.pos:                       # keep labels off the node markers
                gap = lab[i] - p_node
                if np.linalg.norm(gap) < 9.0:
                    lab[i] += gap / (np.linalg.norm(gap) + 1e-9) * 0.5
    for i, p in enumerate(g.pos):
        ax.annotate(f"{i}", p, ha="center", va="center", fontsize=9, fontweight="bold",
                    color="white" if x0[i] > 0.5 * (cfg.x0_low + cfg.x0_high) else INK, zorder=3)
        ax.annotate(f"{i}: |N|={int(g.n_nbrs[i])}, γ={g.gamma[i]:g}", p, xytext=lab[i],
                    textcoords="data", ha="center", va="center", fontsize=8, color=INK_2, zorder=4,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=GRID, lw=0.6),
                    arrowprops=dict(arrowstyle="-", color=INK_2, lw=0.6, shrinkA=0, shrinkB=9))
    fig.colorbar(sc, ax=ax, shrink=0.8, label="initial value $x_i(0)$")
    lo = min(-5.0, *(c - cfg.r_c - 5), *(lab.min(axis=0) - 0.6 * box))   # keep r_c disk and labels in view
    hi = max(cfg.area + 5.0, *(c + cfg.r_c + 5), *(lab.max(axis=0) + 0.6 * box))
    ax.add_patch(plt.Rectangle((0, 0), cfg.area, cfg.area, fill=False, lw=0.8, ec=GRID, zorder=0))
    ax.set(xlim=(lo, hi), ylim=(lo, hi), aspect="equal",
           xlabel="x [m]", ylabel="y [m]",
           title=f"Comms graph (N={len(x0)}, edge width $\\propto a_{{ij}}$)")
    return _save(fig, cfg, "1_graph.png")


def plot_states(t, X, alpha, cfg, tau):
    """Plot 2: x_i(t) for every drone, dashed line at the predicted consensus value alpha."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i in range(X.shape[1]):
        ax.plot(t, X[:, i], color=SERIES[i % len(SERIES)], label=f"drone {i}")
    ax.axhline(alpha, color=INK, ls="--", lw=1.2)
    ax.annotate(f"predicted α = {alpha:.3f}", (t[-1], alpha), xytext=(0, 5),
                textcoords="offset points", ha="right", va="bottom", color=INK)
    ax.set(xlabel="time t [s]", ylabel="$x_i(t)$", title=f"Consensus on a scalar (τ = {tau:g} s)")
    ax.legend(ncol=4, fontsize=8, frameon=False, loc="upper right", bbox_to_anchor=(1, 0.93))
    return _save(fig, cfg, "2_states.png")


def plot_disagreement(t, e, mu_2, rate_meas, cfg):
    """Plot 3: ||x(t) - alpha 1|| on log scale, with a reference line of slope exp(-mu_2 t)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.semilogy(t, e, color=SERIES[0], label="simulated $\\|x(t) - \\alpha\\mathbf{1}\\|$")
    # Anchor the reference at mid-run so it overlays the asymptotic slope (not the transient).
    k = len(t) // 2
    ax.semilogy(t, e[k] * np.exp(-mu_2 * (t - t[k])), color=INK, ls="--", lw=1.2,
                label=f"reference $\\propto e^{{-\\mu_2 t}}$, μ₂ = {mu_2:.4f}")
    ax.set(xlabel="time t [s]", ylabel="disagreement",
           title=f"Disagreement decay (measured rate {rate_meas:.4f} 1/s)")
    ax.set_ylim(max(e[e > 0].min() * 0.3, 1e-16), e.max() * 3)
    ax.legend(frameon=False)
    return _save(fig, cfg, "3_disagreement.png")


def plot_delay_comparison(runs, tau_max, cfg):
    """Plot 4: disagreement for several delays. runs = [(label, t, e), ...]."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for idx, (label, t, e) in enumerate(runs):
        ax.semilogy(t, e, color=SERIES[idx], label=label)
        last = np.nonzero(np.isfinite(e))[0][-1]
        ax.annotate(label.split(" (")[0], (t[last], e[last]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set(xlabel="time t [s]", ylabel="$\\|x(t) - \\alpha\\mathbf{1}\\|$",
           title=f"Delay stability bound: τ_max = π / (2 μ_n) = {tau_max:.3f} s")
    ax.legend(frameon=False, loc="lower left")
    return _save(fig, cfg, "4_delay_comparison.png")
