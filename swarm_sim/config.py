"""All simulation parameters in one place (see CLAUDE.md: 'Parameters in one config object')."""
from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Swarm / geometry ---
    n_drones: int = 8              # N
    area: float = 100.0            # drones placed uniformly in [0, area] x [0, area]  [m]
    r_c: float = 40.0              # comms radius: link iff ||p_i - p_j|| <= r_c        [m]
    sigma: float = 40.0            # edge weight a_ij = exp(-d_ij^2 / sigma^2)           [m]
    seed: int = 1                  # fixed seed -> reproducible positions and x(0)
    max_redraws: int = 10_000      # give up if no connected layout is found

    # --- Information weights gamma_i ---
    # Default gamma_i = 1 for everyone. Give some drones more "inertia" (they trust their own
    # value more and move less) via e.g. gamma_overrides={0: 2.0, 3: 2.0}.
    gamma_default: float = 1.0
    gamma_overrides: dict = field(default_factory=dict)

    # --- Initial values ---
    x0_low: float = 0.0            # x_i(0) ~ Uniform[x0_low, x0_high]
    x0_high: float = 10.0

    # --- Integration ---
    t_final: float = 60.0          # simulated time [s]
    dt: float = 0.01               # forward-Euler step [s]
    tau: float = 0.0               # comms delay [s] for the main run
    tol_rel: float = 0.01          # "converged" when |x_i - alpha| < tol_rel * |alpha| for all i

    # --- Output ---
    out_dir: str = "outputs"
    show: bool = True              # plt.show() at the end (disable for headless runs)
