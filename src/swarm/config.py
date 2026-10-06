"""Scenario parameters. One dataclass, filled from a YAML file in configs/."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    scenario: str = "consensus_static"

    # --- Swarm / geometry ---
    n_drones: int = 8              # N
    area: float = 100.0            # drones placed uniformly in [0, area] x [0, area]  [m]
    r_c: float = 40.0              # comms radius: link iff ||p_i - p_j|| <= r_c        [m]
    sigma: float = 40.0            # edge weight a_ij = exp(-d_ij^2 / sigma^2)           [m]
    seed: int = 1                  # fixed seed -> reproducible positions and x(0)
    max_redraws: int = 10_000      # give up if no connected layout is found

    # --- Information weights gamma_i ---
    # gamma_i = gamma_default for everyone, except drones listed in gamma_overrides, e.g. {0: 2.0}.
    # A larger gamma_i means drone i trusts its own value more and moves less.
    gamma_default: float = 1.0
    gamma_overrides: dict[int, float] = field(default_factory=dict)

    # --- Initial values ---
    x0_low: float = 0.0            # x_i(0) ~ Uniform[x0_low, x0_high]
    x0_high: float = 10.0

    # --- Integration ---
    t_final: float = 60.0          # simulated time [s]
    dt: float = 0.01               # forward-Euler step [s]
    tau: float = 0.0               # comms delay [s] for the main run
    tol_rel: float = 0.01          # "converged" when |x_i - alpha| < tol_rel * |alpha| for all i

    # --- Delay sweep: extra runs at these multiples of tau_max = pi / (2 mu_n) ---
    delay_sweep: list[float] = field(default_factory=lambda: [0.0, 0.5, 1.2])

    # --- Output ---
    out_dir: str = "outputs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load a config file. Keys not listed above are an error, so typos don't pass silently."""
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
        if "gamma_overrides" in data:
            data["gamma_overrides"] = {int(k): float(v) for k, v in data["gamma_overrides"].items()}
        return cls(**data)
