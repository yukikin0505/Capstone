"""Run the step-1 consensus scenario from Python instead of the CLI and compare information weights.

    python examples/consensus_from_python.py
"""

from swarm import sim
from swarm.config import Config

for overrides in ({}, {0: 2.0, 3: 2.0}, {0: 5.0}):
    res = sim.run_consensus_static(Config(gamma_overrides=overrides, delay_sweep=[]))
    print(f"gamma overrides {overrides!s:<20} -> alpha = {res.alpha:.4f} "
          f"(plain average {res.x0.mean():.4f}), x(T) = {res.X[-1].mean():.4f}")
