import numpy as np
import pytest

from swarm.comms import random_connected_graph
from swarm.config import Config


@pytest.fixture
def cfg() -> Config:
    return Config(t_final=40.0, delay_sweep=[])


@pytest.fixture
def setup(cfg):
    """Default 8-drone connected graph, gamma = 1, x(0) uniform in [0, 10]."""
    rng = np.random.default_rng(cfg.seed)
    g = random_connected_graph(cfg.n_drones, cfg.area, cfg.r_c, cfg.sigma, rng, cfg.max_redraws)
    x0 = rng.uniform(cfg.x0_low, cfg.x0_high, cfg.n_drones)
    return g, np.ones(cfg.n_drones), x0
