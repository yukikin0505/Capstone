import pytest

from swarm import sim
from swarm.config import Config


def test_plots_save(tmp_path):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from swarm import viz

    res = sim.run_consensus_static(Config(t_final=5.0, out_dir=str(tmp_path)))
    viz.plot_all(res, show=False)
    assert len(list(tmp_path.glob("*.png"))) == 4
