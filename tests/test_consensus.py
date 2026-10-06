import numpy as np
import pytest

from swarm import consensus


def test_isolated_drone_holds_value():
    assert consensus.consensus_step(3.0, 2.0, [], gamma_i=1.0, dt=0.1) == 3.0


def test_consensus_step_formula():
    # x_i + dt/gamma_i * (1/|N_i|) * sum a_ij (x_j - x_i_delayed)
    x = consensus.consensus_step(1.0, 1.0, [(0.5, 3.0), (1.0, 0.0)], gamma_i=2.0, dt=0.1)
    assert x == pytest.approx(1.0 + 0.1 / 2.0 * (0.5 * 2.0 + 1.0 * -1.0) / 2)


def test_spectrum_real_nonnegative(setup):
    g, gamma, _ = setup
    gamma = gamma.copy()
    gamma[[0, 3]] = 2.0
    A = g.A
    spec = consensus.spectrum(A, gamma)
    assert spec["mu"][0] == pytest.approx(0.0, abs=1e-12)
    assert spec["mu_2"] > 0                                       # connected
    eig = np.linalg.eigvals(consensus.normalized_laplacian(A, gamma))
    assert np.allclose(np.sort(eig.real), spec["mu"]) and np.allclose(eig.imag, 0)


def test_consensus_value_reduces_to_average():
    # complete graph with equal weights and gamma = 1 -> every w_i equal -> plain average
    A = np.ones((4, 4)) - np.eye(4)
    x0 = np.array([1.0, 2.0, 3.0, 10.0])
    assert consensus.consensus_value(x0, A, np.ones(4)) == pytest.approx(x0.mean())


def test_delayed_mode_rate():
    r = consensus.delayed_mode_rate(0.3, 0.6)
    assert r == pytest.approx(0.3 * np.exp(r * 0.6)) and r > 0.3
    with pytest.raises(ValueError):
        consensus.delayed_mode_rate(1.0, 1.0)
