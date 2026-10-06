import numpy as np
import pytest

from swarm import consensus, sim
from swarm.config import Config


def test_message_passing_matches_vectorized(setup):
    g, gamma, x0 = setup
    L_hat = consensus.normalized_laplacian(g.A, gamma)
    for tau in (0.0, 0.5):
        _, X = sim.run_consensus(g, gamma, x0, tau, 0.01, 10.0)
        _, Xv = sim.run_consensus_vectorized(L_hat, x0, tau, 0.01, 10.0)
        assert np.max(np.abs(X - Xv)) < 1e-12


@pytest.mark.parametrize("weights", [{}, {0: 2.0, 3: 2.0}])
def test_converges_to_weighted_alpha_at_rate_mu2(setup, weights):
    g, gamma, x0 = setup
    gamma = gamma.copy()
    for i, gi in weights.items():
        gamma[i] = gi
    A = g.A
    alpha = consensus.consensus_value(x0, A, gamma)
    mu_2 = consensus.spectrum(A, gamma)["mu_2"]
    L_hat = consensus.normalized_laplacian(A, gamma)
    t, X = sim.run_consensus_vectorized(L_hat, x0, 0.0, 0.01, 80.0)
    assert np.allclose(X[-1], alpha, atol=1e-5)
    assert sim.measured_rate(t, sim.disagreement(X, alpha)) == pytest.approx(mu_2, rel=0.02)


def test_alpha_conserved_with_delay(setup):
    g, gamma, x0 = setup
    A = g.A
    w = consensus.info_weights(A, gamma)
    L_hat = consensus.normalized_laplacian(A, gamma)
    _, X = sim.run_consensus_vectorized(L_hat, x0, 0.8, 0.01, 30.0)
    assert np.allclose(X @ w / w.sum(), consensus.consensus_value(x0, A, gamma), atol=1e-10)


@pytest.mark.parametrize("dt", [0.01, 0.001])
def test_delay_bound_is_sharp(setup, dt):
    """Below pi/(2 mu_n) the disagreement decays, above it grows, at both step sizes (not Euler)."""
    g, gamma, x0 = setup
    A = g.A
    L_hat = consensus.normalized_laplacian(A, gamma)
    alpha = consensus.consensus_value(x0, A, gamma)
    tau_max = consensus.spectrum(A, gamma)["tau_max"]
    ratios = []
    for f in (0.95, 1.05):
        _, X = sim.run_consensus_vectorized(L_hat, x0, f * tau_max, dt, 200.0)
        e = sim.disagreement(X, alpha)
        ratios.append(e[-1] / e[0])
    assert ratios[0] < 1 < ratios[1]


def test_small_delay_speeds_up_slow_mode(setup):
    g, gamma, x0 = setup
    A = g.A
    spec = consensus.spectrum(A, gamma)
    tau = round(0.5 * spec["tau_max"] / 0.01) * 0.01
    alpha = consensus.consensus_value(x0, A, gamma)
    L_hat = consensus.normalized_laplacian(A, gamma)
    t, X = sim.run_consensus_vectorized(L_hat, x0, tau, 0.01, 60.0)
    r = sim.measured_rate(t, sim.disagreement(X, alpha))
    assert r == pytest.approx(consensus.delayed_mode_rate(spec["mu_2"], tau), rel=0.02)
    assert r > spec["mu_2"]


def test_scenario_runs_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("scenario: consensus_static\nt_final: 40.0\ndelay_sweep: [0.0, 1.2]\n"
                 "gamma_overrides: {0: 2.0}\n")
    res = sim.run_consensus_static(Config.from_yaml(p))
    assert res.gamma[0] == 2.0
    assert [r.converges for r in res.sweep] == [True, False]


def test_unknown_config_key_is_an_error(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("r_cc: 40\n")
    with pytest.raises(ValueError, match="r_cc"):
        Config.from_yaml(p)
