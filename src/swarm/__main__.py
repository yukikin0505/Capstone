"""CLI entry point.

    python -m swarm --config configs/consensus_static.yaml          # print the summary table
    python -m swarm --config configs/consensus_static.yaml --plot   # also save + show figures
"""

from __future__ import annotations

import argparse

import numpy as np

from swarm import sim
from swarm.config import Config


def print_consensus_report(res: sim.ConsensusResult) -> None:
    cfg, g, spec = res.cfg, res.graph, res.spec
    w = res.gamma * g.n_nbrs
    t_conv = sim.time_to_converge(res.t, res.X, res.alpha, cfg.tol_rel)
    rate = sim.measured_rate(res.t, res.e)
    line = "-" * 66
    print(line)
    print(f"{cfg.scenario} | N={cfg.n_drones}, r_c={cfg.r_c} m, sigma={cfg.sigma} m, "
          f"seed={cfg.seed}")
    print(f"dt={cfg.dt} s, T={cfg.t_final} s, tau={cfg.tau} s")
    print(line)
    print(f"{'drone':>5} {'|N_i|':>6} {'gamma_i':>8} {'w_i':>6} {'x_i(0)':>8} {'x_i(T)':>10} "
          f"{'x_i(T)-alpha':>13}")
    for i in range(cfg.n_drones):
        print(f"{i:>5} {int(g.n_nbrs[i]):>6} {res.gamma[i]:>8.2f} {w[i]:>6.1f} {res.x0[i]:>8.3f} "
              f"{res.X[-1, i]:>10.5f} {res.X[-1, i] - res.alpha:>13.2e}")
    print(line)
    print(f"predicted consensus alpha (weighted)      : {res.alpha:.5f}")
    print(f"plain average of x(0)                     : {res.x0.mean():.5f}")
    drift = abs(w @ res.X[-1] / w.sum() - res.alpha)
    print(f"conservation check |w.x(T)/sum w - alpha| : {drift:.2e}")
    print(line)
    print("eigenvalues of Lhat:", np.array2string(spec["mu"], precision=4))
    print(f"mu_2 = {spec['mu_2']:.4f}   mu_n = {spec['mu_n']:.4f}   "
          f"lambda_2(L) = {spec['lambda_2']:.4f}")
    print(f"delay bound tau_max = pi/(2 mu_n) = {spec['tau_max']:.4f} s")
    if cfg.tau >= spec["tau_max"]:
        print(f"WARNING: tau = {cfg.tau} >= tau_max -> consensus is NOT expected")
    if cfg.dt > 0.1 * 2.0 / spec["mu_n"]:
        print(f"WARNING: dt = {cfg.dt} is not << 2/mu_n = {2 / spec['mu_n']:.3f}; "
              "Euler may distort results")
    print(line)
    print(f"time to |x_i - alpha| < {cfg.tol_rel:.0%}|alpha| for all i : "
          + (f"{t_conv:.2f} s" if t_conv is not None else "not reached"))
    print(f"measured decay rate {rate:.4f} 1/s vs mu_2 {spec['mu_2']:.4f}"
          + ("" if cfg.tau == 0 else "  (with delay these differ)"))
    print(line)
    print(f"{'delay run':<18} {'tau [s]':>8} {'final ||x-alpha||':>18}  converges?")
    for run in res.sweep:
        name = "tau = 0" if run.factor == 0 else f"tau = {run.factor:g} tau_max"
        e_end = run.e[np.isfinite(run.e)][-1]
        print(f"{name:<18} {run.tau:>8.3f} {e_end:>18.2e}  {'yes' if run.converges else 'NO'}")
    print(line)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="swarm", description="Decentralized UAS swarm simulation")
    p.add_argument(
        "--config", required=True, help="scenario YAML, e.g. configs/consensus_static.yaml"
    )
    p.add_argument("--plot", action="store_true", help="save figures to out_dir and show them")
    args = p.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    if cfg.scenario not in sim.SCENARIOS:
        p.error(f"unknown scenario '{cfg.scenario}'; available: {sorted(sim.SCENARIOS)}")
    res = sim.SCENARIOS[cfg.scenario](cfg)
    print_consensus_report(res)

    if args.plot:
        try:
            from swarm import viz
        except ImportError:
            p.error('plotting needs matplotlib: pip install -e ".[viz]"')
        viz.plot_all(res)
        print(f"figures saved to {cfg.out_dir}/")


if __name__ == "__main__":
    main()
