"""Step 1: delayed, weighted, normalized scalar consensus on a static 8-drone graph.

Run from the repository root:
    python -m swarm_sim.main                 # defaults (gamma_i = 1 for all)
    python -m swarm_sim.main --heavy 0 3     # drones 0 and 3 get gamma = 2.0
    python -m swarm_sim.main --tau 0.3 --no-show
"""
import argparse

import matplotlib
import numpy as np

from swarm_sim.config import Config
from swarm_sim.consensus import (disagreement, measured_rate, simulate, simulate_vectorized,
                                 time_to_converge)
from swarm_sim.graph import build_graph, eigen_summary


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--r_c", type=float, default=Config.r_c)
    p.add_argument("--tau", type=float, default=Config.tau, help="delay for the main run [s]")
    p.add_argument("--dt", type=float, default=Config.dt)
    p.add_argument("--t_final", type=float, default=Config.t_final)
    p.add_argument("--heavy", type=int, nargs="*", default=[], help="drone ids with gamma = --heavy_gamma")
    p.add_argument("--heavy_gamma", type=float, default=2.0)
    p.add_argument("--no-show", action="store_true", help="save figures without opening windows")
    a = p.parse_args()
    return Config(seed=a.seed, r_c=a.r_c, tau=a.tau, dt=a.dt, t_final=a.t_final,
                  gamma_overrides={i: a.heavy_gamma for i in a.heavy}, show=not a.no_show)


def main(cfg: Config):
    if not cfg.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from swarm_sim import plots

    rng = np.random.default_rng(cfg.seed)
    g = build_graph(cfg, rng)
    x0 = rng.uniform(cfg.x0_low, cfg.x0_high, cfg.n_drones)
    ev = eigen_summary(g)
    mu_2, mu_n, tau_max = ev["mu_2"], ev["mu_n"], ev["tau_max"]

    # --- Theory ------------------------------------------------------------------
    # alpha = sum_i w_i x_i(0) / sum_i w_i,  w_i = gamma_i |N_i|   (w^T Lhat = 1^T L = 0 -> w^T x conserved)
    w = g.w
    alpha = float(w @ x0 / w.sum())
    plain_avg = float(x0.mean())

    # --- Euler step-size sanity -------------------------------------------------
    # For tau = 0 the Euler map is x <- (I - dt Lhat) x; it is stable iff |1 - dt mu_k| < 1, i.e. dt < 2 / mu_n.
    # We want dt much smaller than that so Euler error, not just stability, is negligible.
    dt_euler_max = 2.0 / mu_n
    if cfg.dt > 0.1 * dt_euler_max:
        print(f"WARNING: dt = {cfg.dt} is not << 2/mu_n = {dt_euler_max:.3f}; Euler may distort results.")

    # --- Main run ----------------------------------------------------------------
    t, X = simulate(g, x0, cfg.tau, cfg.dt, cfg.t_final)
    e = disagreement(X, alpha)
    t_conv = time_to_converge(t, X, alpha, cfg.tol_rel)
    rate = measured_rate(t, e)

    # Cross-checks: vectorized form must equal the per-drone loop; dt/10 must give the same trajectory.
    _, X_vec = simulate_vectorized(g, x0, cfg.tau, cfg.dt, cfg.t_final)
    _, X_fine = simulate_vectorized(g, x0, cfg.tau, cfg.dt / 10, cfg.t_final)
    loop_vs_vec = np.nanmax(np.abs(X - X_vec))
    dt_effect = np.nanmax(np.abs(X - X_fine[::10]))

    # --- Delay sweep ---------------------------------------------------------------
    taus = [("τ = 0", 0.0), ("τ = 0.5 τ_max", 0.5 * tau_max), ("τ = 1.2 τ_max", 1.2 * tau_max)]
    runs, sweep = [], []
    for label, tau in taus:
        tau_eff = round(tau / cfg.dt) * cfg.dt          # delay actually simulated (whole steps)
        ts, Xs = simulate(g, x0, tau, cfg.dt, cfg.t_final)
        es = disagreement(Xs, alpha)
        runs.append((f"{label} ({tau_eff:.3f} s)", ts, es))
        finite = np.isfinite(es)
        sweep.append((label, tau_eff, es[finite][-1], ts[finite][-1],
                      bool(finite.all() and es[-1] < es[0])))

    # Bound sharpness: just below / just above tau_max, at dt and dt/10 (rules out an Euler artefact).
    # Long horizon because modes near the bound are barely damped / barely growing.
    sharp = []
    for f in (0.95, 1.05):
        for h in (cfg.dt, cfg.dt / 10):
            ts, Xs = simulate_vectorized(g, x0, f * tau_max, h, 200.0)
            es = disagreement(Xs, alpha)
            sharp.append((f, h, es[-1] / es[0]))

    # With delay, the slow mode y' = -mu_2 y(t - tau) has (for mu_2 tau < 1/e) a real root s = -r with
    # r = mu_2 exp(r tau) > mu_2, so a moderate delay can make the slow mode decay FASTER. Fixed-point solve:
    tau_half = round(0.5 * tau_max / cfg.dt) * cfg.dt
    r_half = mu_2
    for _ in range(200):
        r_half = mu_2 * np.exp(r_half * tau_half)
    rate_half = measured_rate(runs[1][1], runs[1][2])

    # --- Print summary table -------------------------------------------------------
    line = "-" * 62
    print(line)
    print(f"Step 1 consensus | N={cfg.n_drones}, r_c={cfg.r_c} m, sigma={cfg.sigma} m, seed={cfg.seed}")
    print(f"dt={cfg.dt} s, T={cfg.t_final} s, tau={cfg.tau} s")
    print(line)
    print(f"{'drone':>5} {'|N_i|':>6} {'gamma_i':>8} {'w_i':>6} {'x_i(0)':>8} {'x_i(T)':>10} {'x_i(T)-alpha':>13}")
    for i in range(cfg.n_drones):
        print(f"{i:>5} {int(g.n_nbrs[i]):>6} {g.gamma[i]:>8.2f} {w[i]:>6.1f} {x0[i]:>8.3f} "
              f"{X[-1, i]:>10.5f} {X[-1, i] - alpha:>13.2e}")
    print(line)
    print(f"predicted consensus alpha (weighted) : {alpha:.5f}")
    print(f"plain average of x(0)                : {plain_avg:.5f}")
    print(f"simulated mean of x(T)               : {X[-1].mean():.5f}")
    print(f"conservation check |w.x(T)/sum w - alpha| : {abs(w @ X[-1] / w.sum() - alpha):.2e}")
    print(line)
    print("eigenvalues of Lhat:", np.array2string(ev["mu"], precision=4))
    print(f"mu_2 = {mu_2:.4f}   mu_n = {mu_n:.4f}   lambda_2(L) = {ev['lambda_2']:.4f}")
    print(f"delay bound tau_max = pi/(2 mu_n) = {tau_max:.4f} s")
    if cfg.tau >= tau_max:
        print(f"WARNING: tau = {cfg.tau} >= tau_max -> consensus is NOT expected (oscillation/divergence)")
    print(line)
    print(f"time to |x_i - alpha| < {cfg.tol_rel:.0%}|alpha| for all i : "
          + (f"{t_conv:.2f} s" if t_conv is not None else "not reached"))
    if cfg.tau == 0:
        print(f"measured decay rate {rate:.4f} 1/s vs mu_2 {mu_2:.4f} (ratio {rate / mu_2:.4f})")
    else:
        print(f"measured decay rate {rate:.4f} 1/s (with delay it differs from mu_2 = {mu_2:.4f})")
    print(f"per-drone loop vs vectorized max diff : {loop_vs_vec:.2e}")
    print(f"dt vs dt/10 max trajectory diff       : {dt_effect:.2e}   (Euler limit 2/mu_n = {dt_euler_max:.2f} s)")
    print(line)
    print(f"{'delay run':<16} {'tau [s]':>8} {'final ||x-alpha||':>18} {'at t [s]':>9}  converges?")
    for label, tau_eff, e_end, t_end, ok in sweep:
        label = label.replace("τ", "tau")
        print(f"{label:<16} {tau_eff:>8.3f} {e_end:>18.2e} {t_end:>9.2f}  {'yes' if ok else 'NO'}")
    print(f"tau = 0.5 tau_max slow-mode rate: predicted {r_half:.4f}, measured {rate_half:.4f} 1/s")
    print("bound sharpness, ||e(200 s)|| / ||e(0)||:")
    for f, h, ratio in sharp:
        print(f"   tau = {f:.2f} tau_max, dt = {h:<6g}: {ratio:.2e}  ({'decays' if ratio < 1 else 'GROWS'})")
    print(line)

    # --- Figures -------------------------------------------------------------------
    plots.plot_graph(g, x0, cfg)
    plots.plot_states(t, X, alpha, cfg, cfg.tau)
    plots.plot_disagreement(t, e, mu_2, rate, cfg)
    plots.plot_delay_comparison(runs, tau_max, cfg)
    print(f"figures saved to {cfg.out_dir}/")
    if cfg.show:
        plt.show()


if __name__ == "__main__":
    main(parse_args())
