# 0001: Weighted, normalized, delayed Laplacian consensus as the fusion primitive

- **Status:** accepted
- **Date:** 2026-10-06
- **Deciders:** Yuki Kin

## Context

Drones need to merge their estimates (eventually belief maps) without a leader or judging node,
using only their neighbours and with radio delay. The advisors asked for a fully decentralized method
based on probability or gradients, sitting between low-level control and LLM agents.

## Decision

Each drone runs

    gamma_i * dx_i/dt = (1/|N_i|) * sum_{j in N_i} a_ij * ( x_j(t - tau) - x_i(t - tau) )

on a unit-disk comms graph with a_ij = exp(-d_ij^2 / sigma^2). It is discretized with forward Euler
and a delay buffer. The information weight gamma_i lets a drone with better information move less.
The 1/|N_i| normalization keeps the update scale independent of how crowded a drone's
neighbourhood is.

## Alternatives considered

- **Plain Laplacian (gamma = 1, no normalization).** Simpler, but the step size has to shrink as
  the maximum degree grows, and there is no way to express confidence.
- **Metropolis-Hastings weights.** These give a doubly stochastic matrix and a plain average. They
  are a good discrete-time option for later belief fusion, but they need the neighbours' degrees.
- **Leader- or judge-based fusion.** This is ruled out by the decentralization requirement.

## Consequences

- The consensus value is the weighted average with w_i = gamma_i |N_i|, not the plain average.
  Weight comes from both confidence and connectivity.
- The stability limit is tau < pi / (2 mu_n) of Lhat, and convergence speed is set by mu_2. Both
  are checked in the tests.
- With switching topology |N_i| changes over time, so the weighted average is no longer conserved.
  This needs revisiting in step 2.
