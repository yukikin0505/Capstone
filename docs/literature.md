# Literature

Each entry: full reference, then a 2-3 line takeaway for this project.

## Consensus and networked multi-agent systems

- **R. Olfati-Saber, J. A. Fax, R. M. Murray, "Consensus and Cooperation in Networked Multi-Agent
  Systems," *Proceedings of the IEEE*, 95(1), 2007.**
  This is the basis of our consensus layer: the Laplacian protocol dx/dt = -Lx, consensus on the
  (weighted) average for connected undirected graphs, convergence rate set by lambda_2, and the
  delay bound tau < pi / (2 lambda_n). Step 1 reproduces these results numerically.
