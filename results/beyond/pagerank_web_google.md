PageRank on web-Google (875,713 nodes, 5,105,039 edges), damping 0.85, linear-system form, float64, one CPU thread of scipy.
Sweeps and seconds to reach a relative L1 residual of 1e-6 and 1e-10. One sweep = every node updated once. `experiments/pagerank_sweeps.py`.


**node order as in the file**

| method | sweeps to 1e-6 | sweeps to 1e-10 | seconds to 1e-6 | seconds to 1e-10 | ms per sweep | sweeps vs Jacobi (1e-10) |
|---|---|---|---|---|---|---|
| Jacobi (power iteration) | 65 | 119 | 0.739 | 1.345 | 11.3 | +0% |
| Gauss-Seidel | 34 | 62 | 5.355 | 9.853 | 158.92 | -48% |
| blocks, 4 | 48 | 103 | 0.594 | 1.327 | 12.89 | -13% |
| blocks, 4, order reshuffled each sweep | 50 | 104 | 0.683 | 1.362 | 13.1 | -13% |
| blocks, 16 | 41 | 96 | 0.418 | 0.972 | 10.12 | -19% |
| blocks, 16, order reshuffled each sweep | 45 | 97 | 0.466 | 1.001 | 10.32 | -18% |
| blocks, 64 | 36 | 89 | 0.391 | 0.962 | 10.81 | -25% |
| blocks, 64, order reshuffled each sweep | 42 | 89 | 0.464 | 1.004 | 11.28 | -25% |
| blocks, 256 | 34 | 80 | 0.381 | 0.899 | 11.24 | -33% |
| blocks, 256, order reshuffled each sweep | 41 | 83 | 0.468 | 0.947 | 11.41 | -30% |
| blocks, 1024 | 34 | 71 | 0.462 | 0.979 | 13.79 | -40% |
| blocks, 1024, order reshuffled each sweep | 41 | 78 | 0.609 | 1.155 | 14.81 | -34% |
| blocks, 4096 | 34 | 62 | 0.785 | 1.432 | 23.09 | -48% |
| blocks, 4096, order reshuffled each sweep | 41 | 77 | 1.132 | 2.16 | 28.05 | -35% |
| relaxed Jacobi, w = 0.5 | 135 | 249 | 1.995 | 3.648 | 14.65 | +109% |
| relaxed Jacobi, w = 0.8 | 82 | 152 | 1.159 | 2.16 | 14.21 | +28% |
| relaxed Jacobi, w = 1.1 | not reached | not reached | not reached | not reached | 14.05 |  |
| relaxed Jacobi, w = 1.2 | not reached | not reached | not reached | not reached | 13.85 |  |
| relaxed Jacobi, w = cosine 1->0.1 over 50 sweeps | not reached | not reached | not reached | not reached | 13.9 |  |
| relaxed Jacobi, w = cosine 1->0.1 over 150 sweeps | 80 | not reached | 1.1 | not reached | 14.12 |  |

**nodes relabelled at random**

| method | sweeps to 1e-6 | sweeps to 1e-10 | seconds to 1e-6 | seconds to 1e-10 | ms per sweep | sweeps vs Jacobi (1e-10) |
|---|---|---|---|---|---|---|
| Jacobi (power iteration) | 65 | 119 | 0.772 | 1.414 | 11.89 | +0% |
| Gauss-Seidel | 34 | 62 | 5.372 | 9.775 | 157.66 | -48% |
| blocks, 4 | 46 | 100 | 0.568 | 1.186 | 11.86 | -16% |
| blocks, 16 | 38 | 90 | 0.475 | 1.049 | 11.66 | -24% |
| blocks, 64 | 35 | 83 | 0.368 | 0.904 | 10.89 | -30% |
| blocks, 256 | 34 | 76 | 0.37 | 0.857 | 11.28 | -36% |
| blocks, 1024 | 34 | 68 | 0.472 | 0.958 | 14.09 | -43% |
| blocks, 4096 | 34 | 62 | 0.845 | 1.536 | 24.78 | -48% |

**reverse Cuthill-McKee order (linked nodes placed close together)**

| method | sweeps to 1e-6 | sweeps to 1e-10 | seconds to 1e-6 | seconds to 1e-10 | ms per sweep | sweeps vs Jacobi (1e-10) |
|---|---|---|---|---|---|---|
| Jacobi (power iteration) | 65 | 119 | 0.673 | 1.216 | 10.22 | +0% |
| Gauss-Seidel | 34 | 62 | 3.659 | 6.826 | 110.1 | -48% |
| blocks, 4 | 60 | 114 | 0.522 | 1.03 | 9.03 | -4% |
| blocks, 16 | 57 | 111 | 0.604 | 1.093 | 9.85 | -7% |
| blocks, 64 | 56 | 111 | 0.488 | 0.988 | 8.9 | -7% |
| blocks, 256 | 56 | 110 | 0.517 | 1.042 | 9.47 | -8% |
| blocks, 1024 | 55 | 110 | 0.686 | 1.381 | 12.56 | -8% |
| blocks, 4096 | 54 | 109 | 1.174 | 2.378 | 21.81 | -8% |

## Reading

- Using fresh values before the sweep ends is the one ingredient that carries over, and it is the known Gauss-Seidel idea:
  it halves the sweeps (65 -> 34 to 1e-6, 119 -> 62 to 1e-10), in line with the 40-48 % reported by Del Corso, Gulli and Romani.
- Blocks get most of that at the cost of a plain sweep: 256 blocks reach 1e-6 in 34 sweeps and 0.38 s against 0.74 s for
  Jacobi (-48 %), and 1e-10 in 0.90 s against 1.35 s (-33 %). More blocks need fewer sweeps but pay Python overhead per block.
  scipy's sparse triangular solve makes true Gauss-Seidel 14 times slower per sweep here, which is an implementation cost.
- Reshuffling the block order every sweep is worse than a fixed order (41 against 34 sweeps to 1e-6). Relabelling the nodes at
  random changes nothing. Reverse Cuthill-McKee order removes most of the block gain (54-60 sweeps): it puts linked nodes in
  the same block, where they are updated together from old values.
- A fading running total has no job here: it is under-relaxation, and it is slower (w = 0.5 needs 2.1 times the sweeps).
  The e-means step curve (cosine 1 -> 0.1) stalls before 1e-10. A step above 1 diverges on this graph.
- No meaning for a deterministic fixed point: forgetting totals, a decaying step curve, a timed repair (nothing dies) and a
  final exact clean-up (there is no batch noise to remove). The only setting left for a formula is the number of blocks.
