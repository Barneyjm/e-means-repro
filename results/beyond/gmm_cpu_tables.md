# OnlineGaussianMixture on CPU against scikit-learn's GaussianMixture

Produced by `experiments/bench_gmm_sklearn.py` (the `summary` mode prints the tables below from the `gmm_cpu_*.log` files here).

- Data: SIFT1M reduced to 64 whitened PCA dimensions (`run_gmm.sift64`), float32; held-out = the 100k SIFT learn vectors.
- Score: mean log-likelihood per held-out row, computed in float64 by one function for every model. Higher is better.
- Machine: Apple M2 Max, 12 cores, numpy with Accelerate, scikit-learn 1.9.1. Diagonal covariances everywhere.
- Timed runs went through a one-at-a-time CPU queue, but GPU jobs of other experiments (about one busy core each) ran
  alongside. Trust ratios, not absolute seconds.
- "Random start" = means on random rows, the estimator's default above 64 components. Up to 64 components the default is
  k-means++ seeds; the 100k, K = 64 block below uses the random start, and the "Starting means" table has the seeded numbers.
- scikit-learn rows at 1M are one seed. "Same random start" rows give scikit-learn the same kind of start (random rows,
  global variance, equal weights) with `tol=0`, so it runs exactly the passes shown.
- K = 1,024 has no scikit-learn row (100 passes would take about half an hour). For reference, full-pass EM on the GPU
  from a random start scored -78.270 / -78.175 / -78.130 after 25 / 50 / 100 passes on the same data and metric
  (`gmm_main_and_ablations.md`).

### 1M, K = 1024

| Method | Passes | Seeds | Time (s) | Held-out log-lik | vs scikit-learn defaults |
|---|---|---|---|---|---|
| `OnlineGaussianMixture`, random start | 1 | 3 | 1.52 ± 0.08 | -78.709 ± 0.018 | – |
| `OnlineGaussianMixture`, random start | 2 | 3 | 2.74 ± 0.09 | -78.278 ± 0.005 | – |
| `OnlineGaussianMixture`, random start | 3 | 3 | 4.11 ± 0.10 | -78.207 ± 0.007 | – |
| `OnlineGaussianMixture`, random start | 5 | 3 | 6.03 ± 0.12 | -78.172 ± 0.005 | – |
| `OnlineGaussianMixture`, random start | 10 | 3 | 10.64 ± 0.04 | -78.169 ± 0.006 | – |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 1 | 3 | 2.45 ± 0.28 | -78.861 ± 0.022 | – |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 2 | 3 | 3.73 ± 0.20 | -78.236 ± 0.007 | – |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 3 | 3 | 5.61 ± 0.25 | -78.179 ± 0.009 | – |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 5 | 3 | 7.60 ± 0.11 | -78.156 ± 0.006 | – |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 10 | 3 | 13.66 ± 0.13 | -78.165 ± 0.006 | – |

### 1M, K = 256

| Method | Passes | Seeds | Time (s) | Held-out log-lik | vs scikit-learn defaults |
|---|---|---|---|---|---|
| `GaussianMixture`, defaults (k-means start, to convergence) | 68 | 1 | 319.64 | -80.862 | +0.000 |
| `GaussianMixture`, same random start as online | 30 | 1 | 122.20 | -80.957 | -0.095 |
| `GaussianMixture`, same random start as online | 65 | 1 | 251.83 | -80.860 | +0.002 |
| `GaussianMixture`, same random start as online | 100 | 1 | 393.50 | -80.844 | +0.018 |
| `OnlineGaussianMixture`, random start | 1 | 3 | 0.61 ± 0.09 | -81.189 ± 0.008 | -0.328 |
| `OnlineGaussianMixture`, random start | 2 | 3 | 0.90 ± 0.10 | -80.921 ± 0.005 | -0.059 |
| `OnlineGaussianMixture`, random start | 3 | 3 | 1.31 ± 0.15 | -80.871 ± 0.008 | -0.009 |
| `OnlineGaussianMixture`, random start | 5 | 3 | 1.93 ± 0.13 | -80.891 ± 0.003 | -0.030 |
| `OnlineGaussianMixture`, random start | 10 | 3 | 3.81 ± 0.07 | -80.844 ± 0.003 | +0.018 |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 1 | 3 | 1.12 ± 0.22 | -81.086 ± 0.005 | -0.224 |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 2 | 3 | 1.48 ± 0.22 | -80.906 ± 0.003 | -0.045 |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 3 | 3 | 2.18 ± 0.27 | -80.871 ± 0.007 | -0.009 |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 5 | 3 | 2.95 ± 0.05 | -80.879 ± 0.008 | -0.017 |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 10 | 3 | 5.97 ± 0.25 | -80.837 ± 0.005 | +0.025 |

### 100k, K = 256

| Method | Passes | Seeds | Time (s) | Held-out log-lik | vs scikit-learn defaults |
|---|---|---|---|---|---|
| `GaussianMixture`, defaults (k-means start, to convergence) | 62 ± 8 | 3 | 26.30 ± 2.85 | -81.808 ± 0.014 | +0.000 |
| `GaussianMixture`, same random start as online | 25 | 3 | 8.83 ± 0.33 | -81.888 ± 0.020 | -0.080 |
| `GaussianMixture`, same random start as online | 100 | 3 | 35.27 ± 0.89 | -81.820 ± 0.031 | -0.012 |
| `OnlineGaussianMixture`, random start | 1 | 3 | 0.12 ± 0.00 | -82.640 ± 0.009 | -0.831 |
| `OnlineGaussianMixture`, random start | 2 | 3 | 0.15 ± 0.00 | -82.081 ± 0.013 | -0.272 |
| `OnlineGaussianMixture`, random start | 3 | 3 | 0.22 ± 0.00 | -81.988 ± 0.008 | -0.180 |
| `OnlineGaussianMixture`, random start | 5 | 3 | 0.25 ± 0.00 | -81.936 ± 0.016 | -0.128 |
| `OnlineGaussianMixture`, random start | 10 | 3 | 0.37 ± 0.00 | -81.864 ± 0.007 | -0.056 |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 1 | 3 | 0.30 ± 0.01 | -83.058 ± 0.119 | -1.250 |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 2 | 3 | 0.45 ± 0.01 | -82.097 ± 0.018 | -0.289 |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 3 | 3 | 0.67 ± 0.01 | -81.972 ± 0.024 | -0.164 |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 5 | 3 | 0.62 ± 0.00 | -81.912 ± 0.024 | -0.104 |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 10 | 3 | 0.74 ± 0.00 | -81.849 ± 0.025 | -0.041 |

### 100k, K = 64

| Method | Passes | Seeds | Time (s) | Held-out log-lik | vs scikit-learn defaults |
|---|---|---|---|---|---|
| `GaussianMixture`, defaults (k-means start, to convergence) | 55 ± 0 | 3 | 6.37 ± 0.11 | -84.259 ± 0.004 | +0.000 |
| `GaussianMixture`, same random start as online | 25 | 3 | 2.52 ± 0.11 | -84.292 ± 0.014 | -0.033 |
| `GaussianMixture`, same random start as online | 100 | 3 | 9.82 ± 0.44 | -84.238 ± 0.011 | +0.021 |
| `OnlineGaussianMixture`, random start | 1 | 3 | 0.12 ± 0.00 | -84.816 ± 0.056 | -0.557 |
| `OnlineGaussianMixture`, random start | 2 | 3 | 0.24 ± 0.04 | -84.336 ± 0.004 | -0.078 |
| `OnlineGaussianMixture`, random start | 3 | 3 | 0.20 ± 0.02 | -84.291 ± 0.012 | -0.033 |
| `OnlineGaussianMixture`, random start | 5 | 3 | 0.33 ± 0.04 | -84.282 ± 0.022 | -0.023 |
| `OnlineGaussianMixture`, random start | 10 | 3 | 0.40 ± 0.03 | -84.267 ± 0.021 | -0.008 |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 1 | 3 | 0.24 ± 0.04 | -84.706 ± 0.047 | -0.447 |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 2 | 3 | 0.45 ± 0.06 | -84.315 ± 0.010 | -0.056 |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 3 | 3 | 0.58 ± 0.08 | -84.304 ± 0.013 | -0.045 |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 5 | 3 | 0.91 ± 0.08 | -84.278 ± 0.030 | -0.019 |
| `OnlineGaussianMixture`, random start, batch / 4 and step / 4 | 10 | 3 | 1.16 ± 0.11 | -84.237 ± 0.016 | +0.022 |

Threads (1M rows, K = 256, 3 passes): 1 threads 4.83 s, 4 threads 1.67 s, all threads 1.25 s

### Textbook online EM (step t^-0.6, no revival), K = 256

| Rows | Passes | Method | Batch | Time (s) | Held-out log-lik | vs online recipe |
|---|---|---|---|---|---|---|
| 100k | 3 | `OnlineGaussianMixture` | 4096 | 0.23 ± 0.02 | -81.988 ± 0.008 | +0.000 |
| 100k | 3 | textbook, averaged over 2nd half | 4096 | 0.25 ± 0.04 | -82.705 ± 0.030 | -0.716 |
| 100k | 3 | textbook, averaged over 2nd half | 512 | 0.95 ± 0.06 | -85.321 ± 0.053 | -3.333 |
| 100k | 3 | textbook, averaged over 2nd half | 128 | 1.10 ± 0.02 | -88.250 ± 0.066 | -6.261 |
| 100k | 3 | textbook, averaged over 2nd half | 64 | 1.48 ± 0.04 | -89.248 ± 0.251 | -7.260 |
| 100k | 3 | textbook, last iterate | 4096 | 0.23 ± 0.00 | -82.671 ± 0.030 | -0.683 |
| 100k | 3 | textbook, last iterate | 512 | 0.90 ± 0.01 | -85.323 ± 0.047 | -3.335 |
| 100k | 3 | textbook, last iterate | 128 | 1.08 ± 0.05 | -88.252 ± 0.071 | -6.264 |
| 100k | 3 | textbook, last iterate | 64 | 1.47 ± 0.05 | -89.250 ± 0.254 | -7.262 |
| 100k | 10 | `OnlineGaussianMixture` | 16384 | 0.42 ± 0.02 | -81.864 ± 0.007 | +0.000 |
| 100k | 10 | textbook, averaged over 2nd half | 16384 | 0.38 ± 0.00 | -82.169 ± 0.006 | -0.304 |
| 100k | 10 | textbook, averaged over 2nd half | 2048 | 1.42 ± 0.16 | -83.123 ± 0.018 | -1.259 |
| 100k | 10 | textbook, averaged over 2nd half | 512 | 3.43 ± 0.53 | -85.271 ± 0.037 | -3.407 |
| 100k | 10 | textbook, averaged over 2nd half | 128 | 3.67 ± 0.25 | -88.247 ± 0.066 | -6.383 |
| 100k | 10 | textbook, last iterate | 16384 | 0.40 ± 0.00 | -82.128 ± 0.006 | -0.263 |
| 100k | 10 | textbook, last iterate | 2048 | 1.48 ± 0.09 | -83.113 ± 0.017 | -1.249 |
| 100k | 10 | textbook, last iterate | 512 | 3.19 ± 0.14 | -85.268 ± 0.038 | -3.404 |
| 100k | 10 | textbook, last iterate | 128 | 3.64 ± 0.16 | -88.251 ± 0.067 | -6.387 |
| 1M | 3 | `OnlineGaussianMixture` | 32768 | 1.42 ± 0.11 | -80.871 ± 0.008 | +0.000 |
| 1M | 3 | textbook, averaged over 2nd half | 32768 | 1.38 ± 0.02 | -81.188 ± 0.016 | -0.317 |
| 1M | 3 | textbook, averaged over 2nd half | 4096 | 3.30 ± 0.46 | -81.607 ± 0.051 | -0.736 |
| 1M | 3 | textbook, averaged over 2nd half | 1024 | 10.01 ± 0.71 | -83.340 ± 0.149 | -2.470 |
| 1M | 3 | textbook, averaged over 2nd half | 256 | 14.33 ± 2.39 | -86.304 ± 0.046 | -5.433 |
| 1M | 3 | textbook, last iterate | 32768 | 1.41 ± 0.11 | -81.141 ± 0.014 | -0.270 |
| 1M | 3 | textbook, last iterate | 4096 | 3.57 ± 0.24 | -81.605 ± 0.050 | -0.734 |
| 1M | 3 | textbook, last iterate | 1024 | 9.71 ± 0.60 | -83.342 ± 0.148 | -2.471 |
| 1M | 3 | textbook, last iterate | 256 | 15.04 ± 3.02 | -86.307 ± 0.047 | -5.436 |
| 1M | 10 | `OnlineGaussianMixture` | 65536 | 3.72 ± 0.09 | -80.844 ± 0.003 | +0.000 |
| 1M | 10 | textbook, averaged over 2nd half | 65536 | 3.74 ± 0.07 | -81.098 ± 0.017 | -0.253 |
| 1M | 10 | textbook, averaged over 2nd half | 8192 | 8.53 ± 0.52 | -81.171 ± 0.025 | -0.327 |
| 1M | 10 | textbook, averaged over 2nd half | 2048 | 19.72 ± 1.31 | -82.339 ± 0.106 | -1.495 |
| 1M | 10 | textbook, averaged over 2nd half | 512 | 42.98 ± 2.39 | -84.869 ± 0.120 | -4.025 |
| 1M | 10 | textbook, last iterate | 65536 | 3.82 ± 0.05 | -81.059 ± 0.014 | -0.215 |
| 1M | 10 | textbook, last iterate | 8192 | 8.33 ± 0.67 | -81.161 ± 0.023 | -0.317 |
| 1M | 10 | textbook, last iterate | 2048 | 19.11 ± 1.64 | -82.337 ± 0.105 | -1.493 |
| 1M | 10 | textbook, last iterate | 512 | 43.00 ± 3.63 | -84.870 ± 0.120 | -4.026 |

### Starting means (time includes the seeding)

| Data | Passes | Random rows | Time (s) | k-means++ seeds | Time (s) | vs random | e-means clustering | Time (s) | vs random |
|---|---|---|---|---|---|---|---|---|---|
| 100k, K = 256 | 1 | -82.640 ± 0.009 | 0.15 | -82.507 ± 0.003 | 0.69 | +0.133 ± 0.012 | -82.248 ± 0.005 | 0.28 | +0.392 ± 0.013 |
| 100k, K = 256 | 2 | -82.081 ± 0.013 | 0.17 | -82.059 ± 0.018 | 0.67 | +0.022 ± 0.008 | -81.992 ± 0.047 | 0.30 | +0.088 ± 0.034 |
| 100k, K = 256 | 3 | -81.988 ± 0.008 | 0.26 | -81.980 ± 0.011 | 0.79 | +0.008 ± 0.019 | -81.953 ± 0.021 | 0.37 | +0.035 ± 0.015 |
| 100k, K = 256 | 5 | -81.936 ± 0.016 | 0.28 | -81.936 ± 0.010 | 0.84 | +0.000 ± 0.018 | -81.937 ± 0.001 | 0.40 | -0.000 ± 0.015 |
| 100k, K = 64 | 1 | -84.816 ± 0.056 | 0.11 | -84.660 ± 0.059 | 0.25 | +0.156 ± 0.108 | -84.384 ± 0.016 | 0.16 | +0.432 ± 0.052 |
| 100k, K = 64 | 2 | -84.336 ± 0.004 | 0.22 | -84.385 ± 0.027 | 0.35 | -0.049 ± 0.029 | -84.300 ± 0.003 | 0.27 | +0.037 ± 0.004 |
| 100k, K = 64 | 3 | -84.291 ± 0.012 | 0.18 | -84.314 ± 0.045 | 0.32 | -0.023 ± 0.055 | -84.273 ± 0.020 | 0.24 | +0.019 ± 0.019 |
| 100k, K = 64 | 5 | -84.282 ± 0.022 | 0.30 | -84.276 ± 0.037 | 0.43 | +0.006 ± 0.043 | -84.278 ± 0.029 | 0.36 | +0.004 ± 0.051 |
| 1M, K = 1024 | 1 | -78.709 ± 0.018 | 1.62 | -78.615 ± 0.006 | 9.81 | +0.094 ± 0.012 | -78.289 ± 0.022 | 3.10 | +0.420 ± 0.028 |
| 1M, K = 1024 | 2 | -78.278 ± 0.005 | 2.86 | -78.264 ± 0.008 | 11.75 | +0.014 ± 0.003 | -78.208 ± 0.004 | 4.51 | +0.069 ± 0.009 |
| 1M, K = 1024 | 3 | -78.207 ± 0.007 | 4.34 | -78.198 ± 0.001 | 14.63 | +0.009 ± 0.007 | -78.173 ± 0.007 | 6.04 | +0.033 ± 0.013 |
| 1M, K = 1024 | 5 | -78.172 ± 0.005 | 6.33 | -78.168 ± 0.005 | 15.64 | +0.003 ± 0.008 | -78.159 ± 0.005 | 7.95 | +0.012 ± 0.009 |
| 1M, K = 256 | 1 | -81.189 ± 0.008 | 0.59 | -81.096 ± 0.020 | 1.22 | +0.094 ± 0.027 | -80.977 ± 0.010 | 1.51 | +0.213 ± 0.016 |
| 1M, K = 256 | 2 | -80.921 ± 0.005 | 0.87 | -80.915 ± 0.005 | 1.63 | +0.005 ± 0.009 | -80.919 ± 0.012 | 1.84 | +0.002 ± 0.010 |
| 1M, K = 256 | 3 | -80.871 ± 0.008 | 1.30 | -80.880 ± 0.002 | 1.89 | -0.009 ± 0.008 | -80.893 ± 0.005 | 2.11 | -0.022 ± 0.005 |
| 1M, K = 256 | 5 | -80.891 ± 0.003 | 1.85 | -80.896 ± 0.010 | 2.53 | -0.004 ± 0.007 | -80.903 ± 0.012 | 2.78 | -0.012 ± 0.009 |

### More passes

| Data | Passes | Random start | Time (s) | k-means++ start | Time (s) |
|---|---|---|---|---|---|
| 100k, K = 256 | 3 | -81.988 ± 0.008 | 0.25 | -81.980 ± 0.011 | 0.76 |
| 100k, K = 256 | 10 | -81.864 ± 0.007 | 0.40 | -81.857 ± 0.021 | 0.93 |
| 100k, K = 256 | 20 | -81.844 ± 0.003 | 0.79 | -81.834 ± 0.017 | 1.27 |
| 100k, K = 256 | 30 | -81.811 ± 0.013 | 0.99 | -81.814 ± 0.021 | 1.50 |
| 100k, K = 256 | 50 | -81.809 ± 0.013 | 1.70 | -81.803 ± 0.009 | 2.19 |
| 100k, K = 64 | 3 | -84.291 ± 0.012 | 0.17 | -84.314 ± 0.045 | 0.29 |
| 100k, K = 64 | 10 | -84.267 ± 0.021 | 0.35 | -84.298 ± 0.052 | 0.48 |
| 100k, K = 64 | 20 | -84.284 ± 0.009 | 0.43 | -84.273 ± 0.026 | 0.56 |
| 100k, K = 64 | 30 | -84.269 ± 0.002 | 0.45 | -84.304 ± 0.037 | 0.58 |
| 100k, K = 64 | 50 | -84.247 ± 0.013 | 0.74 | -84.285 ± 0.040 | 0.85 |
| 1M, K = 256 | 20 | -80.861 ± 0.003 | 6.92 | -80.851 ± 0.007 | 7.62 |
| 1M, K = 256 | 30 | -80.885 ± 0.009 | 8.10 | -80.884 ± 0.009 | 8.64 |

### Small and low-dimensional data (10 random 80/20 splits; held-out log-likelihood)

| Data | Method | Held-out log-lik | vs GaussianMixture defaults | Worst split vs defaults | Time (ms) |
|---|---|---|---|---|---|
| iris 150x4, K=3 | GaussianMixture defaults | -2.424 | +0.000 | +0.000 | 6 |
| iris 150x4, K=3 | GaussianMixture n_init=5 | -2.428 | -0.004 | -0.018 | 4 |
| iris 150x4, K=3 | Online 3 passes (defaults) | -2.482 | -0.058 | -0.379 | 2 |
| iris 150x4, K=3 | Online 10 passes (defaults) | -2.492 | -0.067 | -0.468 | 4 |
| iris 150x4, K=3 | Online 3 passes, random start | -2.590 | -0.166 | -0.515 | 2 |
| iris 150x4, K=3 | Online 10 passes, random start | -2.477 | -0.053 | -0.347 | 3 |
| iris 150x4, K=3 | Online 3 passes, e-means start | -2.420 | +0.005 | -0.094 | 16 |
| iris 150x4, K=3 | Online 3 passes, random start, formula batch size | -2.590 | -0.166 | -0.515 | 2 |
| iris 150x4, K=3 | Online 10 passes, random start, formula batch size | -2.540 | -0.115 | -0.498 | 3 |
| iris 150x4, K=3 | Online 10 passes, random start, no revival | -2.561 | -0.136 | -0.642 | 3 |
| wine 178x13, K=3 | GaussianMixture defaults | -15.452 | +0.000 | +0.000 | 2 |
| wine 178x13, K=3 | GaussianMixture n_init=5 | -15.459 | -0.006 | -0.064 | 6 |
| wine 178x13, K=3 | Online 3 passes (defaults) | -15.555 | -0.102 | -0.703 | 2 |
| wine 178x13, K=3 | Online 10 passes (defaults) | -15.405 | +0.047 | -0.422 | 4 |
| wine 178x13, K=3 | Online 3 passes, random start | -15.416 | +0.036 | -0.423 | 2 |
| wine 178x13, K=3 | Online 10 passes, random start | -15.300 | +0.153 | -0.154 | 4 |
| wine 178x13, K=3 | Online 3 passes, e-means start | -15.579 | -0.127 | -1.374 | 18 |
| wine 178x13, K=3 | Online 3 passes, random start, formula batch size | -15.416 | +0.036 | -0.423 | 2 |
| wine 178x13, K=3 | Online 10 passes, random start, formula batch size | -15.365 | +0.088 | -0.180 | 3 |
| wine 178x13, K=3 | Online 10 passes, random start, no revival | -16.794 | -1.342 | -3.358 | 4 |
| digits (PCA 16) 1797x16, K=10 | GaussianMixture defaults | -20.208 | +0.000 | +0.000 | 14 |
| digits (PCA 16) 1797x16, K=10 | GaussianMixture n_init=5 | -20.100 | +0.108 | +0.000 | 64 |
| digits (PCA 16) 1797x16, K=10 | Online 3 passes (defaults) | -20.339 | -0.131 | -0.613 | 7 |
| digits (PCA 16) 1797x16, K=10 | Online 10 passes (defaults) | -20.206 | +0.002 | -0.306 | 19 |
| digits (PCA 16) 1797x16, K=10 | Online 3 passes, random start | -20.432 | -0.224 | -0.590 | 6 |
| digits (PCA 16) 1797x16, K=10 | Online 10 passes, random start | -20.253 | -0.045 | -0.403 | 19 |
| digits (PCA 16) 1797x16, K=10 | Online 3 passes, e-means start | -20.344 | -0.136 | -0.644 | 28 |
| digits (PCA 16) 1797x16, K=10 | Online 3 passes, random start, formula batch size | -20.282 | -0.074 | -0.477 | 4 |
| digits (PCA 16) 1797x16, K=10 | Online 10 passes, random start, formula batch size | -20.371 | -0.163 | -0.845 | 27 |
| digits (PCA 16) 1797x16, K=10 | Online 10 passes, random start, no revival | -20.931 | -0.723 | -1.223 | 18 |
| blobs 10000x2, K=8 | GaussianMixture defaults | -4.732 | +0.000 | +0.000 | 13 |
| blobs 10000x2, K=8 | GaussianMixture n_init=5 | -4.726 | +0.006 | -0.000 | 51 |
| blobs 10000x2, K=8 | Online 3 passes (defaults) | -4.765 | -0.033 | -0.053 | 28 |
| blobs 10000x2, K=8 | Online 10 passes (defaults) | -4.735 | -0.003 | -0.028 | 87 |
| blobs 10000x2, K=8 | Online 3 passes, random start | -4.758 | -0.027 | -0.059 | 27 |
| blobs 10000x2, K=8 | Online 10 passes, random start | -4.741 | -0.009 | -0.047 | 80 |
| blobs 10000x2, K=8 | Online 3 passes, e-means start | -4.762 | -0.031 | -0.056 | 56 |
| blobs 10000x2, K=8 | Online 3 passes, random start, formula batch size | -4.771 | -0.040 | -0.131 | 28 |
| blobs 10000x2, K=8 | Online 10 passes, random start, formula batch size | -4.769 | -0.037 | -0.124 | 63 |
| blobs 10000x2, K=8 | Online 10 passes, random start, no revival | -4.741 | -0.009 | -0.047 | 74 |
| well-separated blobs 20000x20, K=12 | GaussianMixture defaults | -31.074 | +0.000 | +0.000 | 28 |
| well-separated blobs 20000x20, K=12 | GaussianMixture n_init=5 | -30.850 | +0.224 | +0.000 | 117 |
| well-separated blobs 20000x20, K=12 | Online 3 passes (defaults) | -31.074 | -0.000 | -0.001 | 127 |
| well-separated blobs 20000x20, K=12 | Online 10 passes (defaults) | -31.074 | -0.000 | -0.001 | 120 |
| well-separated blobs 20000x20, K=12 | Online 3 passes, random start | -34.867 | -3.793 | -6.537 | 73 |
| well-separated blobs 20000x20, K=12 | Online 10 passes, random start | -35.862 | -4.788 | -7.794 | 99 |
| well-separated blobs 20000x20, K=12 | Online 3 passes, e-means start | -34.588 | -3.514 | -8.689 | 83 |
| well-separated blobs 20000x20, K=12 | Online 3 passes, random start, formula batch size | -34.867 | -3.793 | -6.537 | 74 |
| well-separated blobs 20000x20, K=12 | Online 10 passes, random start, formula batch size | -35.862 | -4.788 | -7.794 | 100 |
| well-separated blobs 20000x20, K=12 | Online 10 passes, random start, no revival | -36.747 | -5.673 | -11.926 | 98 |
| blobs 100000x2, K=30 | GaussianMixture defaults | -4.907 | +0.000 | +0.000 | 407 |
| blobs 100000x2, K=30 | Online 3 passes (defaults) | -4.949 | -0.042 | -0.078 | 319 |
| blobs 100000x2, K=30 | Online 10 passes (defaults) | -4.944 | -0.037 | -0.073 | 769 |
| blobs 100000x2, K=30 | Online 3 passes, random start | -4.953 | -0.047 | -0.084 | 309 |
| blobs 100000x2, K=30 | Online 10 passes, random start | -4.945 | -0.038 | -0.054 | 757 |
| blobs 100000x2, K=30 | Online 3 passes, e-means start | -4.964 | -0.057 | -0.114 | 377 |
| blobs 100000x2, K=30 | Online 3 passes, random start, formula batch size | -4.963 | -0.056 | -0.091 | 169 |
| blobs 100000x2, K=30 | Online 10 passes, random start, formula batch size | -4.970 | -0.063 | -0.127 | 215 |
| blobs 100000x2, K=30 | Online 10 passes, random start, no revival | -4.975 | -0.069 | -0.155 | 761 |
| synthetic mixture 60000x16, K=24 | GaussianMixture defaults | -21.756 | +0.000 | +0.000 | 172 |
| synthetic mixture 60000x16, K=24 | Online 3 passes (defaults) | -21.901 | -0.145 | -0.444 | 264 |
| synthetic mixture 60000x16, K=24 | Online 10 passes (defaults) | -21.783 | -0.027 | -0.355 | 777 |
| synthetic mixture 60000x16, K=24 | Online 3 passes, random start | -21.846 | -0.090 | -0.391 | 230 |
| synthetic mixture 60000x16, K=24 | Online 10 passes, random start | -21.867 | -0.111 | -0.489 | 761 |
| synthetic mixture 60000x16, K=24 | Online 3 passes, e-means start | -21.780 | -0.024 | -0.402 | 268 |
| synthetic mixture 60000x16, K=24 | Online 3 passes, random start, formula batch size | -21.905 | -0.149 | -0.459 | 106 |
| synthetic mixture 60000x16, K=24 | Online 10 passes, random start, formula batch size | -21.954 | -0.198 | -0.720 | 224 |
| synthetic mixture 60000x16, K=24 | Online 10 passes, random start, no revival | -21.926 | -0.170 | -0.488 | 762 |
| SIFT first 8 PCA dims 120000x8, K=64 | GaussianMixture defaults | -8.604 | +0.000 | +0.000 | 3880 |
| SIFT first 8 PCA dims 120000x8, K=64 | Online 3 passes (defaults) | -8.591 | +0.012 | -0.002 | 554 |
| SIFT first 8 PCA dims 120000x8, K=64 | Online 10 passes (defaults) | -8.578 | +0.026 | +0.012 | 1328 |
| SIFT first 8 PCA dims 120000x8, K=64 | Online 3 passes, random start | -8.590 | +0.014 | -0.014 | 486 |
| SIFT first 8 PCA dims 120000x8, K=64 | Online 10 passes, random start | -8.577 | +0.027 | +0.009 | 1289 |
| SIFT first 8 PCA dims 120000x8, K=64 | Online 3 passes, e-means start | -8.593 | +0.010 | -0.008 | 536 |
| SIFT first 8 PCA dims 120000x8, K=64 | Online 3 passes, random start, formula batch size | -8.599 | +0.004 | -0.016 | 154 |
| SIFT first 8 PCA dims 120000x8, K=64 | Online 10 passes, random start, formula batch size | -8.588 | +0.015 | -0.001 | 327 |
| SIFT first 8 PCA dims 120000x8, K=64 | Online 10 passes, random start, no revival | -8.579 | +0.025 | +0.008 | 1260 |
| SIFT first 16 PCA dims 120000x16, K=64 | GaussianMixture defaults | -18.686 | +0.000 | +0.000 | 4081 |
| SIFT first 16 PCA dims 120000x16, K=64 | Online 3 passes (defaults) | -18.674 | +0.012 | -0.006 | 551 |
| SIFT first 16 PCA dims 120000x16, K=64 | Online 10 passes (defaults) | -18.656 | +0.030 | +0.017 | 1363 |
| SIFT first 16 PCA dims 120000x16, K=64 | Online 3 passes, random start | -18.682 | +0.004 | -0.020 | 490 |
| SIFT first 16 PCA dims 120000x16, K=64 | Online 10 passes, random start | -18.665 | +0.021 | +0.003 | 1317 |
| SIFT first 16 PCA dims 120000x16, K=64 | Online 3 passes, e-means start | -18.678 | +0.008 | -0.009 | 556 |
| SIFT first 16 PCA dims 120000x16, K=64 | Online 3 passes, random start, formula batch size | -18.698 | -0.012 | -0.035 | 176 |
| SIFT first 16 PCA dims 120000x16, K=64 | Online 10 passes, random start, formula batch size | -18.689 | -0.004 | -0.042 | 367 |
| SIFT first 16 PCA dims 120000x16, K=64 | Online 10 passes, random start, no revival | -18.676 | +0.010 | -0.019 | 1294 |
