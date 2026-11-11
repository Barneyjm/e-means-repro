# Real-index check: Faiss k-means, the authors' e-means, and `EMeans` (after the split-offset fix)

Made by `experiments/summarize_check.py results/check`. 10,000 held-out queries, 3 seeds, CPU (Apple M2 Max), Faiss 1.16.0. See EXTENSIONS.md section 14.

### ccnews: 300,000 vectors, 768 dimensions, k = 1,024

| Centroids from | Seeds | Training time (s) | Held-out error | List imbalance | Distance computations, 95% R@1 | Change | Query time (µs) | Change |
|---|---|---|---|---|---|---|---|---|
| Faiss k-means, 25 iterations | 3 | 15.9 | – | 1.20 | 4,164 | – | 100 | – |
| authors' e-means, 3 passes | 3 | 3.1 | -0.2% ± 0.1 | 1.24 | 4,313 | +3.7% ± 2.6 | 104 | +3.4% ± 3.8 |
| authors' e-means, 5 passes | 3 | 5.3 | -0.4% ± 0.1 | 1.24 | 4,380 | +5.4% ± 5.9 | 105 | +5.1% ± 6.3 |
| authors' e-means, 3 passes + penalty 0.1, held | 3 | 3.4 | -0.2% ± 0.1 | 1.08 | 4,022 | -3.3% ± 2.8 | 97 | -3.1% ± 2.9 |
| authors' e-means, 3 passes + penalty 0.3, faded | 3 | 3.4 | -0.2% ± 0.1 | 1.08 | 4,052 | -2.7% ± 2.7 | 98 | -2.7% ± 2.8 |
| my estimator, 3 passes | 3 | 2.2 | -0.2% ± 0.1 | 1.24 | 4,234 | +1.9% ± 5.5 | 101 | +1.1% ± 5.5 |
| my estimator, 5 passes | 3 | 3.4 | -0.4% ± 0.1 | 1.24 | 4,189 | +0.7% ± 3.5 | 101 | +0.8% ± 3.4 |
| my estimator, 3 passes + penalty 0.1, held | 3 | 2.2 | -0.1% ± 0.1 | 1.08 | 4,128 | -0.6% ± 6.7 | 100 | +0.3% ± 7.5 |
| my estimator, 3 passes + penalty 0.3, faded | 3 | 2.2 | -0.1% ± 0.0 | 1.08 | 4,042 | -2.9% ± 3.8 | 98 | -2.2% ± 3.9 |

### ccnews: 300,000 vectors, 768 dimensions, k = 4,096

| Centroids from | Seeds | Training time (s) | Held-out error | List imbalance | Distance computations, 95% R@1 | Change | Query time (µs) | Change |
|---|---|---|---|---|---|---|---|---|
| Faiss k-means, 25 iterations | 3 | 60.5 | – | 1.33 | 2,170 | – | 56 | – |
| authors' e-means, 3 passes | 3 | 9.0 | -0.7% ± 0.1 | 1.38 | 2,205 | +1.8% ± 5.9 | 56 | -0.1% ± 4.7 |
| authors' e-means, 5 passes | 3 | 14.6 | -1.0% ± 0.2 | 1.38 | 2,133 | -1.6% ± 4.2 | 55 | -2.2% ± 3.8 |
| authors' e-means, 3 passes + penalty 0.1, held | 3 | 9.5 | -0.8% ± 0.1 | 1.10 | 1,993 | -8.0% ± 7.2 | 52 | -7.7% ± 6.1 |
| authors' e-means, 3 passes + penalty 0.3, faded | 3 | 9.5 | -0.8% ± 0.1 | 1.09 | 1,988 | -8.3% ± 2.4 | 52 | -7.1% ± 1.0 |
| my estimator, 3 passes | 3 | 6.0 | -0.8% ± 0.1 | 1.38 | 2,117 | -2.4% ± 3.8 | 54 | -3.5% ± 3.0 |
| my estimator, 5 passes | 3 | 8.2 | -1.0% ± 0.2 | 1.38 | 2,174 | +0.3% ± 3.2 | 56 | +0.1% ± 3.5 |
| my estimator, 3 passes + penalty 0.1, held | 3 | 6.0 | -0.8% ± 0.1 | 1.10 | 2,038 | -6.1% ± 1.9 | 53 | -5.3% ± 2.7 |
| my estimator, 3 passes + penalty 0.3, faded | 3 | 6.0 | -0.8% ± 0.1 | 1.09 | 2,078 | -4.3% ± 4.7 | 54 | -3.9% ± 3.8 |

### landmark: 300,000 vectors, 768 dimensions, k = 1,024

| Centroids from | Seeds | Training time (s) | Held-out error | List imbalance | Distance computations, 95% R@1 | Change | Query time (µs) | Change |
|---|---|---|---|---|---|---|---|---|
| Faiss k-means, 25 iterations | 3 | 16.0 | – | 1.15 | 1,814 | – | 45 | – |
| authors' e-means, 3 passes | 3 | 3.2 | -0.2% ± 0.0 | 1.15 | 1,816 | +0.2% ± 1.0 | 45 | +0.2% ± 1.7 |
| authors' e-means, 5 passes | 3 | 5.2 | -0.3% ± 0.1 | 1.15 | 1,804 | -0.5% ± 1.0 | 44 | -0.7% ± 1.9 |
| authors' e-means, 3 passes + penalty 0.1, held | 3 | 3.4 | -0.2% ± 0.1 | 1.07 | 1,784 | -1.6% ± 0.4 | 44 | -1.4% ± 2.1 |
| authors' e-means, 3 passes + penalty 0.3, faded | 3 | 3.4 | -0.1% ± 0.1 | 1.07 | 1,743 | -3.9% ± 0.9 | 43 | -4.2% ± 0.3 |
| my estimator, 3 passes | 3 | 2.2 | -0.2% ± 0.1 | 1.16 | 1,790 | -1.3% ± 1.7 | 44 | -1.2% ± 1.5 |
| my estimator, 5 passes | 3 | 3.5 | -0.4% ± 0.0 | 1.16 | 1,785 | -1.6% ± 2.5 | 43 | -2.5% ± 2.6 |
| my estimator, 3 passes + penalty 0.1, held | 3 | 2.3 | -0.2% ± 0.1 | 1.07 | 1,741 | -4.0% ± 1.3 | 43 | -3.7% ± 0.5 |
| my estimator, 3 passes + penalty 0.3, faded | 3 | 2.3 | -0.1% ± 0.1 | 1.07 | 1,771 | -2.4% ± 1.0 | 44 | -2.3% ± 1.2 |

### landmark: 300,000 vectors, 768 dimensions, k = 4,096

| Centroids from | Seeds | Training time (s) | Held-out error | List imbalance | Distance computations, 95% R@1 | Change | Query time (µs) | Change |
|---|---|---|---|---|---|---|---|---|
| Faiss k-means, 25 iterations | 3 | 60.6 | – | 1.21 | 716 | – | 21 | – |
| authors' e-means, 3 passes | 3 | 9.1 | -0.4% ± 0.1 | 1.19 | 684 | -4.6% ± 1.8 | 20 | -6.9% ± 3.7 |
| authors' e-means, 5 passes | 3 | 14.6 | -0.6% ± 0.2 | 1.19 | 684 | -4.5% ± 2.8 | 20 | -7.3% ± 3.4 |
| authors' e-means, 3 passes + penalty 0.1, held | 3 | 9.5 | -0.6% ± 0.2 | 1.08 | 657 | -8.3% ± 1.8 | 19 | -11.0% ± 3.0 |
| authors' e-means, 3 passes + penalty 0.3, faded | 3 | 9.5 | -0.6% ± 0.2 | 1.07 | 661 | -7.6% ± 3.7 | 19 | -9.6% ± 4.2 |
| my estimator, 3 passes | 3 | 5.9 | -0.4% ± 0.1 | 1.19 | 692 | -3.3% ± 3.2 | 20 | -5.3% ± 5.7 |
| my estimator, 5 passes | 3 | 8.3 | -0.7% ± 0.2 | 1.19 | 679 | -5.2% ± 1.2 | 20 | -6.3% ± 4.1 |
| my estimator, 3 passes + penalty 0.1, held | 3 | 6.1 | -0.6% ± 0.2 | 1.08 | 666 | -7.0% ± 1.5 | 19 | -8.8% ± 4.4 |
| my estimator, 3 passes + penalty 0.3, faded | 3 | 6.1 | -0.5% ± 0.1 | 1.07 | 677 | -5.5% ± 1.7 | 20 | -7.7% ± 4.8 |

### lastfm: 252,385 vectors, 65 dimensions, k = 1,024

| Centroids from | Seeds | Training time (s) | Held-out error | List imbalance | Distance computations, 95% R@1 | Change | Query time (µs) | Change |
|---|---|---|---|---|---|---|---|---|
| Faiss k-means, 25 iterations | 3 | 12.1 | – | 6.49 | 10,901 | – | 12 | – |
| authors' e-means, 3 passes | 3 | 1.6 | -0.9% ± 0.2 | 3.91 | 8,787 | -19.3% ± 4.1 | 11 | -7.2% ± 3.7 |
| authors' e-means, 3 passes + penalty 0.1, held | 3 | 1.6 | -0.2% ± 0.2 | 3.89 | 8,860 | -18.7% ± 3.1 | 12 | +2.1% ± 2.4 |
| authors' e-means, 3 passes + penalty 0.3, faded | 3 | 1.7 | -1.0% ± 0.2 | 3.61 | 9,547 | -12.4% ± 2.6 | 13 | +8.7% ± 6.1 |
| my estimator, 3 passes | 3 | 0.4 | -0.6% ± 0.5 | 4.30 | 9,181 | -15.8% ± 0.9 | 11 | -8.0% ± 6.7 |
| my estimator, 5 passes | 3 | 0.7 | -2.3% ± 0.0 | 5.27 | 10,222 | -6.2% ± 2.5 | 12 | -2.4% ± 3.2 |
| my estimator, 3 passes + penalty 0.1, held | 3 | 0.5 | +0.2% ± 0.3 | 3.23 | 7,711 | -29.2% ± 8.6 | 11 | -7.5% ± 5.6 |
| my estimator, 3 passes + penalty 0.3, faded | 3 | 0.5 | -0.4% ± 0.1 | 3.62 | 9,692 | -11.1% ± 3.4 | 13 | +7.7% ± 3.3 |

### lastfm: 252,385 vectors, 65 dimensions, k = 4,096

| Centroids from | Seeds | Training time (s) | Held-out error | List imbalance | Distance computations, 95% R@1 | Change | Query time (µs) | Change |
|---|---|---|---|---|---|---|---|---|
| Faiss k-means, 25 iterations | 3 | 50.7 | – | 6.63 | 3,126 | – | 11 | – |
| authors' e-means, 3 passes | 3 | 6.3 | -1.5% ± 0.2 | 3.06 | 2,008 | -35.7% ± 1.5 | 11 | +2.8% ± 2.1 |
| authors' e-means, 3 passes + penalty 0.1, held | 3 | 6.5 | -0.6% ± 0.2 | 4.31 | 3,044 | -2.6% ± 6.0 | 11 | +6.1% ± 6.2 |
| authors' e-means, 3 passes + penalty 0.3, faded | 3 | 6.5 | -1.0% ± 0.2 | 4.00 | 3,329 | +6.4% ± 5.6 | 11 | +1.1% ± 5.9 |
| my estimator, 3 passes | 3 | 1.2 | -1.4% ± 0.2 | 2.62 | 2,048 | -34.5% ± 1.6 | 10 | -4.4% ± 2.8 |
| my estimator, 5 passes | 3 | 1.9 | -2.4% ± 0.5 | 2.96 | 2,347 | -24.9% ± 2.0 | 10 | -2.8% ± 4.9 |
| my estimator, 3 passes + penalty 0.1, held | 3 | 1.2 | -0.5% ± 0.1 | 3.89 | 2,709 | -13.2% ± 5.6 | 11 | +0.2% ± 6.9 |
| my estimator, 3 passes + penalty 0.3, faded | 3 | 1.2 | -0.8% ± 0.1 | 3.78 | 3,273 | +4.7% ± 0.7 | 11 | +0.9% ± 4.7 |

### lastfmdedup: 249,698 vectors, 65 dimensions, k = 4,096

| Centroids from | Seeds | Training time (s) | Held-out error | List imbalance | Distance computations, 95% R@1 | Change | Query time (µs) | Change |
|---|---|---|---|---|---|---|---|---|
| Faiss k-means, 25 iterations | 3 | 49.2 | – | 6.25 | 2,911 | – | 10 | – |
| authors' e-means, 3 passes | 3 | 6.0 | -1.3% ± 0.3 | 2.42 | 1,998 | -31.2% ± 3.9 | 10 | +0.5% ± 2.4 |
| authors' e-means, 3 passes + penalty 0.1, held | 3 | 6.2 | -0.6% ± 0.2 | 2.77 | 2,546 | -12.2% ± 9.5 | 10 | -1.1% ± 3.6 |
| authors' e-means, 3 passes + penalty 0.3, faded | 3 | 6.0 | -1.0% ± 0.7 | 3.02 | 2,891 | -0.6% ± 4.9 | 10 | -1.0% ± 2.4 |
| my estimator, 3 passes | 3 | 1.1 | -1.3% ± 0.1 | 2.81 | 2,097 | -27.8% ± 4.6 | 11 | +4.2% ± 2.0 |
| my estimator, 3 passes + penalty 0.1, held | 3 | 1.2 | -0.8% ± 0.4 | 2.63 | 2,535 | -13.1% ± 5.6 | 10 | +1.1% ± 3.7 |
| my estimator, 3 passes + penalty 0.3, faded | 3 | 1.2 | -1.1% ± 0.3 | 3.97 | 3,103 | +6.7% ± 3.2 | 11 | +6.9% ± 0.9 |

### nyt256: 250,000 vectors, 256 dimensions, k = 1,024

| Centroids from | Seeds | Training time (s) | Held-out error | List imbalance | Distance computations, 95% R@1 | Change | Query time (µs) | Change |
|---|---|---|---|---|---|---|---|---|
| Faiss k-means, 25 iterations | 3 | 12.8 | – | 4.01 | 63,583 | – | 422 | – |
| authors' e-means, 3 passes | 3 | 2.0 | -2.7% ± 2.5 | 1.71 | 51,769 | -13.2% ± 24.5 | 341 | -13.5% ± 25.2 |
| authors' e-means, 5 passes | 3 | 3.4 | -3.4% ± 2.3 | 1.69 | 55,996 | -7.3% ± 21.9 | 365 | -8.6% ± 23.2 |
| authors' e-means, 3 passes + penalty 0.1, held | 3 | 2.1 | -2.9% ± 2.3 | 1.16 | 25,453 | -60.8% ± 3.8 | 164 | -61.9% ± 4.0 |
| authors' e-means, 3 passes + penalty 0.3, faded | 3 | 2.1 | -2.9% ± 2.4 | 1.25 | 33,975 | -43.1% ± 16.0 | 217 | -45.0% ± 16.0 |
| my estimator, 3 passes | 3 | 0.9 | -3.0% ± 2.1 | 1.67 | 51,394 | -13.5% ± 25.4 | 336 | -14.6% ± 25.4 |
| my estimator, 5 passes | 3 | 1.4 | -3.5% ± 2.2 | 1.63 | 50,626 | -15.3% ± 22.7 | 329 | -16.8% ± 22.7 |
| my estimator, 3 passes + penalty 0.1, held | 3 | 0.9 | -3.0% ± 2.1 | 1.15 | 28,608 | -52.8% ± 10.6 | 185 | -53.8% ± 10.7 |
| my estimator, 3 passes + penalty 0.3, faded | 3 | 0.9 | -3.1% ± 2.2 | 1.23 | 34,641 | -42.1% ± 15.5 | 223 | -43.9% ± 14.9 |

### nyt256: 250,000 vectors, 256 dimensions, k = 4,096

| Centroids from | Seeds | Training time (s) | Held-out error | List imbalance | Distance computations, 95% R@1 | Change | Query time (µs) | Change |
|---|---|---|---|---|---|---|---|---|
| Faiss k-means, 25 iterations | 3 | 51.4 | – | 7.34 | 45,441 | – | 287 | – |
| authors' e-means, 3 passes | 3 | 7.2 | -3.8% ± 0.1 | 1.99 | 25,009 | -45.0% ± 1.0 | 166 | -42.0% ± 0.7 |
| authors' e-means, 5 passes | 3 | 12.0 | -5.0% ± 0.0 | 1.80 | 20,037 | -55.5% ± 6.1 | 129 | -54.7% ± 6.5 |
| authors' e-means, 3 passes + penalty 0.1, held | 3 | 7.1 | -5.0% ± 0.1 | 1.16 | 7,527 | -83.4% ± 0.9 | 52 | -81.8% ± 1.0 |
| authors' e-means, 3 passes + penalty 0.3, faded | 3 | 7.1 | -5.1% ± 0.1 | 1.24 | 11,273 | -75.2% ± 2.3 | 75 | -74.0% ± 2.5 |
| my estimator, 3 passes | 3 | 2.1 | -3.6% ± 0.0 | 2.06 | 26,686 | -41.0% ± 5.1 | 178 | -37.5% ± 5.8 |
| my estimator, 5 passes | 3 | 3.3 | -4.9% ± 0.1 | 1.83 | 21,570 | -52.2% ± 5.2 | 140 | -50.7% ± 5.4 |
| my estimator, 3 passes + penalty 0.1, held | 3 | 2.1 | -4.9% ± 0.2 | 1.16 | 6,979 | -84.6% ± 0.9 | 49 | -82.7% ± 1.0 |
| my estimator, 3 passes + penalty 0.3, faded | 3 | 2.1 | -5.2% ± 0.2 | 1.20 | 10,145 | -77.6% ± 1.6 | 68 | -76.1% ± 1.5 |

### nytdedup: 217,938 vectors, 256 dimensions, k = 4,096

| Centroids from | Seeds | Training time (s) | Held-out error | List imbalance | Distance computations, 95% R@1 | Change | Query time (µs) | Change |
|---|---|---|---|---|---|---|---|---|
| Faiss k-means, 25 iterations | 3 | 44.8 | – | 1.56 | | | | |
| authors' e-means, 3 passes | 3 | 6.2 | -0.1% ± 0.2 | 1.67 | | | | |
| authors' e-means, 3 passes + penalty 0.1, held | 3 | 6.2 | -0.5% ± 0.1 | 1.12 | | | | |
| authors' e-means, 3 passes + penalty 0.3, faded | 3 | 6.2 | -0.4% ± 0.2 | 1.14 | | | | |
| my estimator, 3 passes | 3 | 1.8 | -0.1% ± 0.1 | 1.69 | | | | |
| my estimator, 3 passes + penalty 0.1, held | 3 | 1.8 | -0.4% ± 0.1 | 1.12 | | | | |
| my estimator, 3 passes + penalty 0.3, faded | 3 | 1.8 | -0.4% ± 0.1 | 1.14 | | | | |

### sift: 1,000,000 vectors, 128 dimensions, k = 16,384

| Centroids from | Seeds | Training time (s) | Held-out error | List imbalance | Distance computations, 95% R@1 | Change | Query time (µs) | Change |
|---|---|---|---|---|---|---|---|---|
| Faiss k-means, 25 iterations | 3 | 816.0 | – | 1.22 | 4,366 | – | 47 | – |
| authors' e-means, 3 passes | 3 | 100.8 | -0.0% ± 0.1 | 1.19 | 4,332 | -0.8% ± 1.9 | 44 | -6.3% ± 3.6 |
| authors' e-means, 5 passes | 3 | 169.4 | -0.2% ± 0.0 | 1.19 | 4,081 | -6.5% ± 1.7 | 43 | -7.5% ± 1.1 |
| authors' e-means, 3 passes + penalty 0.1, held | 3 | 100.6 | -0.3% ± 0.1 | 1.06 | 3,968 | -9.1% ± 4.2 | 43 | -8.8% ± 6.0 |
| authors' e-means, 3 passes + penalty 0.3, faded | 3 | 101.5 | -0.4% ± 0.1 | 1.06 | 3,895 | -10.8% ± 0.9 | 42 | -10.3% ± 3.1 |
| my estimator, 3 passes | 3 | 12.5 | +0.1% ± 0.1 | 1.19 | 4,273 | -2.1% ± 2.7 | 44 | -6.1% ± 5.5 |
| my estimator, 5 passes | 3 | 19.2 | -0.1% ± 0.1 | 1.19 | 4,086 | -6.4% ± 2.3 | 43 | -7.1% ± 2.7 |
| my estimator, 3 passes + penalty 0.1, held | 3 | 12.8 | -0.3% ± 0.0 | 1.06 | 3,930 | -10.0% ± 1.4 | 42 | -10.6% ± 3.2 |
| my estimator, 3 passes + penalty 0.3, faded | 3 | 12.5 | -0.3% ± 0.0 | 1.06 | 3,830 | -12.3% ± 2.7 | 42 | -9.9% ± 1.9 |

### yahoo: 300,000 vectors, 384 dimensions, k = 1,024

| Centroids from | Seeds | Training time (s) | Held-out error | List imbalance | Distance computations, 95% R@1 | Change | Query time (µs) | Change |
|---|---|---|---|---|---|---|---|---|
| Faiss k-means, 25 iterations | 3 | 15.8 | – | 1.10 | 5,166 | – | 60 | – |
| authors' e-means, 3 passes | 3 | 2.6 | -0.0% ± 0.1 | 1.12 | 5,119 | -0.8% ± 4.6 | 59 | -1.5% ± 4.9 |
| authors' e-means, 5 passes | 3 | 4.4 | -0.2% ± 0.1 | 1.12 | 5,109 | -1.0% ± 5.0 | 61 | +0.7% ± 7.8 |
| authors' e-means, 3 passes + penalty 0.1, held | 3 | 2.7 | -0.0% ± 0.1 | 1.04 | 5,336 | +3.4% ± 4.7 | 64 | +5.2% ± 4.5 |
| authors' e-means, 3 passes + penalty 0.3, faded | 3 | 2.6 | +0.1% ± 0.0 | 1.04 | 5,372 | +4.1% ± 5.2 | 63 | +4.1% ± 4.7 |
| my estimator, 3 passes | 3 | 1.4 | +0.0% ± 0.1 | 1.13 | 4,975 | -3.6% ± 1.2 | 59 | -3.0% ± 1.6 |
| my estimator, 5 passes | 3 | 2.2 | -0.2% ± 0.0 | 1.12 | 5,082 | -1.6% ± 1.1 | 60 | -1.3% ± 1.6 |
| my estimator, 3 passes + penalty 0.1, held | 3 | 1.4 | -0.0% ± 0.1 | 1.04 | 5,323 | +3.2% ± 6.0 | 63 | +4.6% ± 6.3 |
| my estimator, 3 passes + penalty 0.3, faded | 3 | 1.4 | +0.1% ± 0.1 | 1.04 | 5,167 | -0.0% ± 2.2 | 61 | +1.6% ± 2.2 |

### yahoo: 300,000 vectors, 384 dimensions, k = 4,096

| Centroids from | Seeds | Training time (s) | Held-out error | List imbalance | Distance computations, 95% R@1 | Change | Query time (µs) | Change |
|---|---|---|---|---|---|---|---|---|
| Faiss k-means, 25 iterations | 3 | 64.1 | – | 1.17 | 2,238 | – | 30 | – |
| authors' e-means, 3 passes | 3 | 8.9 | -0.3% ± 0.0 | 1.19 | 2,230 | -0.3% ± 6.3 | 30 | -0.7% ± 4.8 |
| authors' e-means, 5 passes | 3 | 14.6 | -0.6% ± 0.1 | 1.18 | 2,175 | -2.7% ± 5.3 | 29 | -2.9% ± 5.0 |
| authors' e-means, 3 passes + penalty 0.1, held | 3 | 8.9 | -0.5% ± 0.1 | 1.04 | 2,233 | -0.2% ± 2.7 | 30 | +0.3% ± 3.2 |
| authors' e-means, 3 passes + penalty 0.3, faded | 3 | 8.8 | -0.5% ± 0.1 | 1.04 | 2,233 | -0.2% ± 3.1 | 30 | -1.0% ± 3.3 |
| my estimator, 3 passes | 3 | 3.1 | -0.3% ± 0.1 | 1.19 | 2,358 | +5.4% ± 3.7 | 32 | +6.2% ± 2.1 |
| my estimator, 5 passes | 3 | 4.0 | -0.5% ± 0.0 | 1.19 | 2,292 | +2.5% ± 4.7 | 30 | +1.1% ± 3.5 |
| my estimator, 3 passes + penalty 0.1, held | 3 | 3.2 | -0.5% ± 0.0 | 1.04 | 2,215 | -1.1% ± 0.6 | 30 | -1.1% ± 1.1 |
| my estimator, 3 passes + penalty 0.3, faded | 3 | 3.1 | -0.4% ± 0.1 | 1.04 | 2,262 | +1.1% ± 0.5 | 31 | +2.1% ± 1.3 |

### nytdedup at lower recall targets (95% is not reached within 256 probes)

| Centroids from | R@1 70%: distance computations | Change | R@1 80%: distance computations | Change | R@1 85%: distance computations | Change |
|---|---|---|---|---|---|---|
| Faiss k-means, 25 iterations | 1,974 | – | 8,744 | – | 13,790 | – |
| authors' e-means, 3 passes | 3,078 | +56% ± 13 [3] | 12,639 | +46% ± 24 [3] | 18,964 | +38% ± 17 [3] |
| authors' e-means, 3 passes + penalty 0.1, held | 1,972 | +0% ± 11 [3] | 9,734 | +14% ± 27 [3] | 11,527 | -26% ± 0 [1] |
| authors' e-means, 3 passes + penalty 0.3, faded | 2,454 | +25% ± 8 [3] | 11,768 | +36% ± 21 [3] | 16,510 | +15% ± 10 [2] |
| my estimator, 3 passes | 3,083 | +56% ± 7 [3] | 12,729 | +47% ± 20 [3] | 18,975 | +39% ± 15 [3] |
| my estimator, 3 passes + penalty 0.1, held | 2,309 | +17% ± 22 [3] | 12,181 | +39% ± 37 [3] | 11,631 | -13% ± 0 [1] |
| my estimator, 3 passes + penalty 0.3, faded | 2,151 | +9% ± 7 [3] | 10,487 | +21% ± 16 [3] | 14,104 | +13% ± 0 [1] |
