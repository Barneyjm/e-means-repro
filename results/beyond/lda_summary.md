# LDA with the e-means ingredients: UCI NYTimes

Setup: 8,000-word vocabulary (the 100 most document-frequent words dropped), 10,000 held-out documents, K = 100,
alpha = eta = 1/K, Apple M2 Max GPU. Score: held-out per-word perplexity by document completion (lower is better).
Every method uses the same E-step (cold start, up to 100 iterations, tolerance 1e-3 on the mean change of gamma).
"Batch passes to match" = first full-batch pass whose perplexity is at least as good, same seed.
Raw log: `lda_nyt.log`. Scripts: `experiments/lda_*.py`.

## Check against scikit-learn (3,000 documents, K = 20, CPU)

| batch passes | perplexity, scikit-learn | this code, same gamma start | gap | median rel. diff. of lambda | this code, gamma start 1 | gap |
|---|---|---|---|---|---|---|
| 1 | 4019.76 | 4019.76 | +0.0000% | 4.3e-05 | 4014.52 | -0.130% |
| 2 | 3497.72 | 3497.72 | +0.0001% | 4.7e-05 | 3488.61 | -0.260% |
| 5 | 3041.46 | 3041.46 | +0.0000% | 1.5e-05 | 3030.82 | -0.350% |
| 10 | 2937.95 | 2937.95 | -0.0001% | 0.0e+00 | 2936.96 | -0.034% |

one online update (rho = 0.1866): perplexity 4657.10 scikit-learn, 4657.09 here; median relative difference of lambda 9.4e-06

The experiments stop the E-step on the block's mean change rather than per document; on this check that changes the
perplexity after 10 passes from 2936.96 to 2936.67.

### NYTimes, 100,000 documents, K = 100

Full-batch variational EM, seeds [0, 1, 2] (held-out perplexity, lower is better)

| passes | perplexity | train s | dead topics |
|---|---|---|---|
| 1 | 3474.3 ± 30.0 | 11.4 | 0.0 |
| 2 | 2503.0 ± 20.1 | 22.9 | 0.3 |
| 3 | 2253.0 ± 10.0 | 34.3 | 0.3 |
| 5 | 2115.3 ± 2.6 | 51.7 | 0.0 |
| 10 | 2071.2 ± 1.1 | 86.4 | 0.0 |
| 15 | 2070.8 ± 0.8 | 117.4 | 0.0 |
| 20 | 2073.8 ± 1.8 | 146.4 | 0.0 |

main

| run | passes | seeds | perplexity | train s | batch size | peak step | updates | repaired | merged | dead at end | batch passes to match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recipe | 1 | 3 | 2207.6 ± 2.4 | 8.7 | 2048 | 0.532 | 48 | 0.0 | 0.0 | 0.0 | 4, 4, 4 |
| recipe | 2 | 3 | 2289.4 ± 1.9 | 15.5 | 2048 | 0.493 | 96 | 0.3 | 0.0 | 0.0 | 3, 3, 3 |
| recipe | 3 | 3 | 2294.6 ± 7.1 | 17.6 | 4096 | 0.88 | 72 | 0.3 | 0.0 | 0.0 | 3, 3, 3 |
| recipe | 5 | 3 | 2202.7 ± 4.5 | 30.6 | 8192 | 1.0 | 60 | 0.3 | 0.0 | 0.0 | 4, 4, 4 |
| recipe | 10 | 3 | 2233.2 ± 6.5 | 56.3 | 8192 | 1.0 | 120 | 0.3 | 0.0 | 0.0 | 4, 4, 4 |

tuned

| run | passes | seeds | perplexity | train s | batch size | peak step | updates | repaired | merged | dead at end | batch passes to match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recipe, step/4 | 1 | 3 | 2159.5 ± 4.1 | 10.8 | 2048 | 0.133 | 48 | 3.7 | 0.0 | 0.0 | 5, 4, 4 |
| recipe, step/4 | 2 | 3 | 2107.7 ± 2.1 | 19.9 | 2048 | 0.123 | 96 | 4.7 | 0.0 | 0.0 | 6, 6, 6 |
| recipe, step/4 | 3 | 3 | 2101.4 ± 16.1 | 21.8 | 4096 | 0.22 | 72 | 5.7 | 0.0 | 0.0 | 7, 5, 7 |
| recipe, step/4 | 5 | 3 | 2078.0 ± 3.5 | 36.5 | 8192 | 0.25 | 60 | 5.0 | 0.0 | 0.0 | 8, 7, 9 |
| recipe, step/4 | 10 | 3 | 2089.6 ± 3.0 | 66.3 | 8192 | 0.25 | 120 | 5.0 | 0.0 | 0.0 | 7, 6, 7 |

hoffman

| run | passes | seeds | perplexity | train s | batch size | peak step | updates | repaired | merged | dead at end | batch passes to match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Hoffman 2010 best (B=4096, kappa=0.5, tau0=64) | 1 | 3 | 2146.2 ± 16.1 | 8.7 | 4096 | – | 24 | 0.0 | 0.0 | 9.7 | 5, 4, 5 |
| Hoffman 2010 best (B=4096, kappa=0.5, tau0=64) | 2 | 3 | 2104.0 ± 15.6 | 15.6 | 4096 | – | 48 | 0.0 | 0.0 | 7.7 | 7, 5, 6 |
| Hoffman 2010 best (B=4096, kappa=0.5, tau0=64) | 3 | 3 | 2097.7 ± 15.3 | 22.3 | 4096 | – | 72 | 0.0 | 0.0 | 6.7 | 8, 5, 6 |
| Hoffman 2010 best (B=4096, kappa=0.5, tau0=64) | 5 | 3 | 2100.4 ± 14.6 | 34.7 | 4096 | – | 120 | 0.0 | 0.0 | 5.0 | 7, 5, 6 |
| Hoffman 2010 best (B=4096, kappa=0.5, tau0=64) | 10 | 3 | 2108.4 ± 11.1 | 64.4 | 4096 | – | 240 | 0.0 | 0.0 | 3.3 | 7, 5, 6 |

ablate

| run | passes | seeds | perplexity | train s | batch size | peak step | updates | repaired | merged | dead at end | batch passes to match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| constant step (mean of the cosine curve) | 3 | 3 | 2089.4 ± 12.9 | 21.9 | 4096 | 0.121 | 72 | 11.7 | 0.0 | 0.0 | 9, 6, 7 |
| no repair | 3 | 3 | 2106.4 ± 13.4 | 20.4 | 4096 | 0.22 | 72 | 0.0 | 0.0 | 2.3 | 7, 5, 6 |
| no last-pass clean-up | 3 | 3 | 2100.9 ± 16.4 | 20.8 | 4096 | 0.22 | 72 | 5.7 | 0.0 | 0.0 | 7, 5, 7 |
| no duplicate merging | 3 | 3 | 2101.4 ± 16.1 | 20.8 | 4096 | 0.22 | 72 | 5.7 | 0.0 | 0.0 | 7, 5, 7 |
| tiny split (2^-7), no duplicate merging | 3 | 3 | 2096.9 ± 16.4 | 21.3 | 4096 | 0.22 | 72 | 5.7 | 0.0 | 0.3 | 7, 5, 8 |
| start from documents (half document, half corpus) | 3 | 3 | 2661.8 ± 29.6 | 15.9 | 4096 | 0.22 | 72 | 0.3 | 0.0 | 0.0 | 2, 2, 2 |
| start from documents (10% document, 90% corpus) | 3 | 3 | 2187.2 ± 6.8 | 19.0 | 4096 | 0.22 | 72 | 1.0 | 0.0 | 0.0 | 4, 4, 4 |

classic

| run | passes | seeds | perplexity | train s | batch size | peak step | updates | repaired | merged | dead at end | batch passes to match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Hoffman 2010 best (B=4096, kappa=0.5, tau0=64) | 1 | 3 | 2146.2 ± 16.1 | 8.3 | 4096 | – | 24 | 0.0 | 0.0 | 9.7 | 5, 4, 5 |
| Hoffman 2010 best (B=4096, kappa=0.5, tau0=64) + repair | 1 | 3 | 2152.0 ± 14.6 | 8.6 | 4096 | – | 24 | 10.3 | 0.0 | 0.0 | 5, 4, 5 |
| Hoffman 2010 best (B=4096, kappa=0.5, tau0=64) + repair + clean-up | 1 | 3 | 2192.8 ± 14.7 | 8.8 | 4096 | – | 24 | 10.3 | 0.0 | 0.0 | 4, 4, 4 |
| Hoffman 2010 best (B=4096, kappa=0.5, tau0=64) | 3 | 3 | 2097.7 ± 15.3 | 21.1 | 4096 | – | 72 | 0.0 | 0.0 | 6.7 | 8, 5, 6 |
| Hoffman 2010 best (B=4096, kappa=0.5, tau0=64) + repair | 3 | 3 | 2087.3 ± 15.3 | 21.5 | 4096 | – | 72 | 10.7 | 0.0 | 0.0 | >40, 6, 7 |
| Hoffman 2010 best (B=4096, kappa=0.5, tau0=64) + repair + clean-up | 3 | 3 | 2088.3 ± 15.4 | 21.7 | 4096 | – | 72 | 10.7 | 0.0 | 0.0 | 10, 6, 7 |
| Hoffman 2013 SVI (B=512, kappa=0.9, tau0=1) | 1 | 1 | 2572.6 ± 0.0 | 9.9 | 512 | – | 195 | 0.0 | 0.0 | 0.0 | 2 |
| Hoffman 2013 SVI (B=512, kappa=0.9, tau0=1) + repair | 1 | 1 | 2572.6 ± 0.0 | 10.0 | 512 | – | 195 | 0.0 | 0.0 | 0.0 | 2 |
| Hoffman 2013 SVI (B=512, kappa=0.9, tau0=1) | 3 | 1 | 2537.6 ± 0.0 | 27.6 | 512 | – | 585 | 0.0 | 0.0 | 0.0 | 2 |
| Hoffman 2013 SVI (B=512, kappa=0.9, tau0=1) + repair | 3 | 1 | 2537.6 ± 0.0 | 27.6 | 512 | – | 585 | 0.0 | 0.0 | 0.0 | 2 |
| gensim defaults (B=2048, kappa=0.5, tau0=1) | 1 | 1 | 2258.8 ± 0.0 | 8.8 | 2048 | – | 48 | 0.0 | 0.0 | 0.0 | 4 |
| gensim defaults (B=2048, kappa=0.5, tau0=1) + repair | 1 | 1 | 2258.8 ± 0.0 | 8.8 | 2048 | – | 48 | 0.0 | 0.0 | 0.0 | 4 |
| gensim defaults (B=2048, kappa=0.5, tau0=1) | 3 | 1 | 2270.3 ± 0.0 | 23.1 | 2048 | – | 144 | 0.0 | 0.0 | 0.0 | 3 |
| gensim defaults (B=2048, kappa=0.5, tau0=1) + repair | 3 | 1 | 2270.3 ± 0.0 | 23.2 | 2048 | – | 144 | 0.0 | 0.0 | 0.0 | 3 |
| scikit-learn defaults (B=128, kappa=0.7, tau0=10) | 1 | 1 | 2667.0 ± 0.0 | 14.6 | 128 | – | 781 | 0.0 | 0.0 | 0.0 | 2 |
| scikit-learn defaults (B=128, kappa=0.7, tau0=10) + repair | 1 | 1 | 2667.0 ± 0.0 | 15.1 | 128 | – | 781 | 0.0 | 0.0 | 0.0 | 2 |
| scikit-learn defaults (B=128, kappa=0.7, tau0=10) + repair + clean-up | 1 | 1 | 2623.2 ± 0.0 | 15.2 | 128 | – | 781 | 0.0 | 0.0 | 0.0 | 2 |
| scikit-learn defaults (B=128, kappa=0.7, tau0=10) | 3 | 1 | 2664.5 ± 0.0 | 42.4 | 128 | – | 2343 | 0.0 | 0.0 | 0.0 | 2 |
| scikit-learn defaults (B=128, kappa=0.7, tau0=10) + repair | 3 | 1 | 2664.5 ± 0.0 | 43.2 | 128 | – | 2343 | 0.0 | 0.0 | 0.0 | 2 |
| scikit-learn defaults (B=128, kappa=0.7, tau0=10) + repair + clean-up | 3 | 1 | 2663.2 ± 0.0 | 46.7 | 128 | – | 2343 | 0.0 | 0.0 | 0.0 | 2 |

grid

| run | passes | seeds | perplexity | train s | batch size | peak step | updates | repaired | merged | dead at end | batch passes to match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| grid B x0.0625, step 0.125 | 3 | 1 | 2574.8 ± 0.0 | 26.4 | 256 | 0.125 | 1170 | 3.0 | 0.0 | 0.0 | 2 |
| grid B x0.25, step 0.125 | 3 | 1 | 2181.2 ± 0.0 | 39.6 | 1024 | 0.125 | 291 | 0.0 | 0.0 | 0.0 | 4 |
| grid B x1, step 0.125 | 3 | 1 | 2078.0 ± 0.0 | 21.7 | 4096 | 0.125 | 72 | 11.0 | 0.0 | 0.0 | 8 |
| grid B x4, step 0.125 | 3 | 1 | 2142.7 ± 0.0 | 26.1 | 16384 | 0.125 | 18 | 7.0 | 0.0 | 6.0 | 5 |
| grid B x0.0625, step 0.25 | 3 | 1 | 2981.2 ± 0.0 | 20.6 | 256 | 0.25 | 1170 | 0.0 | 0.0 | 0.0 | 2 |
| grid B x0.25, step 0.25 | 3 | 1 | 2331.0 ± 0.0 | 36.3 | 1024 | 0.25 | 291 | 1.0 | 0.0 | 0.0 | 3 |
| grid B x1, step 0.25 | 3 | 1 | 2096.2 ± 0.0 | 20.5 | 4096 | 0.25 | 72 | 4.0 | 0.0 | 0.0 | 6 |
| grid B x4, step 0.25 | 3 | 1 | 2105.6 ± 0.0 | 25.3 | 16384 | 0.25 | 18 | 3.0 | 0.0 | 1.0 | 6 |
| grid B x0.0625, step 0.5 | 3 | 1 | 3654.1 ± 0.0 | 16.7 | 256 | 0.5 | 1170 | 0.0 | 0.0 | 0.0 | 1 |
| grid B x0.25, step 0.5 | 3 | 1 | 2615.5 ± 0.0 | 32.2 | 1024 | 0.5 | 291 | 0.0 | 0.0 | 0.0 | 2 |
| grid B x1, step 0.5 | 3 | 1 | 2172.4 ± 0.0 | 19.2 | 4096 | 0.5 | 72 | 3.0 | 0.0 | 0.0 | 4 |
| grid B x4, step 0.5 | 3 | 1 | 2081.1 ± 0.0 | 24.4 | 16384 | 0.5 | 18 | 1.0 | 0.0 | 0.0 | 8 |

ridge

| run | passes | seeds | perplexity | train s | batch size | peak step | updates | repaired | merged | dead at end | batch passes to match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| grid B=256, step 0.007812 (memory 33k documents) | 3 | 1 | 2158.4 ± 0.0 | 37.7 | 256 | 0.008 | 1170 | 65.0 | 15.0 | 0.0 | 5 |
| grid B=1024, step 0.03125 (memory 33k documents) | 3 | 1 | 2092.4 ± 0.0 | 47.4 | 1024 | 0.031 | 291 | 31.0 | 0.0 | 0.0 | 7 |
| grid B=1024, step 0.01562 (memory 66k documents) | 3 | 1 | 2170.5 ± 0.0 | 52.7 | 1024 | 0.016 | 291 | 50.0 | 4.0 | 0.0 | 4 |
| grid B=1024, step 0.0625 (memory 16k documents) | 3 | 1 | 2118.4 ± 0.0 | 41.6 | 1024 | 0.062 | 291 | 11.0 | 0.0 | 0.0 | 6 |
| grid B=4096, step 0.0625 (memory 66k documents) | 3 | 1 | 2123.3 ± 0.0 | 24.3 | 4096 | 0.062 | 72 | 27.0 | 0.0 | 0.0 | 5 |
| grid B=16384, step 1 (memory 16k documents) | 3 | 1 | 2101.4 ± 0.0 | 22.6 | 16384 | 1.0 | 18 | 1.0 | 0.0 | 0.0 | 6 |

### NYTimes, 287,021 documents, K = 100

Full-batch variational EM, seeds [0] (held-out perplexity, lower is better)

| passes | perplexity | train s | dead topics |
|---|---|---|---|
| 1 | 3458.2 ± 0.0 | 32.5 | 0.0 |
| 2 | 2477.9 ± 0.0 | 65.1 | 0.0 |
| 3 | 2236.1 ± 0.0 | 97.5 | 0.0 |
| 5 | 2096.9 ± 0.0 | 147.1 | 0.0 |

main

| run | passes | seeds | perplexity | train s | batch size | peak step | updates | repaired | merged | dead at end | batch passes to match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recipe | 3 | 1 | 2188.0 ± 0.0 | 51.4 | 8192 | 1.0 | 105 | 1.0 | 0.0 | 0.0 | 4 |

tuned

| run | passes | seeds | perplexity | train s | batch size | peak step | updates | repaired | merged | dead at end | batch passes to match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recipe, step/4 | 3 | 1 | 2059.8 ± 0.0 | 55.9 | 8192 | 0.25 | 105 | 1.0 | 0.0 | 0.0 | 7 |

hoffman

| run | passes | seeds | perplexity | train s | batch size | peak step | updates | repaired | merged | dead at end | batch passes to match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Hoffman 2010 best (B=4096, kappa=0.5, tau0=64) | 3 | 1 | 2065.6 ± 0.0 | 56.1 | 4096 | – | 210 | 0.0 | 0.0 | 1.0 | 7 |
