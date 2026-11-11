# NMF with the e-means ingredients: TF-IDF of UCI NYTimes

CPU (Apple M2 Max, 8 threads), float32 for this code, scikit-learn 1.9.1 as shipped. Raw log: `nmf_nyt.log`. Scripts: `experiments/nmf_*.py`.
Log rows with group `classic` come from a first run whose settings were not applied (they repeat the recipe rows); the tables use the corrected `textbook` rows.

### NYTimes TF-IDF, 100,000 documents, K = 100 (held-out relative squared error, lower is better)

Full-batch HALS, seeds [0, 1, 2]

| passes | error | train s |
|---|---|---|
| 1 | 0.90205 ± 0.00075 | 2.3 |
| 2 | 0.84670 ± 0.00200 | 3.3 |
| 3 | 0.82603 ± 0.00156 | 4.3 |
| 5 | 0.81263 ± 0.00069 | 6.4 |
| 10 | 0.80653 ± 0.00027 | 11.5 |
| 15 | 0.80531 ± 0.00017 | 16.6 |
| 20 | 0.80498 ± 0.00018 | 21.7 |
| 30 | 0.80443 ± 0.00009 | 32.0 |
| 50 | 0.80404 ± 0.00005 | 53.5 |
| 75 | 0.80384 ± 0.00022 | 80.6 |
| 100 | 0.80369 ± 0.00040 | 107.1 |

main

| run | passes | seeds | error | train s | batch size | peak step | updates | repaired | dead at end | batch passes to match | batch seconds to match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recipe | 1 | 3 | 0.80698 ± 0.00039 | 3.5 | 2048 | 0.532 | 48 | 0.0 | 0.0 | 9, 11, 9 | 10.7, 12.0, 10.7 |
| recipe | 2 | 3 | 0.80439 ± 0.00045 | 6.9 | 2048 | 0.493 | 96 | 0.0 | 0.0 | 27, >100, 23 | 29.5, –, 25.6 |
| recipe | 3 | 3 | 0.80398 ± 0.00037 | 7.3 | 4096 | 0.88 | 72 | 0.0 | 0.0 | 34, 36, 79 | 37.9, 36.4, 84.0 |
| recipe | 5 | 3 | 0.80425 ± 0.00045 | 9.2 | 8192 | 1.0 | 60 | 0.0 | 0.0 | 20, 32, 71 | 22.1, 32.5, 75.6 |
| recipe | 10 | 3 | 0.80365 ± 0.00049 | 18.3 | 8192 | 1.0 | 120 | 0.0 | 0.0 | 30, >100, 93 | 33.0, –, 98.8 |

ablate

| run | passes | seeds | error | train s | batch size | peak step | updates | repaired | dead at end | batch passes to match | batch seconds to match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| constant step (peak value) | 3 | 3 | 0.80386 ± 0.00038 | 7.3 | 4096 | 0.88 | 72 | 0.0 | 0.0 | >100, 38, 87 | –, 38.3, 92.5 |
| no repair | 3 | 3 | 0.80398 ± 0.00037 | 7.3 | 4096 | 0.88 | 72 | 0.0 | 0.0 | 34, 36, 79 | 37.9, 36.4, 84.0 |
| no last-pass clean-up | 3 | 3 | 0.80411 ± 0.00036 | 7.4 | 4096 | 0.88 | 72 | 0.0 | 0.0 | 28, 32, 73 | 30.6, 32.5, 77.7 |
| B/4 and step/4 | 3 | 3 | 0.80348 ± 0.00045 | 16.4 | 1024 | 0.22 | 291 | 0.0 | 0.0 | 51, >100, 91 | 58.1, –, 96.7 |
| B/4, same step | 3 | 3 | 0.80311 ± 0.00032 | 16.5 | 1024 | 0.88 | 291 | 0.0 | 0.0 | >100, >100, 86 | –, –, 91.4 |
| step/4, same B | 3 | 3 | 0.80538 ± 0.00033 | 7.3 | 4096 | 0.22 | 72 | 0.0 | 0.0 | 13, 13, 23 | 14.8, 13.9, 25.6 |
| B x4, step 1 | 3 | 3 | 0.80640 ± 0.00011 | 4.8 | 16384 | 1.0 | 18 | 0.0 | 0.0 | 10, 11, 12 | 11.7, 12.0, 13.9 |

textbook

| run | passes | seeds | error | train s | batch size | peak step | updates | repaired | dead at end | batch passes to match | batch seconds to match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| running average (B=512) | 1 | 3 | 0.80982 ± 0.00119 | 10.3 | 512 | – | 195 | 0.0 | 0.0 | 7, 6, 8 | 8.6, 7.0, 9.7 |
| running average (B=512) + repair + clean-up | 1 | 3 | 0.80982 ± 0.00119 | 10.4 | 512 | – | 195 | 0.0 | 0.0 | 7, 6, 8 | 8.6, 7.0, 9.7 |
| running average (B=512) | 3 | 3 | 0.80827 ± 0.00079 | 31.1 | 512 | – | 585 | 0.0 | 0.0 | 8, 8, 9 | 9.6, 9.0, 10.7 |
| running average (B=512) + repair + clean-up | 3 | 3 | 0.80758 ± 0.00059 | 30.6 | 512 | – | 585 | 0.0 | 0.0 | 8, 9, 10 | 9.6, 10.0, 11.8 |
| power step (1+t)^-0.5 (B=512) | 1 | 3 | 0.80738 ± 0.00060 | 10.2 | 512 | – | 195 | 0.0 | 0.0 | 9, 11, 8 | 10.7, 12.0, 9.7 |
| power step (1+t)^-0.5 (B=512) + repair + clean-up | 1 | 3 | 0.80568 ± 0.00017 | 10.2 | 512 | – | 195 | 0.0 | 0.0 | 12, 15, 13 | 13.8, 15.8, 14.9 |
| power step (1+t)^-0.5 (B=512) | 3 | 3 | 0.80563 ± 0.00066 | 30.5 | 512 | – | 585 | 0.0 | 0.0 | 13, 23, 11 | 14.8, 23.6, 12.8 |
| power step (1+t)^-0.5 (B=512) + repair + clean-up | 3 | 3 | 0.80425 ± 0.00069 | 30.8 | 512 | – | 585 | 0.0 | 0.0 | >100, >100, 20 | –, –, 22.3 |
| power step (64+t)^-0.5 (B=4096) | 1 | 3 | 0.81228 ± 0.00080 | 2.4 | 4096 | – | 24 | 0.0 | 0.0 | 6, 5, 6 | 7.5, 6.0, 7.6 |
| power step (64+t)^-0.5 (B=4096) + repair + clean-up | 1 | 3 | 0.81498 ± 0.00099 | 2.5 | 4096 | – | 24 | 0.0 | 0.0 | 5, 5, 5 | 6.5, 6.0, 6.5 |
| power step (64+t)^-0.5 (B=4096) | 3 | 3 | 0.80615 ± 0.00025 | 7.5 | 4096 | – | 72 | 0.0 | 0.0 | 11, 11, 13 | 12.7, 12.0, 14.9 |
| power step (64+t)^-0.5 (B=4096) + repair + clean-up | 3 | 3 | 0.80607 ± 0.00025 | 8.0 | 4096 | – | 72 | 0.0 | 0.0 | 11, 11, 13 | 12.7, 12.0, 14.9 |
| running average (formula batch) | 1 | 3 | 0.81190 ± 0.00024 | 3.6 | 2048 | – | 48 | 0.0 | 0.0 | 6, 6, 5 | 7.5, 7.0, 6.5 |
| running average (formula batch) + repair + clean-up | 1 | 3 | 0.81190 ± 0.00024 | 3.6 | 2048 | – | 48 | 0.0 | 0.0 | 6, 6, 5 | 7.5, 7.0, 6.5 |
| running average (formula batch) | 3 | 3 | 0.80949 ± 0.00040 | 7.4 | 4096 | – | 72 | 0.0 | 0.0 | 7, 7, 7 | 8.6, 8.0, 8.6 |
| running average (formula batch) + repair + clean-up | 3 | 3 | 0.80798 ± 0.00046 | 7.3 | 4096 | – | 72 | 0.0 | 0.0 | 8, 8, 9 | 9.6, 9.0, 10.7 |

sklearn

| run | passes | seeds | error | train s | batch size | peak step | updates | repaired | dead at end | batch passes to match | batch seconds to match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NMF default (nndsvda start, coordinate descent) | 117 | 1 | 0.80314 ± 0.00000 | 193.1 | – | – | – | – | 0.0 | >100 | – |
| MiniBatchNMF default | 11 | 3 | 0.81938 ± 0.00062 | 19.1 | – | – | – | – | 0.0 | 4, 4, 4 | 5.4, 5.0, 5.5 |
| NMF random start | 172 | 3 | 0.80352 ± 0.00066 | 277.4 | – | – | – | – | 0.0 | 29, >100, >100 | 31.7, –, – |
| NMF random start, 25 iterations | 25 | 3 | 0.80486 ± 0.00031 | 39.1 | – | – | – | – | 0.0 | 14, 25, 32 | 15.9, 25.6, 34.7 |
