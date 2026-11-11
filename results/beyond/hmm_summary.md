
### synth, K = 64, 9,765 training rows, seeds [0, 1, 2]

Reference, held-out bits/symbol: unigram 4.2247, order-1 Markov 3.5943, true model 2.8802

| method | passes | train s | held-out bits | Baum-Welch passes to match (per seed) |
|---|---|---|---|---|
| Baum-Welch | 1 | 0.4 | 4.2072 | – |
| Baum-Welch | 2 | 0.6 | 4.1798 | – |
| Baum-Welch | 3 | 0.9 | 4.1249 | – |
| Baum-Welch | 5 | 1.5 | 3.9849 | – |
| Baum-Welch | 10 | 2.9 | 3.7860 | – |
| Baum-Welch | 20 | 5.7 | 3.4102 | – |
| Baum-Welch | 30 | 8.5 | 3.1348 | – |
| Baum-Welch | 40 | 11.4 | 3.0240 | – |
| Baum-Welch | 50 | 14.4 | 3.0006 | – |
| Baum-Welch | 60 | 17.2 | 2.9939 | – |
| Baum-Welch | 75 | 21.6 | 2.9877 | – |
| Baum-Welch | 100 | 28.8 | 2.9763 | – |
| Baum-Welch | 150 | 43.2 | 2.9688 | – |
| online, formula with n = rows (B=256 rows, step 0.287) | 1 | 0.8 | 3.9054 | [8, 8, 8] |
| online, formula with n = rows (B=256 rows, step 0.265) | 2 | 1.6 | 3.6064 | [15, 15, 15] |
| online, formula with n = rows (B=512 rows, step 0.474) | 3 | 1.5 | 3.4815 | [18, 19, 18] |
| online, formula with n = rows (B=512 rows, step 0.448) | 5 | 2.5 | 3.1861 | [29, 28, 28] |
| online, formula with n = rows (B=1024 rows, step 0.774) | 10 | 3.4 | 3.0082 | [39, 59, 44] |
| online, formula with n = symbols (B=16 rows, step 0.287) | 1 | 9.0 | 3.0697 | [32, 37, 34] |
| online, formula with n = symbols (B=32 rows, step 0.474) | 3 | 13.6 | 2.9790 | ['>150', 50, '>150'] |
| online, formula with n = symbols (B=64 rows, step 0.774) | 10 | 22.5 | 2.9496 | ['>150', '>150', 99] |

| 3-pass variant | held-out bits | vs default (negative = better) | train s | revived | seeds |
|---|---|---|---|---|---|
| default | 3.4815 | +0.0000 ± 0.0000 | 1.7 | 0 | 3 |
| constant step | 3.2291 | -0.2524 ± 0.0201 | 1.6 | 2 | 3 |
| no revival | 3.4815 | +0.0000 ± 0.0000 | 1.5 | 0 | 3 |
| no last-pass clean-up | 3.4937 | +0.0122 ± 0.0015 | 1.5 | 0 | 3 |
| tiny split (1 +- 2^-7) | 3.4815 | +0.0000 ± 0.0000 | 1.5 | 0 | 3 |
| B/4, step/4 | 3.3902 | -0.0913 ± 0.0132 | 3.4 | 0 | 3 |
| n = symbols | 2.9790 | -0.5025 ± 0.0371 | 14.0 | 4 | 3 |
| n = symbols, constant step | 2.9636 | -0.5179 ± 0.0216 | 14.1 | 6 | 3 |
| n = symbols, no revival | 2.9735 | -0.5080 ± 0.0203 | 13.2 | 0 | 3 |
| n = symbols, no last-pass clean-up | 2.9799 | -0.5016 ± 0.0370 | 13.4 | 4 | 3 |

### text8, K = 64, 78,125 training rows, seeds [0, 1, 2]

Reference, held-out bits/symbol: unigram 4.1251, order-1 Markov 3.4381

| method | passes | train s | held-out bits | Baum-Welch passes to match (per seed) |
|---|---|---|---|---|
| Baum-Welch | 1 | 2.4 | 4.1165 | – |
| Baum-Welch | 2 | 4.7 | 4.1063 | – |
| Baum-Welch | 3 | 7.0 | 4.0914 | – |
| Baum-Welch | 5 | 11.5 | 4.0199 | – |
| Baum-Welch | 10 | 22.9 | 3.5567 | – |
| Baum-Welch | 20 | 45.7 | 3.1955 | – |
| Baum-Welch | 30 | 68.5 | 3.1028 | – |
| Baum-Welch | 40 | 91.4 | 3.0661 | – |
| Baum-Welch | 50 | 114.3 | 3.0474 | – |
| Baum-Welch | 60 | 137.1 | 3.0365 | – |
| Baum-Welch | 75 | 171.3 | 3.0254 | – |
| Baum-Welch | 100 | 228.3 | 3.0153 | – |
| online, formula with n = rows (B=1024 rows, step 0.391) | 1 | 2.9 | 3.4347 | [12, 13, 13] |
| online, formula with n = rows (B=2048 rows, step 0.677) | 2 | 4.8 | 3.1136 | [29, 28, 28] |
| online, formula with n = rows (B=4096 rows, step 1.0) | 3 | 6.7 | 3.0956 | [32, 33, 31] |
| online, formula with n = rows (B=4096 rows, step 1.0) | 5 | 11.1 | 3.0451 | [54, 56, 47] |
| online, formula with n = rows (B=8192 rows, step 1.0) | 10 | 21.2 | 3.0481 | [52, 53, 46] |
| online, formula with n = symbols (B=64 rows, step 0.391) | 1 | 18.8 | 3.0268 | [80, 80, 62] |
| online, formula with n = symbols (B=128 rows, step 0.677) | 2 | 20.2 | 2.9853 | ['>200', '>100', '>100'] |
| online, formula with n = symbols (B=256 rows, step 1.0) | 3 | 19.9 | 2.9824 | ['>200', '>100', '>100'] |
| online, formula with n = symbols (B=256 rows, step 1.0) | 5 | 31.0 | 2.9804 | ['>200', '>100', '>100'] |
| online, formula with n = symbols (B=512 rows, step 1.0) | 10 | 40.4 | 2.9817 | ['>200', '>100', '>100'] |

| 3-pass variant | held-out bits | vs default (negative = better) | train s | revived | seeds |
|---|---|---|---|---|---|
| default | 3.0956 | +0.0000 ± 0.0000 | 6.9 | 1 | 3 |
| constant step | 3.0536 | -0.0420 ± 0.0054 | 6.8 | 1 | 3 |
| no revival | 3.0926 | -0.0030 ± 0.0040 | 6.7 | 0 | 3 |
| no last-pass clean-up | 3.0935 | -0.0021 ± 0.0004 | 6.8 | 1 | 3 |
| tiny split (1 +- 2^-7) | 3.0965 | +0.0009 ± 0.0008 | 6.8 | 1 | 3 |
| B/4, step/4 | 3.0869 | -0.0088 ± 0.0073 | 8.9 | 3 | 3 |
| n = symbols | 2.9824 | -0.1132 ± 0.0070 | 20.5 | 1 | 3 |
| n = symbols, constant step | 2.9814 | -0.1142 ± 0.0059 | 21.3 | 1 | 3 |
| n = symbols, no revival | 2.9891 | -0.1065 ± 0.0029 | 20.7 | 0 | 3 |
| n = symbols, no last-pass clean-up | 2.9826 | -0.1130 ± 0.0068 | 20.2 | 1 | 3 |

| textbook step (t+2)^-kappa, 3 passes | batch rows | revival | held-out bits | vs recipe, n = rows (negative = better) | vs recipe, n = symbols | train s | revived | seeds |
|---|---|---|---|---|---|---|---|---|
| kappa 0.5 | 4096 | no | 3.3171 | +0.2206 ± 0.0000 | +0.3407 ± 0.0000 | 6.7 | 0 | 1 |
| kappa 0.5 | 4096 | yes | 3.3171 | +0.2206 ± 0.0000 | +0.3407 ± 0.0000 | 6.8 | 0 | 1 |
| kappa 0.6 | 4096 | no | 3.4411 | +0.3446 ± 0.0000 | +0.4647 ± 0.0000 | 6.9 | 0 | 1 |
| kappa 0.6 | 4096 | yes | 3.4411 | +0.3446 ± 0.0000 | +0.4647 ± 0.0000 | 6.6 | 0 | 1 |
| kappa 0.9 | 4096 | no | 3.8558 | +0.7592 ± 0.0000 | +0.8793 ± 0.0000 | 6.6 | 0 | 1 |
| kappa 0.9 | 4096 | yes | 3.8558 | +0.7592 ± 0.0000 | +0.8793 ± 0.0000 | 6.6 | 0 | 1 |
| kappa 0.5 | 1024 | no | 3.0987 | +0.0021 ± 0.0000 | +0.1222 ± 0.0000 | 8.5 | 0 | 1 |
| kappa 0.5 | 1024 | yes | 3.0943 | -0.0023 ± 0.0000 | +0.1178 ± 0.0000 | 10.2 | 3 | 1 |
| kappa 0.6 | 1024 | no | 3.1714 | +0.0748 ± 0.0000 | +0.1949 ± 0.0000 | 8.4 | 0 | 1 |
| kappa 0.6 | 1024 | yes | 3.1703 | +0.0738 ± 0.0000 | +0.1939 ± 0.0000 | 8.4 | 3 | 1 |
| kappa 0.9 | 1024 | no | 3.6146 | +0.5180 ± 0.0000 | +0.6382 ± 0.0000 | 8.4 | 0 | 1 |
| kappa 0.9 | 1024 | yes | 3.6146 | +0.5180 ± 0.0000 | +0.6382 ± 0.0000 | 9.7 | 0 | 1 |
| kappa 0.5 | 256 | no | 3.0431 | -0.0534 ± 0.0000 | +0.0667 ± 0.0000 | 18.9 | 0 | 1 |
| kappa 0.5 | 256 | yes | 3.0380 | -0.0586 ± 0.0000 | +0.0616 ± 0.0000 | 20.7 | 3 | 1 |
| kappa 0.6 | 256 | no | 3.0676 | -0.0290 ± 0.0000 | +0.0912 ± 0.0000 | 18.7 | 0 | 1 |
| kappa 0.6 | 256 | yes | 3.0661 | -0.0305 ± 0.0000 | +0.0897 ± 0.0000 | 21.0 | 3 | 1 |
| kappa 0.9 | 256 | no | 3.4332 | +0.3366 ± 0.0000 | +0.4567 ± 0.0000 | 18.5 | 0 | 1 |
| kappa 0.9 | 256 | yes | 3.4332 | +0.3366 ± 0.0000 | +0.4567 ± 0.0000 | 21.0 | 0 | 1 |

Settings grid, 3 passes (formula: B = 4096, step = 1.0); change in held-out bits vs the formula's run, negative = better

| batch rows | peak step | held-out bits | vs formula | train s | seeds |
|---|---|---|---|---|---|
| 4096 | 1.0 | 3.0966 | +0.0000 ± 0.0000 | 6.7 | 1 |
| 4096 | 0.5 | 3.2228 | +0.1262 ± 0.0000 | 6.6 | 1 |
| 4096 | 0.25 | 3.4422 | +0.3456 ± 0.0000 | 6.6 | 1 |
| 1024 | 1.0 | 3.0078 | -0.0887 ± 0.0000 | 8.4 | 1 |
| 1024 | 0.5 | 3.0400 | -0.0566 ± 0.0000 | 8.4 | 1 |
| 1024 | 0.25 | 3.0777 | -0.0189 ± 0.0000 | 8.4 | 1 |
| 1024 | 0.125 | 3.1832 | +0.0866 ± 0.0000 | 8.4 | 1 |
| 256 | 1.0 | 2.9764 | -0.1201 ± 0.0000 | 18.5 | 1 |
| 256 | 0.5 | 2.9902 | -0.1064 ± 0.0000 | 18.4 | 1 |
| 256 | 0.25 | 2.9948 | -0.1018 ± 0.0000 | 18.4 | 1 |
| 256 | 0.125 | 3.0336 | -0.0630 ± 0.0000 | 18.4 | 1 |
| 256 | 0.06 | 3.0701 | -0.0265 ± 0.0000 | 18.4 | 1 |
| 64 | 0.25 | 2.9880 | -0.1086 ± 0.0000 | 54.1 | 1 |

### text8, K = 256, 78,125 training rows, seeds [0, 1]

Reference, held-out bits/symbol: unigram 4.1251, order-1 Markov 3.4381

| method | passes | train s | held-out bits | Baum-Welch passes to match (per seed) |
|---|---|---|---|---|
| Baum-Welch | 1 | 7.6 | 4.1183 | – |
| Baum-Welch | 2 | 15.1 | 4.1108 | – |
| Baum-Welch | 3 | 22.7 | 4.1004 | – |
| Baum-Welch | 5 | 37.7 | 4.0476 | – |
| Baum-Welch | 10 | 75.3 | 3.5343 | – |
| Baum-Welch | 20 | 150.4 | 2.9271 | – |
| Baum-Welch | 30 | 225.7 | 2.7138 | – |
| Baum-Welch | 40 | 301.0 | 2.6431 | – |
| Baum-Welch | 50 | 376.3 | 2.6077 | – |
| Baum-Welch | 60 | 451.6 | 2.5888 | – |
| online, formula with n = rows (B=2048 rows, step 0.386) | 1 | 7.3 | 3.6529 | [9, 9] |
| online, formula with n = rows (B=2048 rows, step 0.358) | 2 | 14.5 | 3.0355 | [18, 18] |
| online, formula with n = rows (B=4096 rows, step 0.638) | 3 | 21.7 | 2.8635 | [22, 22] |
| online, formula with n = rows (B=4096 rows, step 0.603) | 5 | 36.6 | 2.6872 | [34, 33] |
| online, formula with n = rows (B=8192 rows, step 1.0) | 10 | 69.6 | 2.6095 | [53, 48] |
| online, formula with n = symbols (B=128 rows, step 0.386) | 1 | 11.6 | 2.6458 | [43, 38] |
| online, formula with n = symbols (B=256 rows, step 0.638) | 3 | 29.3 | 2.5315 | ['>60', '>60'] |

| 3-pass variant | held-out bits | vs default (negative = better) | train s | revived | seeds |
|---|---|---|---|---|---|
| default | 2.8722 | +0.0000 ± 0.0000 | 21.7 | 46 | 1 |
| constant step | 2.6967 | -0.1755 ± 0.0000 | 21.9 | 48 | 1 |
| no revival | 2.8682 | -0.0040 ± 0.0000 | 21.6 | 0 | 1 |
| no last-pass clean-up | 2.8737 | +0.0015 ± 0.0000 | 21.9 | 46 | 1 |
| n = symbols | 2.5305 | -0.3417 ± 0.0000 | 30.5 | 48 | 1 |
| n = symbols, constant step | 2.5340 | -0.3381 ± 0.0000 | 29.2 | 51 | 1 |
| n = symbols, no revival | 2.5399 | -0.3323 ± 0.0000 | 30.2 | 0 | 1 |


Notes

- text8, K = 64, seed 0 only: Baum-Welch continued to 200 passes reached 3.0022 bits at 150 and 2.9931 at 200 (450 s).
- Planned but not run (time): rows of 64 symbols with context, state splitting on the GPU, K = 256 third seed, textbook comparator on more than one seed.
- hmm_cpu_preview_k32.txt: CPU-only preview (K = 32, 3M symbols, 2 seeds) of batch size x step.
