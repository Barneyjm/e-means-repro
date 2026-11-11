# ALS with the e-means recipe: result tables

Explicit-feedback ALS (ALS-WR ridge), held-out RMSE on a random 10 % of ratings, split and start fixed by the seed.
`experiments/als_torch.py`, `als_run.py`. 'formula' = batch size and step from the e-means paper's laws with n = users, k = items;
'fixed' = one batch size and peak step for every pass count. Seconds inside one table come from different queue jobs on a shared
machine; the side-by-side timing is in its own section.

## Reading of the tables

- **At the paper's formula settings (users as n, items as k) the recipe gives no time gain.** MovieLens-20M, rank 64: 1 / 3 / 10
  online passes equal 5 / 8 / 11-12 ALS sweeps in held-out RMSE and cost about as many sweeps of time. The formula's peak step
  (0.07-0.33) is far too small for this problem.
- **With a peak step of 1.0 and a batch of about 3 % of the users it is a clear gain in held-out error per second.**
  MovieLens-20M, rank 64, B = 4096: one pass (4.7 sweeps of time) reaches 0.7735, which full ALS reaches after 20-27 sweeps;
  from two passes on (0.7724) full ALS does not match it within 40 sweeps (0.7729). Rank 32: one pass equals 15-16 sweeps,
  three passes equal ALS at 40 sweeps.
- **The gain is regularization, not better optimization.** The online solution has a higher training objective than ALS at 40
  sweeps (0.6061 against 0.6034) and a lower held-out error. It depends on the ridge strength: at the ridge that is best for
  ALS (0.04) online is ahead by about 0.001; at 0.05 the two end level; at 0.03 online is ahead by 0.002. On MovieLens-1M the
  online version's best ridge is lower than ALS's (0.06 against 0.07); each at its own best ridge, online is ahead by 0.0013 to
  0.0018 (3 seeds, paired, spread 0.0003).
- **Ingredients (MovieLens-20M, 3 passes).** Last-pass clean-up: matters (+0.0011 without). Cosine curve: no effect against a
  constant step of 1.0 once the clean-up is there (it mattered a lot on MovieLens-1M with B = 128: +0.0205). Textbook step
  (t+2)^-kappa, no clean-up: +0.0019 to +0.0068. Fading only the touched items: no effect. Re-solving items every 5 batches
  instead of every batch: +0.0013 and no faster.
- **No failure mode that needs a timed repair was found** (no dead or exploding factors; errors by item popularity match ALS).
  Two guards were needed: items with no training rating are set to zero (they kept their random start), and with a step of 1
  the totals of items absent from a batch are wiped, so only items rated in the current batch may be re-solved.
- **Cost of a pass.** In this implementation one online pass costs 2.9 (B = 16384) to 4.7 (B = 4096) ALS sweeps of time: each
  batch re-solves about 13,000 items and the per-batch kernels are less efficient than one big sweep. The state per item is a
  K x K matrix, K times larger than the factor it produces; in k-means the state is the size of the centroid.
- Not built or not run: biases, implicit feedback, iALS++ block updates (Rendle 2021); the plain SGD reference
  (`experiments/als_sgd.py`, checked on MovieLens-1M only) was dropped from the MovieLens-20M queue for time, as were the
  one-pass ablations and a fixed-settings table at the grid's best but slower setting (B = 1024). Reference point from the
  literature: ALS-WR at rank 200 on a 90/10 split of MovieLens-20M, 0.7746.
- Seconds in the per-dataset tables were taken in different jobs on a shared machine and vary by up to 1.7x; use the
  side-by-side timing table for speed.

## Side-by-side timing (MovieLens-20M, rank 64, one process, 3 repeats)

| run | seconds | in ALS sweeps of time | held-out RMSE |
|---|---|---|---|
| full ALS, per sweep | 1.12 | 1 | 0.7753 after 10 sweeps |
| online, 1 pass(es), B=16384, a0=1.0 | 3.20 | 2.9 | 0.7834 |
| online, 1 pass(es), B=8192, a0=1.0 | 3.80 | 3.4 | 0.7768 |
| online, 1 pass(es), B=4096, a0=1.0 | 5.27 | 4.7 | 0.7734 |
| online, 3 pass(es), B=4096, a0=1.0 | 14.01 | 12.5 | 0.7720 |

## MovieLens-20M

### ml20m, rank 32, ridge 0.04, seeds [0, 1, 2]

| method | passes | settings | seconds | held-out RMSE | vs ALS 40 | ALS sweeps to match (per seed) | training objective |
|---|---|---|---|---|---|---|---|
| ALS | 1 | | 0.61 | 1.0041 ± 0.0065 | +0.2292 | | 0.9510 |
| ALS | 2 | | 1.09 | 0.8241 ± 0.0007 | +0.0492 | | 0.6741 |
| ALS | 3 | | 1.58 | 0.7991 ± 0.0006 | +0.0242 | | 0.6458 |
| ALS | 5 | | 2.54 | 0.7856 ± 0.0008 | +0.0107 | | 0.6319 |
| ALS | 10 | | 4.98 | 0.7784 ± 0.0004 | +0.0035 | | 0.6248 |
| ALS | 15 | | 7.40 | 0.7766 ± 0.0004 | +0.0017 | | 0.6232 |
| ALS | 20 | | 9.80 | 0.7759 ± 0.0004 | +0.0010 | | 0.6225 |
| ALS | 30 | | 14.58 | 0.7752 ± 0.0004 | +0.0003 | | 0.6219 |
| ALS | 40 | | 19.36 | 0.7749 ± 0.0004 | +0.0000 | | 0.6217 |
| ALS, best sweep of 40 | [40, 40, 40] | | | 0.7749 ± 0.0004 | +0.0000 | | |
| online (formula) | 1 | B=4096, a0=0.0656 | 2.30 | 0.7865 ± 0.0006 | +0.0116 | [5, 5, 5] | 0.6348 |
| online (formula) | 2 | B=8192, a0=0.1134 | 2.95 | 0.7825 ± 0.0006 | +0.0076 | [7, 7, 7] | 0.6294 |
| online (formula) | 3 | B=8192, a0=0.1085 | 4.19 | 0.7806 ± 0.0006 | +0.0058 | [8, 8, 8] | 0.6274 |
| online (formula) | 5 | B=16384, a0=0.1914 | 5.14 | 0.7797 ± 0.0006 | +0.0048 | [9, 9, 9] | 0.6263 |
| online (formula) | 10 | B=32768, a0=0.3309 | 8.07 | 0.7779 ± 0.0005 | +0.0030 | [12, 11, 11] | 0.6245 |
| online (fixed) | 1 | B=4096, a0=1.0 | 3.11 | 0.7767 ± 0.0005 | +0.0018 | [16, 15, 15] | 0.6297 |
| online (fixed) | 2 | B=4096, a0=1.0 | 5.78 | 0.7750 ± 0.0005 | +0.0001 | [33, None, 39] | 0.6242 |
| online (fixed) | 3 | B=4096, a0=1.0 | 8.16 | 0.7748 ± 0.0004 | -0.0001 | [None, None, None] | 0.6232 |
| online (fixed) | 5 | B=4096, a0=1.0 | 12.56 | 0.7746 ± 0.0004 | -0.0003 | [None, None, None] | 0.6224 |
| online (fixed) | 10 | B=4096, a0=1.0 | 21.78 | 0.7744 ± 0.0004 | -0.0005 | [None, None, None] | 0.6220 |

ml20m, rank 32: grid at 3 pass(es), seed 0, held-out RMSE (seconds)

| batch size | step 0.03 | step 0.1 | step 0.25 | step 0.5 | step 1.0 |
|---|---|---|---|---|---|
| 1024 | 0.7769 (17.8) | 0.7751 (17.5) | 0.7745 (25.9) | 0.7743 (39.6) | 0.7744 (44.2) |
| 2048 | 0.7786 (20.8) | 0.7761 (20.9) | 0.7750 (20.6) | 0.7746 (21.5) | 0.7745 (22.2) |
| 4096 | 0.7805 (14.3) | 0.7780 (10.6) | 0.7759 (15.0) | 0.7750 (9.3) | 0.7748 (15.0) |
| 8192 | 0.7825 (5.3) | 0.7806 (8.4) | 0.7780 (8.4) | 0.7762 (5.6) | 0.7750 (8.7) |
| 16384 | 0.7847 (5.4) | 0.7835 (5.5) | 0.7811 (5.5) | 0.7784 (5.5) | 0.7762 (5.7) |

### ml20m, rank 64, ridge 0.04, seeds [0, 1, 2]

| method | passes | settings | seconds | held-out RMSE | vs ALS 40 | ALS sweeps to match (per seed) | training objective |
|---|---|---|---|---|---|---|---|
| ALS | 1 | | 1.50 | 0.9780 ± 0.0034 | +0.2051 | | 0.8839 |
| ALS | 2 | | 2.63 | 0.8131 ± 0.0014 | +0.0402 | | 0.6466 |
| ALS | 3 | | 3.74 | 0.7923 ± 0.0007 | +0.0194 | | 0.6245 |
| ALS | 5 | | 5.98 | 0.7811 ± 0.0004 | +0.0082 | | 0.6130 |
| ALS | 10 | | 11.56 | 0.7755 ± 0.0005 | +0.0026 | | 0.6067 |
| ALS | 15 | | 17.15 | 0.7742 ± 0.0005 | +0.0013 | | 0.6050 |
| ALS | 20 | | 22.73 | 0.7737 ± 0.0005 | +0.0007 | | 0.6043 |
| ALS | 30 | | 33.89 | 0.7732 ± 0.0005 | +0.0002 | | 0.6037 |
| ALS | 40 | | 45.04 | 0.7729 ± 0.0005 | +0.0000 | | 0.6034 |
| ALS, best sweep of 40 | [40, 40, 40] | | | 0.7729 ± 0.0005 | +0.0000 | | |
| online (formula) | 1 | B=4096, a0=0.0656 | 5.08 | 0.7817 ± 0.0004 | +0.0088 | [5, 5, 5] | 0.6165 |
| online (formula) | 2 | B=8192, a0=0.1134 | 6.38 | 0.7784 ± 0.0005 | +0.0054 | [7, 7, 7] | 0.6112 |
| online (formula) | 3 | B=8192, a0=0.1085 | 9.13 | 0.7770 ± 0.0005 | +0.0041 | [8, 8, 8] | 0.6092 |
| online (formula) | 5 | B=16384, a0=0.1914 | 10.94 | 0.7765 ± 0.0004 | +0.0035 | [9, 9, 9] | 0.6082 |
| online (formula) | 10 | B=32768, a0=0.3309 | 17.36 | 0.7752 ± 0.0004 | +0.0023 | [11, 11, 12] | 0.6065 |
| online (fixed) | 1 | B=4096, a0=1.0 | 8.48 | 0.7735 ± 0.0004 | +0.0005 | [20, 24, 27] | 0.6136 |
| online (fixed) | 2 | B=4096, a0=1.0 | 15.62 | 0.7724 ± 0.0005 | -0.0005 | [None, None, None] | 0.6076 |
| online (fixed) | 3 | B=4096, a0=1.0 | 22.75 | 0.7723 ± 0.0004 | -0.0007 | [None, None, None] | 0.6061 |
| online (fixed) | 5 | B=4096, a0=1.0 | 35.73 | 0.7720 ± 0.0004 | -0.0010 | [None, None, None] | 0.6047 |
| online (fixed) | 10 | B=4096, a0=1.0 | 70.68 | 0.7718 ± 0.0004 | -0.0011 | [None, None, None] | 0.6038 |

ml20m, rank 64: grid at 1 pass(es), seed 0, held-out RMSE (seconds)

| batch size | step 0.03 | step 0.1 | step 0.25 | step 0.5 | step 1.0 |
|---|---|---|---|---|---|
| 512 | 0.7764 (19.1) | 0.7730 (18.3) | 0.7722 (18.4) | 0.7744 (18.4) | 0.7811 (18.7) |
| 1024 | 0.7787 (11.3) | 0.7755 (11.9) | 0.7729 (12.1) | 0.7725 (11.9) | 0.7755 (12.2) |
| 2048 | 0.7805 (7.4) | 0.7781 (7.2) | 0.7750 (7.5) | 0.7731 (7.4) | 0.7730 (7.5) |
| 4096 | 0.7826 (5.0) | 0.7812 (5.1) | 0.7786 (5.4) | 0.7757 (5.1) | 0.7734 (5.0) |
| 8192 | 0.7857 (3.8) | 0.7849 (4.0) | 0.7831 (4.0) | 0.7804 (4.0) | 0.7768 (4.1) |
| 16384 | 0.7905 (3.2) | 0.7900 (3.2) | 0.7890 (3.1) | 0.7872 (3.1) | 0.7834 (3.1) |

ml20m, rank 64: grid at 3 pass(es), seed 0, held-out RMSE (seconds)

| batch size | step 0.03 | step 0.1 | step 0.25 | step 0.5 | step 1.0 |
|---|---|---|---|---|---|
| 1024 | 0.7743 (29.6) | 0.7724 (29.8) | 0.7721 (30.4) | 0.7720 (30.7) | 0.7715 (31.2) |
| 2048 | 0.7756 (18.5) | 0.7733 (18.4) | 0.7723 (18.5) | 0.7720 (18.5) | 0.7719 (18.4) |
| 4096 | 0.7772 (12.4) | 0.7750 (12.3) | 0.7731 (12.3) | 0.7725 (12.4) | 0.7720 (12.1) |
| 8192 | 0.7787 (8.9) | 0.7771 (8.8) | 0.7748 (8.6) | 0.7732 (8.8) | 0.7725 (8.8) |
| 16384 | 0.7807 (6.8) | 0.7796 (6.9) | 0.7775 (6.9) | 0.7753 (6.9) | 0.7734 (6.7) |
| 32768 | 0.7832 (5.7) | 0.7825 (6.0) | 0.7810 (5.9) | 0.7788 (6.0) | 0.7757 (5.8) |

ml20m, rank 64: 3 pass(es), B=4096, a0=1.0, seeds [0, 1, 2]

| variant | held-out RMSE | vs default | training objective | seconds |
|---|---|---|---|---|
| default | 0.7723 ± 0.0004 | +0.0000 ± 0.0000 | 0.6061 | 12.74 |
| constant step | 0.7721 ± 0.0003 | -0.0002 ± 0.0001 | 0.6174 | 14.65 |
| fade only touched items | 0.7722 ± 0.0004 | -0.0000 ± 0.0000 | 0.6060 | 13.52 |
| no last-pass clean-up | 0.7734 ± 0.0004 | +0.0011 ± 0.0001 | 0.6088 | 12.27 |
| re-solve items every 5 batches | 0.7735 ± 0.0004 | +0.0013 ± 0.0001 | 0.6073 | 12.98 |
| textbook step (t+2)^-0.5 | 0.7742 ± 0.0004 | +0.0019 ± 0.0001 | 0.6077 | 12.59 |
| textbook step (t+2)^-0.75 | 0.7750 ± 0.0004 | +0.0028 ± 0.0000 | 0.6072 | 12.40 |
| textbook step (t+2)^-1.0 | 0.7791 ± 0.0004 | +0.0068 ± 0.0002 | 0.6124 | 15.58 |

## MovieLens-20M, rank 64: ridge strength (split seed 100, online at B=4096, a0=1.0)

| ridge | seeds | ALS 10 sweeps | ALS 25 | ALS 40 | online 1 pass | 2 | 3 | 5 | 10 | training objective: ALS 40 | online 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.03 | [100] | 0.7825 | 0.7798 | 0.7793 | 0.7767 | 0.7774 | 0.7778 | 0.7775 | 0.7775 | 0.5398 | 0.5432 |
| 0.04 | [100] | 0.7753 | 0.7732 | 0.7728 | 0.7731 | 0.7720 | 0.7721 | 0.7716 | 0.7718 | 0.6035 | 0.6062 |
| 0.05 | [100] | 0.7752 | 0.7735 | 0.7732 | 0.7758 | 0.7737 | 0.7734 | 0.7728 | 0.7727 | 0.6567 | 0.6587 |

## MovieLens-20M: by item popularity

ml20m.npy rank 64 ridge 0.04 seed 0

| training ratings of the item | held-out ratings | ALS 40 sweeps | online 3 passes B=4096 a0=1.0 | online, no clean-up |
|---|---|---|---|---|
| 0-0 | 435 | 1.1973 | 1.1973 | 1.1973 |
| 1-4 | 1,814 | 0.9884 | 0.9891 | 0.9902 |
| 5-19 | 5,290 | 0.9031 | 0.9027 | 0.9093 |
| 20-99 | 24,728 | 0.8518 | 0.8470 | 0.8574 |
| 100-499 | 106,106 | 0.7980 | 0.7941 | 0.8015 |
| 500-... | 1,861,648 | 0.7694 | 0.7689 | 0.7695 |
| all | 2,000,021 | 0.7727 | 0.7720 | 0.7731 |

| | ALS 40 sweeps | online 3 passes B=4096 a0=1.0 | online, no clean-up |
|---|---|---|---|
| mean factor norm, items with 1-4 training ratings (8430) | 0.814 | 0.957 | 0.896 |
| mean factor norm, items with 5-99 (9612) | 1.747 | 1.776 | 1.678 |
| mean factor norm, items with 100+ (8282) | 1.755 | 1.672 | 1.709 |
| largest factor norm | 4.33 | 4.45 | 4.45 |
| non-finite entries | 0 | 0 | 0 |

## MovieLens-1M (CPU runs; times not comparable with the GPU tables)

### ml1m, rank 32, ridge 0.06, seeds [0, 1, 2]

| method | passes | settings | seconds | held-out RMSE | vs ALS 40 | ALS sweeps to match (per seed) | training objective |
|---|---|---|---|---|---|---|---|
| ALS | 1 | | 0.11 | 1.0566 ± 0.0046 | +0.2020 | | 1.1174 |
| ALS | 2 | | 0.22 | 0.8831 ± 0.0020 | +0.0285 | | 0.8111 |
| ALS | 3 | | 0.33 | 0.8663 ± 0.0023 | +0.0117 | | 0.7831 |
| ALS | 5 | | 0.56 | 0.8589 ± 0.0026 | +0.0043 | | 0.7696 |
| ALS | 10 | | 1.11 | 0.8554 ± 0.0027 | +0.0008 | | 0.7627 |
| ALS | 15 | | 1.66 | 0.8546 ± 0.0025 | -0.0000 | | 0.7610 |
| ALS | 20 | | 2.24 | 0.8544 ± 0.0025 | -0.0002 | | 0.7603 |
| ALS | 30 | | 3.55 | 0.8545 ± 0.0024 | -0.0001 | | 0.7597 |
| ALS | 40 | | 4.74 | 0.8546 ± 0.0024 | +0.0000 | | 0.7595 |
| ALS, best sweep of 40 | [21, 15, 22] | | | 0.8544 ± 0.0025 | -0.0002 | | |
| online (formula) | 1 | B=256, a0=0.055 | 0.57 | 0.8578 ± 0.0016 | +0.0032 | [6, 5, 8] | 0.7793 |
| online (formula) | 2 | B=512, a0=0.0951 | 0.69 | 0.8562 ± 0.0017 | +0.0016 | [8, 7, 13] | 0.7699 |
| online (formula) | 3 | B=512, a0=0.0909 | 1.05 | 0.8559 ± 0.0017 | +0.0013 | [9, 7, 15] | 0.7673 |
| online (formula) | 5 | B=1024, a0=0.1604 | 1.11 | 0.8560 ± 0.0021 | +0.0014 | [8, 8, 12] | 0.7664 |
| online (formula) | 10 | B=1024, a0=0.1486 | 2.38 | 0.8553 ± 0.0022 | +0.0007 | [10, 9, 14] | 0.7636 |
| online (fixed) | 1 | B=128, a0=1.0 | 0.91 | 0.8601 ± 0.0022 | +0.0055 | [5, 5, 5] | 0.7970 |
| online (fixed) | 2 | B=128, a0=1.0 | 2.09 | 0.8542 ± 0.0018 | -0.0004 | [15, 11, None] | 0.7813 |
| online (fixed) | 3 | B=128, a0=1.0 | 2.36 | 0.8526 ± 0.0022 | -0.0020 | [None, None, None] | 0.7755 |
| online (fixed) | 5 | B=128, a0=1.0 | 3.98 | 0.8520 ± 0.0022 | -0.0026 | [None, None, None] | 0.7709 |
| online (fixed) | 10 | B=128, a0=1.0 | 8.34 | 0.8520 ± 0.0023 | -0.0026 | [None, None, None] | 0.7668 |
| online (fixed) | 1 | B=256, a0=0.5 | 0.56 | 0.8554 ± 0.0017 | +0.0008 | [9, 8, None] | 0.7792 |
| online (fixed) | 2 | B=256, a0=0.5 | 1.14 | 0.8531 ± 0.0025 | -0.0015 | [None, None, None] | 0.7685 |
| online (fixed) | 3 | B=256, a0=0.5 | 1.84 | 0.8527 ± 0.0024 | -0.0020 | [None, None, None] | 0.7652 |
| online (fixed) | 5 | B=256, a0=0.5 | 2.47 | 0.8529 ± 0.0023 | -0.0017 | [None, None, None] | 0.7621 |
| online (fixed) | 10 | B=256, a0=0.5 | 4.78 | 0.8536 ± 0.0026 | -0.0010 | [None, None, None] | 0.7604 |

ml1m, rank 32: grid at 1 pass(es), seed 0, held-out RMSE (seconds)

| batch size | step 0.03 | step 0.1 | step 0.25 | step 0.5 | step 1.0 |
|---|---|---|---|---|---|
| 64 | 0.8551 (1.3) | 0.8541 (1.4) | 0.8538 (1.4) | 0.8565 (1.4) | 0.8647 (1.4) |
| 128 | 0.8559 (0.9) | 0.8552 (0.9) | 0.8544 (0.9) | 0.8548 (0.9) | 0.8594 (0.9) |
| 256 | 0.8571 (0.6) | 0.8566 (0.6) | 0.8559 (0.6) | 0.8551 (0.6) | 0.8562 (0.6) |
| 512 | 0.8593 (0.4) | 0.8590 (0.4) | 0.8581 (0.4) | 0.8569 (0.4) | 0.8556 (0.4) |
| 1024 | 0.8645 (0.3) | 0.8643 (0.3) | 0.8639 (0.3) | 0.8631 (0.3) | 0.8616 (0.3) |
| 2048 | 0.8747 (0.2) | 0.8747 (0.2) | 0.8747 (0.2) | 0.8747 (0.2) | 0.8747 (0.2) |

ml1m, rank 32: grid at 3 pass(es), seed 0, held-out RMSE (seconds)

| batch size | step 0.03 | step 0.1 | step 0.25 | step 0.5 | step 1.0 |
|---|---|---|---|---|---|
| 64 | 0.8543 (3.8) | 0.8521 (3.8) | 0.8517 (3.8) | 0.8508 (3.8) | 0.8527 (3.8) |
| 128 | 0.8547 (2.4) | 0.8538 (2.4) | 0.8515 (2.3) | 0.8518 (2.3) | 0.8513 (2.4) |
| 256 | 0.8553 (1.5) | 0.8545 (1.5) | 0.8535 (1.5) | 0.8521 (1.5) | 0.8526 (1.5) |
| 512 | 0.8559 (1.0) | 0.8553 (1.0) | 0.8544 (1.0) | 0.8531 (1.0) | 0.8525 (1.0) |
| 1024 | 0.8571 (0.7) | 0.8567 (0.7) | 0.8558 (0.7) | 0.8546 (0.7) | 0.8532 (0.7) |
| 2048 | 0.8587 (0.6) | 0.8585 (0.6) | 0.8582 (0.5) | 0.8575 (0.6) | 0.8560 (0.6) |

ml1m, rank 32: 3 pass(es), B=128, a0=1.0, seeds [0, 1, 2]

| variant | held-out RMSE | vs default | training objective | seconds |
|---|---|---|---|---|
| default | 0.8526 ± 0.0022 | +0.0000 ± 0.0000 | 0.7755 | 2.76 |
| constant step | 0.8731 ± 0.0023 | +0.0205 ± 0.0010 | 0.8285 | 2.41 |
| fade only touched items | 0.8530 ± 0.0025 | +0.0004 ± 0.0003 | 0.7750 | 2.46 |
| no last-pass clean-up | 0.8624 ± 0.0021 | +0.0098 ± 0.0002 | 0.7944 | 2.24 |
| textbook step (t+2)^-0.5 | 0.8578 ± 0.0019 | +0.0052 ± 0.0007 | 0.7803 | 2.48 |
| textbook step (t+2)^-0.75 | 0.8536 ± 0.0021 | +0.0010 ± 0.0009 | 0.7641 | 2.28 |
| textbook step (t+2)^-1.0 | 0.8549 ± 0.0019 | +0.0023 ± 0.0007 | 0.7696 | 2.24 |
| re-solve items every 5 batches | 0.8534 ± 0.0024 | +0.0008 ± 0.0002 | 0.7754 | 1.90 |

ml1m, rank 32: 3 pass(es), B=512, a0=0.0909, seeds [0, 1, 2]

| variant | held-out RMSE | vs default | training objective | seconds |
|---|---|---|---|---|
| default | 0.8559 ± 0.0017 | +0.0000 ± 0.0000 | 0.7673 | 0.98 |
| constant step | 0.8552 ± 0.0018 | -0.0007 ± 0.0001 | 0.7657 | 1.08 |
| fade only touched items | 0.8559 ± 0.0017 | -0.0000 ± 0.0000 | 0.7673 | 1.35 |
| no last-pass clean-up | 0.8558 ± 0.0017 | -0.0001 ± 0.0001 | 0.7713 | 1.04 |
| textbook step (t+2)^-0.5 | 0.8544 ± 0.0019 | -0.0015 ± 0.0001 | 0.7668 | 1.03 |
| textbook step (t+2)^-0.75 | 0.8546 ± 0.0018 | -0.0013 ± 0.0001 | 0.7666 | 0.99 |
| textbook step (t+2)^-1.0 | 0.8565 ± 0.0017 | +0.0006 ± 0.0001 | 0.7720 | 1.00 |
| re-solve items every 5 batches | 0.8585 ± 0.0025 | +0.0027 ± 0.0007 | 0.7730 | 0.85 |

## MovieLens-1M, rank 32: ridge strength (online at B=128, a0=1.0)

| ridge | seeds | ALS 10 sweeps | ALS 25 | ALS 40 | online 1 pass | 2 | 3 | 5 | 10 | training objective: ALS 40 | online 3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.05 | [0, 1, 2] | 0.8599 | 0.8589 | 0.8592 | 0.8599 | 0.8554 | 0.8550 | 0.8549 | 0.8550 | 0.7119 | 0.7308 |
| 0.06 | [0, 1, 2] | 0.8554 | 0.8545 | 0.8546 | 0.8601 | 0.8542 | 0.8526 | 0.8520 | 0.8520 | 0.7595 | 0.7755 |
| 0.07 | [0, 1, 2] | 0.8546 | 0.8538 | 0.8539 | 0.8627 | 0.8562 | 0.8545 | 0.8535 | 0.8528 | 0.8013 | 0.8149 |
| 0.08 | [0, 1, 2] | 0.8565 | 0.8558 | 0.8559 | 0.8669 | 0.8606 | 0.8588 | 0.8572 | 0.8563 | 0.8380 | 0.8492 |
| 0.09 | [0, 1, 2] | 0.8605 | 0.8598 | 0.8598 | 0.8719 | 0.8660 | 0.8642 | 0.8628 | 0.8618 | 0.8700 | 0.8788 |

## MovieLens-1M: by item popularity

ml1m.npy rank 32 ridge 0.06 seed 0

| training ratings of the item | held-out ratings | ALS 40 sweeps | online 3 passes B=128 a0=1.0 | online, no clean-up |
|---|---|---|---|---|
| 0-0 | 11 | 1.2750 | 1.2750 | 1.2750 |
| 1-4 | 88 | 1.2131 | 1.2385 | 1.2487 |
| 5-19 | 577 | 1.1147 | 1.0978 | 1.1308 |
| 20-99 | 6,051 | 0.9407 | 0.9327 | 0.9525 |
| 100-499 | 37,557 | 0.8680 | 0.8639 | 0.8782 |
| 500-... | 55,932 | 0.8290 | 0.8295 | 0.8350 |
| all | 100,216 | 0.8532 | 0.8513 | 0.8613 |

| | ALS 40 sweeps | online 3 passes B=128 a0=1.0 | online, no clean-up |
|---|---|---|---|
| mean factor norm, items with 1-4 training ratings (303) | 0.966 | 1.207 | 1.054 |
| mean factor norm, items with 5-99 (1461) | 1.836 | 1.784 | 1.770 |
| mean factor norm, items with 100+ (1931) | 1.521 | 1.388 | 1.477 |
| largest factor norm | 2.93 | 3.25 | 3.25 |
| non-finite entries | 0 | 0 | 0 |
