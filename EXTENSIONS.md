# Beyond the paper

Follow-up experiments on top of the reproduction in the README. None of this is from the e-means paper; the
ideas tested are old ones from the vector-quantization, neural-codebook and statistics literature, and prior
art is named where I found it. Everything ran on one Apple M2 Max (GPU through PyTorch, 12 CPU cores) on
2026-11-10/11. Raw logs and full tables are in `results/beyond/`; scripts are in `experiments/`.

Contents:

1. [Summary](#1-summary)
2. [Do more passes keep helping?](#2-do-more-passes-keep-helping)
3. [Search indexes on six datasets](#3-search-indexes-on-six-datasets)
4. [Old ideas the recipe does not use](#4-old-ideas-the-recipe-does-not-use)
5. [The one that mattered: a size penalty](#5-the-one-that-mattered-a-size-penalty)
6. [Product quantization](#6-product-quantization)
7. [Gaussian mixtures](#7-gaussian-mixtures)
8. [Topic models (LDA) and NMF](#8-topic-models-lda-and-nmf)
9. [Hidden Markov models](#9-hidden-markov-models)
10. [Recommender factorization (ALS)](#10-recommender-factorization-als)
11. [PageRank](#11-pagerank)
12. [My implementation against the authors' code](#12-my-implementation-against-the-authors-code)
13. [Caveats](#13-caveats)

## 1. Summary

Two questions. Does the recipe (update per batch, forgetting running totals, a big-then-small step curve, timed
repair, a final clean-up, settings from formulas) carry over to other classic algorithms? And are there old
ideas the recipe leaves out that would add to it?

**Where the recipe carried over**

| Algorithm | Result | Verdict |
|---|---|---|
| k-means for search indexes (6 datasets) | 3 passes ≈ 25 Lloyd passes at small `k`, 4–9% less scanning at `k` = 16,384; on one clumpy dataset 49% less. One case (text embeddings, `k` = 16,384) where it is 9% worse. | works, with one exception |
| Product quantization | 3 passes beat 100 Lloyd passes once the authors' low-dimension batch rule is used | works |
| Gaussian mixtures | CPU estimator: 250× faster than scikit-learn at 1M rows for the same score; needs k-means++ seeds on well-separated clusters | works |
| NMF | 3 passes ≈ 34–79 full passes; within 0.1% of scikit-learn's `NMF` in 4% of the time | works |
| Hidden Markov models | 1 pass ≈ 40–80 Baum-Welch passes, but only when the formulas are fed the number of symbols; the step curve, revival and clean-up add nothing | works after re-mapping |
| Recommender ALS | nothing with the formulas; with a hand-picked batch and step, 5× less time to the same error, and the gain is partly regularization | partly |
| Topic models (LDA) | the published step is 4–8× too large; tuned, it ties the 2010 online rule | no real gain |
| PageRank | only "use fresh values" applies, which is Gauss-Seidel | does not apply |

The pattern: the recipe pays where the full-pass method needs many passes and gets stuck (k-means, mixtures,
hidden Markov models, NMF). It pays little where the full-pass method already converges in ten passes (LDA,
ALS) or where there is no noise to average (PageRank). The fitted formulas are specific to k-means: outside it,
the batch size and step had to be re-chosen in every case but NMF and mixtures.

**Old ideas added to the recipe**

- A dozen ideas tested on k-means. All but one did nothing, hurt, or gave no more than the same time spent on
  an extra pass (section 4). The paper's own choices held up.
- The exception, a cluster-size penalty from 1988, is a clear win for search indexes on some data: 3–11% fewer
  vectors scanned on image descriptors, 60–80% fewer on one clumpy dataset (confirmed in a real Faiss index:
  queries 4.6× faster than with Faiss's own k-means), nothing on word vectors. A faded version is the safe
  default. It is in the estimator as `EMeans(balance=0.3)`.
- A rule for the penalty's strength predicts the best value on three datasets, is wrong on one and cannot be
  fitted on another.
- For mixtures, two annealing ideas made things worse. For hidden Markov models, growing the state count did.

**Corrections to earlier statements in this repository**

- The paper does report search recall (its Appendix L), contrary to what the README first said.
- The authors' code is public; reading it exposed one bug in mine (a step of exactly 1) and one rule I had
  missed (smaller batches in low dimension). Both are fixed and the affected runs redone.

## 2. Do more passes keep helping?

Not in a straight line. Each doubling of the passes buys less: big steps up to three passes, about two more
points from three to five, and about two more from five to twenty-five.
SIFT1M, `k` = 16,384, 100,000 held-out queries; change against 25 Lloyd passes (3 seeds; 2 for 15 and 25):

| Passes | Train time | Vectors scanned for 95% recall@1 | for 95% recall@10 | Test MSE | Train MSE |
|---|---|---|---|---|---|
| Lloyd 25 | 18.9 s | 3,948 | 6,158 | 42,515 | 38,260 |
| Lloyd 50 / 100 (1 seed) | 37 s / 74 s | 0.0% / +0.2% | −0.3% / −0.2% | −0.02% | −0.09% |
| 1 | 0.9 s | +11.6% ± 0.7 | +10.8% ± 0.4 | +0.90% | +3.51% |
| 2 | 1.7 s | −0.4% ± 0.2 | −1.1% ± 0.1 | −0.09% | +0.39% |
| 3 | 2.4 s | −3.8% ± 0.1 | −3.2% ± 0.2 | −0.20% | −0.26% |
| 4 | 3.2 s | −4.2% ± 0.6 | −4.5% ± 0.3 | −0.43% | −0.73% |
| 5 | 4.0 s | −6.0% ± 0.4 | −5.3% ± 0.3 | −0.42% | −1.03% |
| 6 | 4.4 s | −5.1% ± 0.3 | −4.8% ± 0.4 | −0.22% | −0.96% |
| 8 | 5.8 s | −6.3% ± 0.7 | −6.1% ± 0.2 | −0.32% | −1.30% |
| 10 | 7.3 s | −6.3% ± 0.6 | −6.8% ± 0.5 | −0.32% | −1.51% |
| 15 | 9.3 s | −7.5% ± 0.8 | −6.7% ± 0.2 | −0.43% | −1.45% |
| 25 | 15.7 s | −8.2% ± 0.5 | −8.6% ± 0.1 | −0.45% | −1.89% |

- Held-out error stops improving at four passes (about −0.4%). Training error keeps falling, and search cost
  follows the training error, because the database vectors of an index *are* the training vectors.
- The curve is bumpy because the published formula doubles the batch size at 3, 6 and 15 passes; the first
  pass count on a larger batch (6, 15) is no better than the one before it.
- More Lloyd passes do nothing: 50 and 100 passes are within 0.2% of 25. Lloyd has converged to a worse
  solution, and e-means is past it from three passes on.
- Twenty-five e-means passes cost less time than 25 Lloyd passes (15.7 s against 18.9 s) and give an index
  that scans 8% less.
- At `k` = 4,096 (one seed) the curve flattens sooner: every count from 5 to 25 passes is 1–3% below Lloyd 25,
  and so are 50 and 100 Lloyd passes. On GloVe at `k` = 4,096 (5 seeds) it keeps going: −1.4% ± 2.2 at three
  passes, −3.1% at five, −5.8% ± 2.2 at ten.

A correction that came out of this sweep: my first run of 15 and 25 passes failed (worse than one pass at 15,
not-a-number at 25). The cause was in my code, not in the formulas. At those settings the formula caps the
step at 1, and my update zeroed the running count of every cluster that the first batch missed. The authors'
released code leaves such clusters untouched; with the same handling (now in `emeans_torch.py`,
`emeans_lab.py` and `emeans_sklearn.py`) the numbers are the ones above. No earlier result used a step of 1
with empty clusters, so nothing else changes. See section 12.

## 3. Search indexes on six datasets

The README measured search-index quality on SIFT1M only. Here are five more public datasets, including modern
text embeddings. Method as before: every database vector is filed under its nearest centroid, a query scans
the nearest lists, and the measure is how many vectors must be scanned to reach a recall target. Queries are
held-out vectors with exact neighbours recomputed (10,000–100,000 per dataset). Change is against centroids
from 25 Lloyd passes, paired by seed. The last three columns add the size penalty of section 5.

| Data | Dim. | Vectors | k | Seeds | Recall target | Lloyd 25: vectors scanned | e-means 3 | e-means 5 | 3 passes + penalty 0.3, faded | + penalty 0.1, held | + penalty 0.3, held | List imbalance: Lloyd / e-means 3 / faded |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dbpedia | 1536 | 950,000 | 4,096 | 2 | 95 % R@1 | 18,070 | −0.8% ± 0.0 | −1.6% | −0.3% ± 0.3 | −1.8% ± 0.4 | +0.1% ± 0.0 | 1.22 / 1.28 / 1.07 |
| dbpedia | 1536 | 950,000 | 16,384 | 3 | 95 % R@1 | 9,059 | +8.9% ± 2.8 |  | −1.4% ± 2.1 | −2.4% ± 1.9 |  | 1.31 / 1.40 / 1.08 |
| deep1m | 96 | 1,000,000 | 4,096 | 3 | 95 % R@1 | 6,335 | +2.0% ± 0.8 | −1.4% ± 0.5 | −4.5% ± 1.2 | −5.1% ± 1.0 | −6.1% ± 1.0 | 1.12 / 1.14 / 1.04 |
| deep1m | 96 | 1,000,000 | 16,384 | 2 | 95 % R@1 | 3,258 | −5.3% ± 1.1 | −8.0% ± 0.4 | −11.1% ± 0.7 | −11.3% ± 0.1 | −11.5% ± 0.3 | 1.23 / 1.19 / 1.06 |
| fmnist | 784 | 60,000 | 256 | 3 | 95 % R@1 | 926 | −2.2% ± 1.7 | −2.5% ± 2.5 | −5.1% ± 2.1 | −4.7% ± 2.5 | −5.6% ± 0.8 | 1.16 / 1.16 / 1.07 |
| fmnist | 784 | 60,000 | 1,024 | 3 | 95 % R@1 | 403 | −2.9% ± 1.1 | −5.2% ± 3.5 | −9.0% ± 0.8 | −7.1% ± 1.9 | −8.4% ± 1.8 | 1.27 / 1.20 / 1.07 |
| glove | 100 | 1,083,514 | 4,096 | 3 | 95 % R@1 | 48,848 | −1.4% ± 2.4 | −2.5% ± 2.1 | −0.6% ± 1.1 | +1.2% ± 1.4 | +2.0% ± 1.6 | 1.15 / 1.15 / 1.04 |
| glove | 100 | 1,083,514 | 16,384 | 2 | 90 % R@1 | 11,019 | −8.5% ± 0.1 |  | −7.9% ± 0.2 | −6.3% ± 0.2 | −0.7% ± 0.6 | 1.33 / 1.26 / 1.05 |
| nyt256 | 256 | 250,000 | 1,024 | 3 | 95 % R@1 | 70,054 | −15.4% ± 5.0 | −21.0% ± 3.4 | −51.3% ± 3.6 | −54.0% ± 22.6 | −71.8% ± 10.7 | 5.19 / 1.82 / 1.26 |
| nyt256 | 256 | 250,000 | 4,096 | 3 | 95 % R@1 | 47,541 | −49.0% ± 1.2 | −59.8% ± 3.3 | −79.6% ± 1.1 | −83.6% ± 2.3 | −80.4% ± 1.5 | 7.36 / 2.08 / 1.20 |
| sift | 128 | 1,000,000 | 1,024 | 3 | 95 % R@1 | 14,871 | −0.6% ± 0.9 | −1.2% ± 0.8 | −3.7% ± 0.7 | −3.5% ± 1.3 | −3.8% ± 0.8 | 1.08 / 1.09 / 1.03 |
| sift | 128 | 1,000,000 | 4,096 | 2 | 95 % R@1 | 7,680 | +1.0% ± 0.5 | −1.5% ± 0.5 | −4.1% ± 0.1 | −4.3% ± 0.1 | −5.3% ± 0.7 | 1.12 / 1.14 / 1.04 |
| sift | 128 | 1,000,000 | 16,384 | 3 | 95 % R@1 | 3,948 | −3.7% ± 0.1 | −6.0% ± 0.4 | −10.0% ± 0.9 | −9.6% ± 0.2 | −10.6% ± 0.8 | 1.22 / 1.19 / 1.06 |

| Dataset | What it is |
|---|---|
| sift | SIFT1M image descriptors (TEXMEX) |
| deep1m | first 1.1M of DEEP image descriptors, unit length (ann-benchmarks `deep-image-96-angular`) |
| glove | GloVe word vectors, unit length (ann-benchmarks `glove-100-angular`) |
| dbpedia | DBpedia entities embedded with OpenAI `text-embedding-3-large`, 1,536 dimensions (Hugging Face, Qdrant) |
| fmnist | Fashion-MNIST images as raw pixels (ann-benchmarks `fashion-mnist-784-euclidean`) |
| nyt256 | New York Times article embeddings, unit length (ann-benchmarks `nytimes-256-angular`) |

What the plain recipe does (columns "e-means 3" and "e-means 5"):

- On image descriptors, raw images and word vectors, three passes are within about ±3% of 25 Lloyd passes at
  small `k` and 4–9% better at `k` = 16,384. Five passes are at least as good as Lloyd on all of those.
- On the 1,536-dimension text embeddings it depends on `k`. At 4,096 there is no real difference (±2%), at one
  seventh of the training time (5.1 s against 36.5 s). At 16,384 (3 seeds) the three-pass index is **9% worse**
  than Lloyd's, because its lists are less even (imbalance 1.40 against 1.31); with the faded size penalty it is
  level with Lloyd's (−1.4% ± 2.1), at 17.8 s of training against 139 s.
- On the New York Times embeddings the difference is large: the index from three passes scans 49% fewer
  vectors than the Lloyd index at `k` = 4,096. That dataset is very clumpy. Lloyd from a random start ends
  with more than half of its 4,096 clusters holding fewer than 10 vectors and a few holding over a thousand
  (imbalance 7.4, where 1.0 is perfectly even); a k-means++ start and 100 passes do not fix it. The revival
  step in e-means is what helps here (imbalance 2.1).
- One pass is 4–13% worse than Lloyd on five of the six datasets, as on SIFT.
- A k-means++ start for Lloyd made the index worse on Fashion-MNIST (by 5–11% against a random start), as it
  did for e-means on SIFT in section 4.

### Checked in a real Faiss index, against Faiss's own k-means

The numbers above come from my own recall arithmetic and my own Lloyd baseline. For the dataset with the
largest effect I repeated the comparison end to end with Faiss: centroids from `faiss.Kmeans` (25 iterations,
no subsampling), a real `IndexIVFFlat`, 40,000 held-out queries, 3 seeds, `k` = 4,096, measured at 95% recall@1.

| Centroids from | Distance computations per query | vs Faiss k-means | Query time | vs Faiss k-means |
|---|---|---|---|---|
| Faiss k-means, 25 iterations (47 s on CPU) | 46,432 | – | 316 µs | – |
| Faiss k-means as an IVF index trains it by default (10 iterations) | 44,036 | −5% ± 3 | 291 µs | −8% ± 2 |
| My Lloyd, 25 passes | 47,976 | +4% ± 8 | 315 µs | +1% ± 10 |
| e-means, 3 passes (0.4 s on GPU) | 24,535 | −47% ± 5 | 168 µs | −46% ± 6 |
| e-means, 5 passes | 19,378 | −58% ± 4 | 130 µs | −58% ± 6 |
| e-means, 3 passes + size penalty 0.3, faded | 9,907 | −79% ± 2 | 69 µs | −78% ± 3 |
| e-means, 5 passes + size penalty 0.3, faded | 8,362 | −82% ± 2 | 57 µs | −82% ± 3 |
| e-means, 3 passes + size penalty 0.1, held | 7,922 | −83% ± 3 | 57 µs | −82% ± 4 |

So on this dataset a query is 1.9× faster with plain e-means centroids and 4.6–5.5× faster with the size
penalty added, at the same recall, in the library people actually use. My Lloyd baseline and Faiss's k-means
agree with each other. This is the extreme case; the same check on SIFT1M is in section 5.

## 4. Old ideas the recipe does not use

The recipe has five ingredients. The older literature has many more that the paper does not combine with them.
I added them to the GPU implementation as switches (`experiments/emeans_lab.py`, which with all switches off
reproduces `emeans_torch.py` bit for bit) and ran each one alone on top of 3-pass e-means (a dozen ideas, some in two or three variants). SIFT1M,
`k` = 4,096, 3 seeds; change against plain 3-pass e-means. For scale: a fourth pass is worth −1.0% / −1.4%
on the two search columns and costs 33% more time.

| Idea | Where it comes from | Scan, 95% R@1 | Scan, 95% R@10 | Test MSE | Time | Verdict |
|---|---|---|---|---|---|---|
| Step curve: lower or higher end point, linear | – | −0.5 to +0.3% | −0.6 to −0.4% | ±0.1% | same | no difference |
| Step curve: exponential decay | – | +0.2% | +0.1% | +0.17% | same | slightly worse |
| Step curve: constant | – | +2.9% | +2.8% | +0.12% | same | worse (the paper's ablation agrees) |
| Batches 4× smaller with a 4× smaller step | the direction my mixture grid favoured | −1.1% ± 0.7 | −1.4% ± 0.3 | −0.02% | +20% | small gain, costs time |
| Growing batches (start at 1/16 size) | Smith 2018; Newling & Fleuret 2016 | −1.5% ± 1.1 | −1.7% ± 0.5 | −0.07% | +25% | same as above |
| Re-seed starved clusters at a random batch vector | VQ-VAE / Jukebox codebook reset | −0.7% | −0.6% | +0.15% | same | worse error (the paper's ablation agrees) |
| Re-seed at a high-error batch vector | k-means++ sampling applied online | +0.1% | +0.4% | +0.20% | same | worse |
| Split the cluster with the largest error, not the largest count | Gersho 1979; Fritzke 1997 (LBG-U) | +0.9% | −0.1% | −0.03% | same | no gain (+3% scan at `k` = 16,384) |
| Also relocate low-utility clusters | Fritzke 1997 (LBG-U) | −0.1 to +0.4% | −0.7 to +0.3% | ±0.1% | 3.6× | no gain, slow |
| Soft start: each vector also pulls its 2nd–8th nearest centroid | neural gas, Martinetz 1993 | −0.4 to +0.2% | −0.6 to −0.4% | ±0.1% | 1.6–2× | no gain, slow |
| Look-ahead on the centroids | Nesterov momentum | +0.9 to +4.8% | +1.0 to +3.5% | up to +0.18% | same | worse |
| k-means++ start (on a subsample) | Arthur & Vassilvitskii 2007 | +2.2% ± 0.5 | +1.6% ± 1.0 | −0.01% | 1.6× | worse for search |
| Codebook doubling before the main passes | Linde, Buzo, Gray 1980 | −1.2% | −1.2% | −0.12% | 1.45× | same as spending the time on a 4th pass |
| One exact Lloyd pass at the end | Bottou & Bengio 1995 | −0.4% | −0.5% | −0.03% | 1.3× | worse than a 4th online pass |
| Stricter revival (threshold 0.3 instead of 0.1 of the mean) | – | −0.5% ± 0.8 | −1.1% ± 0.4 | 0.00% | same | small gain |
| **Size penalty when assigning ("conscience")** | DeSieno 1988; balanced k-means | **−4.1 to −5.3%** | **−4.1 to −5.3%** | ±0.04% | +4% | **the only clear win** |

Apart from the last row, nothing here beats spending the same time on an extra pass. The recipe as published is hard to improve on its own
terms: its step curve, its revival rule and its random start all held up, and the things the paper already
ablated (constant step, re-seeding from the batch) came out the same way here. The exception is the last row.

## 5. The one that mattered: a size penalty

### What it is

When a vector picks its centroid, busy centroids are handicapped:

`assign x to the j that minimises  |x − c_j|² + λ · D · (k · n_j / n − 1)`

where `n_j / n` is cluster j's running share of the data (e-means already keeps it), `D` is the running mean
squared distance, and `λ` is a strength with no units. Training takes about 2–4% longer.

This is the assignment step for the objective "clustering error + (λ·D/2) × imbalance", where
imbalance = `k · Σ (n_j/n)²` (1.0 when all clusters are the same size). Imbalance is exactly what an
inverted-file index pays for uneven lists: with one list probed per query, the expected number of vectors
scanned is `(n/k) × imbalance`. So `λ` is an exchange rate between clustering error and list balance.

**Prior art, so nobody is misled.** The penalty is DeSieno's "conscience" for competitive learning (1988). The
same linear size penalty is used for search indexes as balanced k-means: Liu et al. 2018 (FCFC), SPANN 2021,
Microsoft SPTAG, Lance, and MicroNN 2025 in mini-batch form. Published results are mixed: SPANN reports a
gain, while Tavenard, Jégou & Amsaleg 2011 and an Elastic engineering post report that balancing hurt. What
I did not find elsewhere: the penalty combined with the e-means running totals, a fade-out version that ends
as plain k-means, a strength expressed in units of the running error, and a rule for that strength. Those
absences come from one literature search, not a proof.

Two versions, same strength `λ`:

- **Faded**: `λ` follows the cosine curve down to zero, so training ends as plain k-means.
- **Held**: `λ` stays to the end.

### Results

Section 3's table shows the penalty against Lloyd in its right-hand columns. Here it is against plain 3-pass
e-means, which takes the same training time (vectors scanned for 95% recall@1):

| Data | Faded, λ = 0.3 | Held, λ = 0.3 | Comment |
|---|---|---|---|
| SIFT, k = 1,024 / 4,096 / 16,384 | −3.1% / −5.4% / −6.5% | −3.2% / −6.5% / −7.1% | 2–3 seeds, spread about ±1 |
| DEEP, k = 4,096 / 16,384 | −6.4% / −6.1% | −7.9% / −6.5% | 2–3 seeds |
| DBpedia text embeddings, k = 4,096 | +0.5% (95% R@1), −1.6% (90% R@10) | +0.9%, −1.6% | no real effect; 2 seeds |
| DBpedia text embeddings, k = 16,384 | −9.5% ± 0.5 | (0.1 held: −10.4% ± 1.5) | 3 seeds; plain e-means is 9% behind Lloyd here, so this only brings it level |
| Fashion-MNIST, k = 256 / 1,024 | −2.9% ± 3.8 / −6.4% ± 0.3 | −3.5% / −5.7% | 3 seeds |
| GloVe, k = 4,096 | +0.9% ± 2.4 (95% R@1), +1.8% ± 0.6 (90% R@10) | +3.5% ± 1.7, +4.8% ± 0.6 | neutral to slightly worse faded; worse held |
| GloVe, k = 16,384 (90% R@1) | +0.7% | +8.6% | 2 seeds |
| New York Times, k = 1,024 / 4,096 | −42% / −60% | −66% / −62% | 3 seeds; plain e-means is itself 15–49% below Lloyd here |

- The gain depends on the data: very large where plain k-means leaves many starved clusters (New York Times),
  3–8% on image descriptors and raw images, nothing on the text embeddings at `k` = 4,096 and 10% at 16,384
  (where it only repairs what plain e-means lost against Lloyd), and a small loss on word vectors.
- The faded version is the safer one: its worst case was +1 to +2% on GloVe. The held version is 1–2 points
  better on image descriptors and 4–9% worse on GloVe.
- It stacks with extra passes. SIFT1M, `k` = 16,384: five passes with the held penalty scan 15% fewer
  vectors than 25 Lloyd passes, against 6% for five plain passes; two passes with the penalty beat ten without.
- It is not a substitute for revival, but nearly: with revival switched off and the penalty on, the SIFT index
  is still better than plain e-means with revival (−5.8% / −6.1% at `k` = 16,384, 2 seeds).
- The naive full-pass version does not work. Lloyd with the same penalty, using cluster sizes from the previous
  pass, oscillates and ends far worse above λ ≈ 0.1 (+50% to +75% scan cost at λ = 0.3). Published balanced k-means
  avoids this with sequential or flow-based assignment; with per-batch running sizes the plain rule is stable.

### Strength

Held penalty at each strength, change in vectors scanned against plain 3-pass e-means (last column: 25 Lloyd
passes, for reference):

| Data | k | Seeds | Scan cost at | λ=0.03 | λ=0.1 | λ=0.2 | λ=0.3 | λ=0.5 | λ=1 | λ=2 | Lloyd 25 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| dbp | 4,096 | 2 | 95% R@10 | −2.4 ± 0.5 | −3.5 ± 0.6 | −3.3 ± 0.4 | −2.0 ± 0.1 | +0.5 ± 0.2 | +15.3 ± 0.4 |  | −2.5 ± 0.4 |
| deep1m | 4,096 | 3 | 95% R@10 | −2.9 ± 0.5 | −5.0 ± 0.3 | −5.6 ± 0.1 | −5.3 ± 0.4 | −4.4 ± 0.2 | −0.7 ± 0.5 |  | −0.1 ± 0.6 |
| deep1m | 16,384 | 2 | 95% R@10 | −2.7 ± 0.1 | −5.7 ± 0.1 | −6.3 ± 0.2 | −6.5 ± 0.2 | −5.6 ± 0.1 | −2.3 ± 0.3 |  | +5.3 ± 0.2 |
| glove | 4,096 | 3 | 95% R@1 | +3.3 ± 2.2 | +2.7 ± 2.2 | +4.0 ± 2.5 | +3.5 ± 1.7 | +9.5 ± 1.7 | +25.6 ± 2.3 |  | +1.5 ± 2.6 |
| nyt256 | 1,024 | 3 | 95% R@10 | −31.2 ± 6.3 | −47.4 ± 25.5 | −71.3 ± 9.7 | −70.2 ± 15.3 | −70.4 ± 9.4 | −58.4 ± 3.1 |  | +19.2 ± 6.3 |
| nyt256 | 4,096 | 3 | 95% R@10 | −67.3 ± 4.3 | −66.2 ± 7.7 | −70.1 ± 3.7 | −68.1 ± 0.6 | −58.3 ± 17.0 | −34.7 ± 26.7 |  | +97.6 ± 9.8 |
| sift | 1,024 | 3 | 95% R@10 | −1.8 ± 0.3 | −3.2 ± 0.1 | −3.4 ± 0.2 | −3.5 ± 0.3 | −2.6 ± 0.2 | −0.9 ± 0.5 | +5.4 ± 0.2 | +0.4 ± 0.2 |
| sift | 4,096 | 2 | 95% R@10 | −3.4 ± 0.0 | −5.3 ± 0.3 | −6.3 ± 0.4 | −6.0 ± 0.3 | −5.4 ± 0.0 | −2.9 ± 0.4 | +3.5 ± 0.6 | −0.6 ± 0.5 |
| sift | 16,384 | 2 | 95% R@10 | −3.1 ± 0.0 | −6.1 ± 0.0 | −7.6 ± 0.5 | −7.7 ± 0.3 | −7.3 ± 0.5 | −4.1 ± 0.1 | +4.0 ± 0.2 | +3.2 ± 0.0 |

The useful range is wide (0.1 to 0.5). Above 1 the centroids are pulled too far and the index gets worse.

### In a real Faiss index

SIFT1M, `k` = 16,384, a real Faiss `IndexIVFFlat` per set of centroids, 20,000 held-out queries, 3 seeds, at
95% recall@1; change against centroids from my 25 Lloyd passes:

| Centroids from | Distance computations per query | Change | Query time | Change |
|---|---|---|---|---|
| Lloyd, 25 passes | 4,387 | – | 28 µs | – |
| Faiss k-means as an IVF index trains it by default (10 iterations; 1 seed) | 4,504 | +2% | 29 µs | +4% |
| e-means, 3 passes | 4,268 | −2.7% ± 2.3 | 27 µs | −2% ± 4 |
| e-means, 5 passes | 4,129 | −5.9% ± 1.3 | 27 µs | −4% ± 1 |
| e-means, 3 passes + penalty 0.3, faded | 3,967 | −9.6% ± 0.5 | 26 µs | −6% ± 1 |
| e-means, 3 passes + penalty 0.3, held | 3,988 | −9.1% ± 1.5 | 26 µs | −5% ± 2 |
| e-means, 5 passes + penalty 0.3, held | 3,765 | −14.1% ± 1.6 | 26 µs | −8% ± 1 |

The distance counts agree with my arithmetic (−10% and −15% against Lloyd in section 3 and above). Query time
improves less than the distance count because, at this `k`, finding the nearest lists among 16,384 centroids
is a large fixed part of each query. The New York Times check is in section 3: there the penalty takes query
time from 316 µs (Faiss k-means) to 57–69 µs.

### It also lowers the clustering error itself when clusters are starved

The penalty was added for the search index, but it also changes the centroids' own error on held-out vectors
(3 passes, change against plain 3-pass e-means):

| Data | Faded 0.3: test MSE | Train MSE | Clusters with fewer than 10 vectors, plain → faded |
|---|---|---|---|
| New York Times, k = 4,096 (61 vectors per cluster) | −1.6% ± 0.04 | −1.9% | 216 → 44 (seed 0) |
| New York Times, k = 1,024 | −0.7% ± 0.1 | −0.7% | |
| SIFT, k = 16,384 (61 per cluster) | −0.43% ± 0.09 | −0.10% | |
| DEEP, k = 16,384 | −0.2% ± 0.1 | | |
| SIFT / DEEP / DBpedia at k ≤ 4,096, GloVe | 0.0 to +0.1% | | |

For scale, on SIFT at `k` = 16,384 no number of plain passes reached this: plain e-means levels off at −0.45%
against Lloyd after 25 passes, and three passes with the faded penalty are at about −0.6%. The effect shows up
where there are few vectors per cluster. A plausible reason, not tested: in high dimensions the best
quantizers have nearly equal-sized cells (Gersho 1979), k-means gets stuck in solutions that are less even
than that, and the penalty steers away from them early on.

### How strong? A rule, and where it fails

Scan cost at a fixed recall depends on two things the training controls: how even the lists are and how good
the centroids are. Fit, per dataset and `k`, over runs that use **no penalty** (different pass counts,
revival thresholds, batch sizes):

`log(scan cost) = a · log(imbalance) + b · log(training error) + constant`

Raising `λ` moves along a trade-off curve whose slope is `d(error)/d(imbalance) = −λ·D/2`. Cost is lowest
where the two effects cancel:

`λ* = 2a / (b · imbalance)`

| Data | k | Runs in fit | Recall target | a | b | R² | Predicted strength | Best on the grid (scan change) | At the predicted strength |
|---|---|---|---|---|---|---|---|---|---|
| dbp | 4,096 | 10 | R@1 95 % | 0.30 | 5.8 | 0.80 | 0.10 | 0.1 (−0.9%) | 0.1 (−0.9%) |
| dbp | 4,096 | 10 | R@10 95 % | 0.27 | 5.0 | 0.92 | 0.10 | 0.1 (−3.5%) | 0.1 (−3.5%) |
| deep1m | 4,096 | 27 | R@1 95 % | 0.84 | 5.0 | 0.95 | 0.33 | 0.3 (−7.9%) | 0.3 (−7.9%) |
| deep1m | 4,096 | 27 | R@10 95 % | 0.80 | 4.5 | 0.99 | 0.35 | 0.2 (−5.6%) | 0.3 (−5.3%) |
| deep1m | 16,384 | 18 | R@1 95 % | 0.62 | 4.6 | 0.99 | 0.26 | 0.3 (−6.5%) | 0.3 (−6.5%) |
| deep1m | 16,384 | 18 | R@10 95 % | 0.62 | 3.7 | 0.99 | 0.32 | 0.3 (−6.5%) | 0.3 (−6.5%) |
| glove | 4,096 | 27 | R@1 95 % | 0.93 | 7.0 | 0.85 | 0.26 | 0.1 (+2.7%) | 0.3 (+3.5%) |
| nyt256 | 1,024 | 9 | R@1 95 % | 0.46 | −5.9 | 0.74 | – | 0.03 (fit not usable) | |
| nyt256 | 1,024 | 9 | R@10 95 % | 0.23 | −8.0 | 0.67 | – | 0.03 (fit not usable) | |
| sift | 1,024 | 21 | R@1 95 % | 0.47 | 4.8 | 0.94 | 0.19 | 0.2 (−3.8%) | 0.2 (−3.8%) |
| sift | 1,024 | 21 | R@10 95 % | 0.51 | 4.4 | 0.98 | 0.22 | 0.3 (−3.5%) | 0.2 (−3.4%) |
| sift | 4,096 | 116 | R@1 95 % | 0.61 | 4.3 | 0.78 | 0.28 | 0.3 (−6.5%) | 0.3 (−6.5%) |
| sift | 4,096 | 116 | R@10 95 % | 0.61 | 4.0 | 0.93 | 0.30 | 0.2 (−6.3%) | 0.3 (−6.0%) |
| sift | 16,384 | 48 | R@1 95 % | 0.53 | 3.0 | 0.84 | 0.34 | 0.2 (−7.9%) | 0.3 (−7.0%) |
| sift | 16,384 | 48 | R@10 95 % | 0.46 | 2.8 | 0.84 | 0.31 | 0.3 (−7.7%) | 0.3 (−7.7%) |

- On SIFT and DEEP (image descriptors), `a` ≈ 0.5–0.8 and `b` ≈ 3–5, the rule says λ ≈ 0.2–0.35, and the
  measured optimum is 0.2–0.3 at every `k`. The form of the rule was fixed on SIFT; DEEP was run afterwards.
- On the 1,536-dimension text embeddings the rule says 0.1 and the grid agrees; the gain there is small
  (−1% to −3.5%, and only 10 runs in the fit).
- **On GloVe the rule is wrong.** It predicts 0.26; every held strength made the index worse. The two-factor
  model does not capture what the penalty does to word vectors.
- On the New York Times embeddings the fit is meaningless (error and imbalance move together across the
  penalty-free runs), yet any strength from 0.03 to 0.5 cuts scan cost by half or more.

So the rule explains why 0.3 is right where the penalty helps, and it is not a reliable test of whether the
penalty helps. That is the reason for recommending the faded version: it keeps most of the gain and did not
lose anywhere.

### In the scikit-learn estimator

`EMeans(n_clusters, balance=0.3)` is the faded version; `balance_end=1.0` holds it. Default is `balance=0`
(the paper's algorithm).

CPU only, scikit-learn API, New York Times embeddings (250,000 × 256), `k` = 4,096, 3 seeds. Error is the mean
squared distance to the nearest centre; the last two columns are from a real Faiss `IndexIVFFlat` built on each
set of centres, at 95% recall@1 (10,000 queries):

| Estimator | Fit time | Train error | Test error | Clusters under 10 vectors | Imbalance | Distance computations per query | Query time |
|---|---|---|---|---|---|---|---|
| `KMeans()` (k-means++, to convergence: 115 iterations) | 480 s | 0.7360 | 0.7587 | 1,598 | 5.9 | 41,043 | 273 µs |
| `KMeans(init="random", max_iter=25)` | 10.5 s | +4.6% | +4.0% | 2,398 | 7.1 | +15% | +19% |
| `MiniBatchKMeans()` | 37 s | +4.2% | +3.7% | 2,202 | 13.3 | +45% | +43% |
| `EMeans(n_epochs=3)` | 2.2 s | +0.3% | +0.4% | 211 | 2.1 | −29% ± 9 | −23% |
| `EMeans(n_epochs=3, balance=0.3)` | 2.2 s | −1.8% | −1.2% | 37 | 1.20 | −73% ± 5 | −71% |
| `EMeans(n_epochs=5, balance=0.3)` | 3.5 s | −2.4% | −1.6% | 16 | 1.16 | −78% ± 1 | −75% |
| `EMeans(n_epochs=3, balance=0.1, balance_end=1.0)` | 2.2 s | −1.5% | −1.0% | 29 | 1.15 | −82% ± 1 | −80% |

On this dataset `EMeans(balance=0.3)` is about 200× faster to fit than scikit-learn's default `KMeans`, ends
with lower error on both the training and the held-out vectors, and gives an index that answers queries 3.5×
faster. This is the favourable case (a clumpy dataset with 61 vectors per cluster).

The same comparison on an ordinary dataset, Fashion-MNIST (60,000 × 784, `k` = 1,024, 3 seeds): `KMeans()` takes
48 s; `EMeans(n_epochs=3, balance=0.3)` takes 0.65 s, has the same held-out error (−0.05%; training error
+0.3%), and its index needs 9.4% ± 1.2 fewer distance computations. Five passes: 1.1 s, −0.24% held-out error,
−10.8% distance computations. On GloVe the switch gains nothing.

## 6. Product quantization

The README test used the paper's batch formula as published. The authors' released code uses one eighth of that
batch size when the data has 16 dimensions or fewer, which is the case for product-quantization sub-vectors
(8 dimensions here). With that rule (16 codebooks × 256 codes, held-out reconstruction error, 3 seeds, change
against Lloyd 25):

| Trained on | Rule | 1 pass | 2 | 3 | 5 | 10 | Lloyd 100 |
|---|---|---|---|---|---|---|---|
| 1M vectors | as published | +0.77% | −0.34% | −0.73% | −0.57% | −1.00% | −1.00% |
| | time | 1.9 s | 1.7 s | 2.4 s | 2.9 s | 5.4 s | 30.9 s (Lloyd 25: 7.1 s) |
| | low-dimension rule | +0.08% | −0.84% | −1.05% | −1.19% | −1.35% | |
| | time | 6.7 s | 7.2 s | 10.6 s | 9.6 s | 19.1 s | |
| 100k vectors | as published | +2.95% | +0.49% | −0.25% | −0.87% | −0.72% | −0.99% |
| | time | 0.7 s | 0.7 s | 1.1 s | 1.1 s | 1.2 s | 6.7 s (Lloyd 25: 1.7 s) |
| | low-dimension rule | +2.02% | −0.18% | −0.65% | −1.12% | −1.48% | |
| | time | 5.1 s | 4.9 s | 7.4 s | 6.6 s | 6.4 s | |

- With the low-dimension rule the result improves with every extra pass, and three passes on 1M vectors beat
  100 Lloyd passes.
- In my GPU code the small batches are slow: 16 codebooks times several hundred tiny updates is mostly
  per-call overhead, so the rule costs 3–6× the time of the large-batch version and is slower than Lloyd 25
  at three passes. On a CPU, or with the 16 codebooks batched into one tensor, that overhead would largely go
  away; I did not build that.
- The size penalty of section 5 does not help here. With the faded penalty the three-pass result is −0.61%
  instead of −1.05%, and five passes give −0.79% instead of −1.19%. In 8 dimensions the best codebooks are not
  equal-sized, so pushing toward balance costs error.

## 7. Gaussian mixtures

The README has the GPU experiment: the recipe applied to EM for diagonal-covariance mixtures matches 40–57
passes of ordinary EM in three. Two additions.

### A scikit-learn-style estimator on CPU

`online_gmm_sklearn.OnlineGaussianMixture` (numpy and scipy only) is a drop-in for
`GaussianMixture(covariance_type="diag")`: `fit`, `predict`, `predict_proba`, `score`, `score_samples`,
`sample`, `bic`, `aic`. Diagonal and spherical covariances; full and tied are refused. No `partial_fit`
(the step curve needs the total number of updates). It passes scikit-learn's estimator checks (89 tests pass,
2 skipped for the array API), gives the same per-row scores as scikit-learn for the same parameters to 1e-9,
and matches the GPU version's held-out score (−80.871 against −80.877 at 1M rows, 256 components).

Against `sklearn.mixture.GaussianMixture` on the same CPU (SIFT in 64 PCA dimensions, held-out log-likelihood
per row, higher is better):

| Data | Method | Passes | Time | Score |
|---|---|---|---|---|
| 1M rows, 256 components | `GaussianMixture()` defaults (1 seed) | 68 | 320 s | −80.862 |
| | `GaussianMixture`, random start | 30 / 65 / 100 | 122 / 252 / 394 s | −80.957 / −80.860 / −80.844 |
| | `OnlineGaussianMixture` (3 seeds) | 3 | 1.3 s | −80.871 |
| | | 10 | 3.8 s | −80.844 |
| 1M rows, 1,024 components | `OnlineGaussianMixture` | 3 / 5 | 4.1 / 6.0 s | −78.207 / −78.172 |
| 100k rows, 256 components | `GaussianMixture()` defaults (3 seeds) | 62 | 26.3 s | −81.808 |
| | `OnlineGaussianMixture` | 3 / 10 / 30 | 0.22 / 0.37 / 0.99 s | −81.988 / −81.864 / −81.811 |
| 100k rows, 64 components | `GaussianMixture()` defaults | 55 | 6.4 s | −84.259 |
| | `OnlineGaussianMixture` | 3 / 10 | 0.20 / 0.40 s | −84.291 / −84.267 |

- At a million rows: about 250× faster than scikit-learn's defaults at three passes for 0.009 less, and about
  100× faster at equal quality (ten passes against 100).
- At 100,000 rows and 256 components (about 400 rows per component) three passes are 0.18 short; it takes about
  30 passes to match, which is still about 26× faster.
- Where it loses: well-separated clusters. On 20,000 points in 12 separated blobs a random start is 3.8 worse
  than scikit-learn and more passes do not fix it; k-means++ seeds do, so the estimator uses them by default
  up to 64 components (that cut-off is a judgement, not a fitted value). On iris, wine and digits three passes
  are 0.06–0.13 worse than scikit-learn. Below about 100,000 rows with few components there is no time to gain.
- The textbook online-EM step rule (Cappé & Moulines 2009, with or without iterate averaging, no revival) was
  0.27 to 5.4 worse than the recipe at 1M rows.
- Timings shared the machine with GPU jobs; trust the ratios. scikit-learn at 1M is one seed.

### Two more old ideas, tested on the GPU version: both negative

`experiments/gmm_lab.py`, three passes, 3 seeds, change in held-out log-likelihood:

| Idea | Source | sift64 | synth | skewed |
|---|---|---|---|---|
| Deterministic annealing (flatter responsibilities early) | Ueda & Nakano 1998 | −0.01 | −0.30 | −0.16 |
| Equal-prior start (ignore mixture weights early) | soft cousin of the size penalty | +0.02 | −0.29 | −0.17 |
| Stricter revival threshold (0.3) | – | −0.03 | −0.28 | −0.66 |
| Batch ÷ 8, step ÷ 2 | settings shift | −0.02 | +0.63 | −0.50 |

Both annealing ideas make the model commit later, which nearly switches revival off (a handful of revivals per run against
166 on synth and 823 on skewed), and revival is what finds the better solutions on the synthetic sets. The size-penalty idea that
helps k-means does not carry over to mixtures in this form. No settings shift is good on all three datasets.

## 8. Topic models (LDA) and NMF

### LDA: the recipe as published does not transfer

Variational LDA already has a well-known online form (Hoffman, Blei & Bach 2010), and it is the recipe's update
with a different step rule: a decayed running total of topic-word counts. So the test is narrower here: do the
cosine step curve, the batch and step formulas, a repair step for dead or duplicate topics, and a final exact
clean-up beat Hoffman's rule? `experiments/lda_torch.py`, validated against scikit-learn's batch LDA (perplexity
equal to 0.0001% over 10 passes from the same start).

UCI New York Times bag of words, 100,000 documents, 8,000-word vocabulary, 100 topics, 3 seeds. Held-out
perplexity by document completion on 10,000 documents; lower is better.

| Method | 1 pass | 2 | 3 | 5 | 10 |
|---|---|---|---|---|---|
| Full-batch variational EM | 3474 | 2503 | 2253 | 2115 | 2071 |
| Recipe, published formulas | 2208 | 2289 | 2295 | 2203 | 2233 |
| Recipe, step ÷ 4 | 2160 | 2108 | 2101 ± 16 | 2078 ± 4 | 2090 |
| Hoffman's online rule at its best published setting (batch 4,096, κ = 0.5, τ0 = 64) | 2146 ± 16 | 2104 | 2098 ± 15 | 2100 ± 15 | 2108 |
| Hoffman's rule + the repair step | 2152 | – | 2087 ± 15 | – | – |

- With the published formulas the step is 4–8× too large for LDA: the result gets worse after the first pass.
- With the step divided by four, the recipe ties Hoffman's rule at 1–3 passes and is about 1% better at 5–10.
  Three online passes (22 s) match 5–7 full-batch passes (52–69 s).
- Full-batch EM is not slow on this problem. It reaches its best value in 10–15 passes and then overfits, so
  there is much less to gain than with k-means.
- The repair step helps a little (10 points at 3 passes, on Hoffman's rule too). The final clean-up does nothing.
- Library defaults run in the same code are far behind Hoffman's best setting: gensim's settings 2259,
  scikit-learn's (batch 128) 2667 at one pass (1 seed each).
- The best batch and step lie on a line of constant memory, batch ÷ step ≈ 33,000 documents (a third of the
  corpus); the formula's choice is about 4,700. No new formula was fitted.

### NMF: it transfers

`experiments/nmf_torch.py`: the same recipe on top of HALS updates from decayed Gram statistics (the running
totals of Mairal et al. 2010). TF-IDF of the same 100,000 documents, 100 components, CPU, 3 seeds; held-out
relative reconstruction error.

| Method | Error | Time |
|---|---|---|
| Recipe, published formulas, 1 pass | 0.80698 | 3.5 s |
| Recipe, published formulas, 3 passes | 0.80398 | 7.3 s |
| Full-batch HALS, 30 / 100 passes | 0.80443 / 0.80369 | 32 s / 107 s |
| scikit-learn `NMF()` (1 seed) | 0.80314 | 193 s |
| scikit-learn `MiniBatchNMF()` | 0.81938 | 19 s |
| Textbook running average (batch 512), 3 passes | 0.80827 | 31 s |

Three passes match 34–79 full passes and come within 0.1% of scikit-learn's converged `NMF` in about 4% of its
time; scikit-learn's own mini-batch version is 2% worse and slower. Revival never triggered here and the step
curve made no difference: the gain comes from large batches with fast forgetting.

## 9. Hidden Markov models

`experiments/hmm_torch.py`: a categorical hidden Markov model trained by Baum-Welch (full passes) or by the
recipe (mini-batches of 256-character chunks, decayed expected counts, revival by splitting a state, last-pass
clean-up). The full-pass code matches `hmmlearn` to 9e-16 from the same start.

Data: text8 (English Wikipedia as 27 symbols), 20M characters for training, 5M held out. Score: held-out bits
per character, lower is better. For scale, single-letter frequencies give 4.125 and a previous-letter model 3.438.

| Method | 64 states (3 seeds) | Time | 256 states (2 seeds) | Time |
|---|---|---|---|---|
| Baum-Welch, 20 passes | 3.196 | 46 s | 2.927 | 150 s |
| Baum-Welch, 60 passes | 3.037 | 137 s | 2.589 | 452 s |
| Baum-Welch, 100 passes | 3.015 | 228 s | – | – |
| Recipe, formulas fed the number of chunks: 3 passes | 3.096 | 6.7 s | 2.864 | 21.7 s |
| Same, 10 passes | 3.048 | 21 s | 2.610 | 70 s |
| Recipe, formulas fed the number of characters: 1 pass | 3.027 | 18.8 s | 2.646 | 11.6 s |
| Same, 3 passes | 2.982 | 19.9 s | 2.532 | 29.3 s |

- The recipe transfers once the formulas are given the number of symbols rather than the number of sequences,
  which makes the batches small (64–256 chunks). Then one pass equals 62–80 Baum-Welch passes at 64 states and
  about 40 at 256 states, for roughly 8–26× less time.
- Three such passes were not matched by Baum-Welch within 100 passes (64 states) or 60 passes (256 states). One
  seed continued to 200 passes reached 2.993, against 2.976 for the recipe.
- Fed the number of chunks, as a literal reading would have it, the batch is about a million characters and
  three passes are worth only 22–33 Baum-Welch passes.
- On a synthetic set with a known 64-state model (true score 2.880) the two end close: Baum-Welch 2.969 after
  150 passes (43 s), the recipe 2.950 after 10 passes (22.5 s).
- Which ingredients matter is different here. Many small updates do the work. The cosine curve does not help (a
  constant step is equal or better), revival is worth under 0.01 bits and the clean-up nothing.
- The textbook online step rule at its best setting, with revival, reached 3.038 at 64 states: 0.06 worse than
  the recipe at the same batch size.
- Growing the number of states by splitting (start at a quarter, double twice) was worse than a fixed size in
  every small test and was dropped.
- Small batches are slow on this GPU kernel (3–20× more per pass), which is why one small-batch pass takes as
  long as three large-batch ones. The 256-state numbers are 2 seeds, and Baum-Welch there was still improving
  at 60 passes.

## 10. Recommender factorization (ALS)

`experiments/als_torch.py`: alternating least squares for explicit ratings, and an online version in the
recipe's shape: mini-batches of users solved exactly and never stored (like assignments in k-means), decayed
per-item normal-equation totals, items re-solved each batch, cosine step, last-pass clean-up. Full ALS matches
an independent numpy implementation to 4e-8.

MovieLens-20M, rank 64, 3 seeds, 10% of ratings held out; held-out RMSE, lower is better. The ridge strength was
tuned for full ALS and used for both.

| Method | Passes | RMSE | Time |
|---|---|---|---|
| ALS | 10 / 20 / 40 sweeps | 0.7755 / 0.7737 / 0.7729 | 11 / 22 / 45 s |
| Online, settings from the formulas | 1 / 3 / 10 | 0.7817 / 0.7770 / 0.7752 | no gain: a pass costs about as many sweeps as it is worth |
| Online, batch 4,096 users, step 1.0 | 1 | 0.7735 | 5.3 s |
| | 2 / 3 / 10 | 0.7724 / 0.7723 / 0.7718 | 14 s for 3 |

- With the paper's formulas there is nothing to gain; the formula's step is far too small for this problem.
- With a batch of about 3% of the users and a step of 1, one pass reaches in about 5 s what ALS reaches in
  about 27 s, and from two passes on the held-out error is below anything ALS reached in 40 sweeps.
- That second gain is regularization, not better optimization: the online solution fits the training ratings
  worse than ALS (0.6061 against 0.6034) and predicts held-out ratings better. Raising the ridge strength for
  ALS closes most of the gap; on MovieLens-1M with each method at its own best ridge the online version still
  led by 0.0013–0.0018.
- Ingredients: the last-pass clean-up matters (+0.0011 without it), the cosine curve does not at this batch
  size, and no repair step was needed (no dead or exploding factors).
- An online pass costs 3–5 ALS sweeps of time in this implementation. No biases, no implicit feedback; the
  block-update method iALS++ was not built for comparison.

## 11. PageRank

A short CPU study on the web-Google graph (875,713 pages, 5.1M links, damping 0.85), sweeps and seconds to a
residual of 1e-6 / 1e-10 (`experiments/pagerank_sweeps.py`):

| Method | Sweeps | Seconds |
|---|---|---|
| Power iteration (update everything from the previous sweep) | 65 / 119 | 0.74 / 1.35 |
| Gauss-Seidel (use each new value at once) | 34 / 62 | 5.4 / 9.9 (slow only because of scipy's triangular solve) |
| 256 blocks, fresh values between blocks | 34 / 80 | 0.38 / 0.90 |
| Fading running total (step 0.5) | 135 / 249 | 2.0 / 3.6 |
| Cosine step 1 → 0.1 over 150 sweeps | 80 / not reached | 1.1 / – |

Only one of the recipe's ingredients means anything here: update in pieces and use fresh values, which is the
known Gauss-Seidel idea and halves the sweeps. PageRank is a deterministic fixed point with no noise to average
away, so forgetting, a step curve, timed repair and a final clean-up either do nothing or slow it down.

## 12. My implementation against the authors' code

The authors' code is public (`contrib/e_means` in the Faiss repository). The Faiss release installed here
(1.16.0) does not ship it, and I had written mine from the paper. I read theirs (did not run it) and compared.
The main recipe matches: cosine step to 10% indexed by batch, revival every 5 updates below 0.1 × mean,
donors drawn in proportion to mass without replacement, mass halved, alternating-sign split, last-pass mean
from assignments recorded while the centroids move, one permutation per pass.

| Difference | Theirs | Mine | Effect |
|---|---|---|---|
| Step equal to 1 | no decay; only clusters the batch touched are rewritten | decayed everything to zero, giving 0/0 | broke my 15- and 25-pass runs; **fixed**, see section 2 |
| 16 dimensions or fewer | batch size ÷ 8; revival off for 2 dimensions or fewer | no rule | my first product-quantization test used 8× their batch; rerun in section 6; batch rule now in `EMeans` and `common.laws(d=...)` |
| More than 2M vectors | no size rule in the shipped code | batch ÷ 8 (from the paper's text) | none so far (all runs ≤ 1.1M) |
| Tail of each pass | kept as a smaller batch | dropped (up to one batch of vectors per pass unused) | small |
| Split size, initial mass, formula constants | ±1e-4, 2⁻¹⁶, full precision | ±2⁻⁷, 1e-10, rounded | negligible (batch and step agree within 1%) |
| Precision | BF16 on CUDA | FP32 | paper reports ≤ 0.13% |

I have not run their code, so "matches" means the same algorithm on paper and the same headline numbers as
their Table 5 for SIFT1M (−0.35% reported, −0.37% measured), not a bit-for-bit comparison.

## 13. Caveats

- One machine, one GPU type. GPU timings are training only and exclude data loading and scoring; CPU timings
  for the mixture estimator and the scikit-learn comparison shared the machine with GPU jobs.
- Seeds: 2–5 per configuration for the search experiments, 3 for most others, 1 for some large runs (marked in
  the tables). Where the spread is not shown the number is one seed.
- Search cost is exact-distance scanning inside the probed lists (IVF-Flat). No compressed vectors (IVF-PQ),
  no graph indexes, no re-ranking. Queries are held-out vectors from the same dataset, not real query logs.
- The largest dataset is 1.1M vectors; `k` goes up to 16,384. Nothing here is at the paper's 100M–10B scale.
- The size penalty was tuned on SIFT and then tried on five other datasets. It helped on three, was mixed on one
  (text embeddings: nothing at one `k`, 10% at another) and hurt slightly on one (GloVe). Six datasets do not
  tell you which case yours is; measure it.
- The strength rule in section 5 is a two-factor model fitted per dataset. It was right on three datasets and
  wrong on one, and it needs pilot runs, so it is an explanation more than a tool.
- Prior art for each idea comes from literature searches made the same night. Where I write that something
  was not found, that is the search coming up empty, not a claim of novelty.
- I read the authors' code and did not run it.
- The mixture, topic-model, hidden-Markov and factorization experiments each use one or two datasets.

