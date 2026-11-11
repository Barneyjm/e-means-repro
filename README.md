# e-means-repro

An independent check of the paper **"e-means: outperforming k-means (in one-tenth the epochs)"**
(Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze, Jégou; [arXiv 2611.09501](https://arxiv.org/abs/2611.09501), Nov 2026),
plus a version of the algorithm that drops into scikit-learn, and follow-up experiments on whether the same recipe
helps search indexes, product quantization and Gaussian mixtures.

Everything here was written from the text of the paper. It is **not the authors' code**. The paper says theirs lives in
Faiss; the Faiss release used here (1.16.0) does not include it.

## What the paper claims

k-means groups vectors into `k` clusters. The standard method (Lloyd's algorithm) passes over the whole dataset about
25 times. The paper says a mini-batch variant with a decaying running average, a cosine step schedule and a rule that
re-seeds near-empty clusters reaches a *better* result in 2–3 passes.

## What I measured

SIFT1M (1,000,000 vectors, 128 dimensions), `k` = 16,384, random initialization everywhere.
"MSE" is the mean squared distance from a vector to its nearest centre; lower is better.
"Test" uses the 10,000 SIFT query vectors, which the clustering never sees.

### GPU (PyTorch on an Apple M2 Max), 3 seeds

Same distance kernel for both methods. Time is training only.

| Method | Passes | Time | Speedup | Test MSE vs Lloyd | Train MSE vs Lloyd | Clusters with < 10 points |
|---|---|---|---|---|---|---|
| Lloyd | 25 | 18.46 s | – | 41,711 | 38,260 | 182 |
| e-means | 1 | 0.83 s | 22× | +0.80% | +3.51% | 257 |
| e-means | 2 | 1.62 s | 11.4× | −0.09% | +0.39% | 53 |
| e-means | 3 | 2.33 s | 7.9× | −0.37% | −0.26% | 40 |
| e-means | 5 | 3.86 s | 4.8× | −0.44% | −1.03% | 21 |

The paper reports −0.35% at 3 passes for this dataset and `k` (its Table 5, against Flash-KMeans at 25 iterations).
I measured −0.37% against my own Lloyd baseline. The paper says 1.4–2.6 passes are enough to match 25 Lloyd
iterations; here it took 2 passes on test error and 3 on training error.

Faiss's own CPU k-means, run as an outside reference for the baseline (25 iterations, no subsampling, seed 0),
gave test 41,704 and train 38,256, which is within 0.02% of the Lloyd row above.

With re-seeding of near-empty clusters switched off (100,000 vectors, `k` = 1,024), 182 clusters stayed empty and the
error was about 2% worse, which matches the direction and rough size of the paper's ablation.

### CPU (numpy + scipy + scikit-learn only), seed 0

`emeans_sklearn.EMeans` against scikit-learn 1.9.1 on the same machine (12 cores).

| Data | Method | Time | Test MSE | Train MSE |
|---|---|---|---|---|
| 1M, k = 16,384 | `KMeans(init="random", max_iter=25)` | 112.7 s | 41,658 | 38,254 |
| | `EMeans(n_epochs=2)` | 9.7 s | 41,661 (+0.01%) | 38,414 (+0.42%) |
| | `EMeans(n_epochs=3)` | 12.5 s | 41,581 (−0.19%) | 38,148 (−0.28%) |
| 1M, k = 4,096 | `KMeans(init="random", max_iter=25)` | 26.1 s | 45,626 | 44,389 |
| | `MiniBatchKMeans()` | 42.0 s | 46,082 (+1.00%) | 45,335 (+2.13%) |
| | `EMeans(n_epochs=3)` | 4.2 s | 45,472 (−0.34%) | 44,375 (−0.03%) |
| 100k, k = 1,024 (3 seeds) | `KMeans(init="random", max_iter=25)` | 0.81 s | 49,409 | 49,525 |
| | `KMeans()` (k-means++, to convergence) | 18.5 s | 49,293 (−0.24%) | 49,402 (−0.25%) |
| | `MiniBatchKMeans()` | 1.41 s | 50,872 (+2.96%) | 50,944 (+2.87%) |
| | `EMeans(n_epochs=3)` | 0.40 s | 49,313 (−0.19%) | 49,450 (−0.15%) |

So in plain numpy, three passes of e-means are 6–9× faster than scikit-learn's 25-iteration `KMeans` at a million
vectors and end slightly better. scikit-learn's default k-means++ run to convergence is still a little better at
100k, at 46× the time.

**Small data is a different story.** On `load_digits` (1,797 × 64, k = 10) and 2-D blobs (10,000 points, k = 8),
10 seeds each, there is nothing to gain: everything finishes in about 10 ms, and `EMeans` with random initialization
lands in a poor local minimum about as often as `KMeans` does (mean MSE 2.03 vs 1.90 on the blobs, 658 vs 658 on
digits). The paper says its batch-size and step-size formulas are not tuned for tiny or low-dimensional data. Use
this for large `n` and large `k`.

## Beyond the paper: does the recipe carry over?

The paper only reports clustering error. These are my own follow-up experiments, not claims from the paper. Raw logs
and full tables are in `results/beyond/`. All runs: Apple M2 Max GPU, random initialization, SIFT1M unless stated.

### 1. Does a search index built on the centroids get better?

The usual reason to run k-means at this scale is a vector-search index (IVF): each database vector is filed under its
nearest centroid, and a query scans only the nearest few lists. What matters is how many vectors you must scan to
reach a given recall. I filed the 1M base vectors under each set of centroids and searched with 100,000 held-out
vectors (the SIFT learn set, with exact neighbours recomputed). 5 seeds; change is against Lloyd 25, paired by seed.

| k | Method | Train time | Test MSE | Vectors scanned for 95% recall@1 | for 95% recall@10 |
|---|---|---|---|---|---|
| 1,024 | Lloyd 25 | 1.41 s | 50,765 | 14,870 | 19,682 |
| | Lloyd 10 | 0.57 s | +0.42% | +3.6% ± 0.8 | +3.7% ± 0.4 |
| | e-means 1 pass | 0.14 s | +0.49% | +4.9% ± 1.3 | +4.2% ± 0.8 |
| | e-means 3 passes | 0.30 s | −0.09% | −0.1% ± 1.2 | −0.2% ± 0.3 |
| | e-means 5 passes | 0.56 s | −0.12% | −0.9% ± 0.5 | −1.1% ± 0.6 |
| 4,096 | Lloyd 25 | 4.86 s | 45,535 | 7,691 | 11,049 |
| | Lloyd 10 | 1.95 s | +0.35% | +2.5% ± 1.2 | +3.4% ± 0.3 |
| | e-means 1 pass | 0.29 s | +0.62% | +6.7% ± 1.2 | +5.9% ± 0.6 |
| | e-means 3 passes | 0.68 s | −0.06% | +0.4% ± 0.9 | +0.6% ± 0.5 |
| | e-means 5 passes | 1.13 s | −0.19% | −1.3% ± 1.3 | −1.2% ± 0.4 |
| 16,384 | Lloyd 25 | 18.47 s | 41,694 | 3,945 | 6,144 |
| | Lloyd 10 | 7.40 s | +0.11% | +3.0% ± 0.7 | +2.5% ± 0.2 |
| | e-means 1 pass | 0.82 s | +0.82% | +11.7% ± 2.0 | +10.7% ± 0.4 |
| | e-means 3 passes | 2.33 s | −0.32% | −3.6% ± 1.3 | −3.0% ± 0.4 |
| | e-means 5 passes | 3.86 s | −0.35% | −6.0% ± 0.6 | −5.3% ± 0.5 |

- Three passes give an index as good as 25 Lloyd iterations at `k` = 1,024 and 4,096, and one that scans 3–4% fewer
  vectors at `k` = 16,384 (6% fewer at five passes), for 5–8× less training time.
- One pass is not enough: the index scans 5–12% more.
- A clustering-error gap of a few tenths of a percent turns into a scan-cost gap about ten times larger.
- Lloyd with 10 iterations (the Faiss default for index training) scans about 3% more than Lloyd 25 and takes
  longer than three e-means passes.
- Check on the method: for one set of centroids a real `faiss.IndexIVFFlat` returned the same recall and the same
  number of distance computations as my calculation, at every setting tried (`experiments/faiss_ivf_check.py`).

### 2. Product-quantization codebooks (small `k`, low dimension)

16 sub-vectors of 8 dimensions, 256 codes each. This is outside the range the paper's formulas were fitted on.
Error is the held-out reconstruction error summed over the 16 codebooks; 3 seeds.

| Trained on | Lloyd 10 | Lloyd 25 | Lloyd 100 | e-means 1 | e-means 2 | e-means 3 | e-means 5 | e-means 10 |
|---|---|---|---|---|---|---|---|---|
| 1M vectors | +2.02% | 10,555 | −1.00% | +0.77% | −0.34% | −0.73% | −0.57% | −1.00% |
| time | 2.7 s | 7.1 s | 30.9 s | 1.9 s | 1.7 s | 2.4 s | 2.9 s | 5.4 s |
| 100k vectors | +2.11% | 10,441 | −0.99% | +2.95% | +0.49% | −0.25% | −0.87% | −0.72% |
| time | 0.70 s | 1.70 s | 6.72 s | 0.72 s | 0.73 s | 1.06 s | 1.05 s | 1.23 s |

Three passes beat Lloyd 25 here too (seed spread is under 0.1%), and ten passes equal Lloyd 100 on the 1M set.
The result is not monotonic in the number of passes (5 is worse than 3 on 1M), because the formula changes the
batch size with the pass count. Timings at this size are dominated by per-call overhead.

### 3. The same recipe for Gaussian mixtures (EM)

k-means is the hard-assignment special case of fitting a Gaussian mixture. I applied the same five ingredients to
EM for diagonal-covariance mixtures (`experiments/gmm_torch.py`): per-batch updates of decayed totals (weight, sum,
sum of squares), the cosine step curve, revival of starved components every 5 batches, a final clean-up from the
plain totals of the last pass, and the paper's batch-size and step-size formulas unchanged.

Score: mean log-likelihood per held-out vector (higher is better). 1M training vectors, 64 dimensions, 3 seeds.
`sift64` is SIFT reduced to 64 whitened PCA dimensions. `synth` and `skewed` are drawn from known 256-component
mixtures (near-equal and very unequal component weights), so the best possible score is known.
My full-pass EM matches scikit-learn's `GaussianMixture` to 5 decimals from the same start.

| Data | Best possible | EM 25 passes | EM 100 | EM 200 | Online 1 pass | Online 3 | Online 10 | EM passes needed to match online 3 |
|---|---|---|---|---|---|---|---|---|
| sift64, K = 256 | – | −80.971 | −80.838 | −80.820 | −81.227 | −80.877 | −80.838 | 50–57 |
| time | | 1.95 s | 7.80 s | 15.6 s | 0.13 s | 0.31 s | 0.96 s | |
| sift64, K = 1,024 | – | −78.270 | −78.130 | −78.110 | −78.699 | −78.196 | −78.181 | 38–44 |
| time | | 8.06 s | 32.1 s | 64.2 s | 0.45 s | 1.11 s | 3.17 s | |
| synth, K = 256 | −87.780 | −89.240 | −89.183 | −89.147 | −89.659 | −88.951 | −88.869 | not reached in 200 |
| skewed, K = 256 | −86.758 | −87.454 | −87.431 | −87.429 | −88.118 | −87.313 | −87.255 | not reached in 200 |

- On SIFT, three online passes equal 40–57 passes of ordinary EM, about 12× less time for the same quality.
  With many more passes ordinary EM ends slightly ahead (by 0.04–0.07 at 100 passes).
- On both synthetic sets the online version ends at a *better* solution than ordinary EM reaches in 200 passes,
  because revival moves components out of crowded regions. Neither gets close to the true model.
- One pass is not enough, as with k-means.

Which ingredients matter (3 passes; change in held-out log-likelihood when one thing is altered):

| Change | sift64 K=256 | sift64 K=1,024 | synth | skewed |
|---|---|---|---|---|
| Constant step instead of cosine | −0.065 | −0.399 | +0.135 | −0.204 |
| No revival | −0.101 | −0.518 | −0.460 | −0.198 |
| Paper's tiny split (±2⁻⁷) instead of ±0.5 standard deviations | +0.005 | +0.004 | −0.252 | −0.152 |
| No last-pass clean-up | −0.004 | −0.003 | −0.030 | −0.509 |
| Revival threshold 1% instead of 10% | −0.001 | −0.005 | −0.149 | −0.073 |
| Textbook step rule `(t+2)^−κ`, best κ, no revival | −0.218 | −0.409 | −0.377 | −0.164 |

Revival is the ingredient that matters everywhere. The cosine curve helps in three of four cases. The last-pass
clean-up matters only when component weights are very unequal.

Three further checks:

- **The paper's formulas, fitted for k-means, are close to the best settings but not on them.** A grid on 100k
  vectors (`results/beyond/gmm_grid_100k.md`) shows a ridge where batch size and step size shrink together.
  A 4× smaller batch with a 4× smaller step, tested on 1M, was better at 2–10 passes for K = 1,024 (+0.02) and
  on synth at 3 passes (+0.09), mixed for sift64 K = 256 (+0.03 to −0.02), and 10–80% slower. I did not fit a new
  formula.
- **The textbook online-EM step rule is not the main problem; the missing revival is.** Given small batches the
  textbook rule without revival is far worse (−0.2 to −3.0 on sift64, −0.4 to −4.4 on synth). With revival added
  and its best batch size it comes within 0.04 of the cosine recipe on sift64 at 2–4× the time, and beats the
  default recipe on synth (+0.17 at 5× the time).
- **scikit-learn as people run it** (`GaussianMixture`, CPU, k-means start, to convergence; one seed, different
  hardware and precision, so the times are not a like-for-like speed comparison): sift64 K = 256 −80.871 after 65
  passes and 526 s; synth −89.292 (160 s); skewed −87.514 (163 s). Three online passes from a random start score
  −80.877, −88.951 and −87.313.

## What I did not test

- Billion-vector runs, and the approximate (IVF) assignment the paper uses there.
- The paper's GPU comparison against Flash-KMeans, or any k-means++ baseline at 1M.
- Datasets other than SIFT.
- The CPU numbers at 1M are one seed each.
- Search-index quality on anything but SIFT1M (no modern text or image embeddings), and no end-to-end index with
  compressed vectors.
- Mixtures with full covariance matrices, a downstream task that uses the mixture, or more than 3 seeds.
- Ordinary EM from a k-means start on the GPU (the scikit-learn row above is the only such run).

## Use it

```python
from emeans_sklearn import EMeans

km = EMeans(n_clusters=4096, n_epochs=3, random_state=0).fit(X)   # X: (n_samples, n_features) float32 or float64
km.cluster_centers_, km.labels_, km.inertia_
km.predict(X_new); km.transform(X_new); km.score(X_new)
```

`pip install -e .` installs it (numpy, scipy, scikit-learn ≥ 1.6). It follows the scikit-learn estimator
contract: it works in `Pipeline`, `clone`, `GridSearchCV` and passes scikit-learn's `check_estimator` suite
(50 checks pass, 1 skipped; `pytest test_emeans_sklearn.py`). Batch size and step size default to the paper's
scaling laws and can be overridden. Dense input only.

How it is built from standard parts:

- **Assignment** (nearest centre): BLAS matrix multiply in cache-sized chunks, run across a thread pool. numpy
  releases the GIL in the multiply, subtract and argmax, so this scales across cores without compiled code.
- **Per-cluster sums**: a `scipy.sparse` one-hot matrix times the batch.
- **Everything else** (running averages, cosine schedule, re-seeding, final averaging) is a few lines of numpy.

## The algorithm in brief

For each cluster keep a decayed count `N` and a decayed sum `S`; the centre is `S / N`. For each mini-batch:

1. Assign each vector in the batch to its nearest centre.
2. `N ← (1 − α)·N + α·(batch count)`, `S ← (1 − α)·S + α·(batch sum)`.
3. `α` follows a cosine from `α0` down to `0.1·α0` over the whole run.
4. Every 5 batches, any cluster whose `N` is below 10% of the average takes half of a large cluster, picked with
   probability proportional to `N`. The two halves are nudged apart by ±2⁻⁷.
5. At the end, each centre is replaced by the plain average of the vectors it won during the last pass.

Batch size and `α0` come from the paper's Table 2:
`B = 0.035·k^0.13·n^0.88·T^0.81` (÷8 above 2M vectors, rounded to a power of two) and
`α0 = 0.824·k^−0.46·B^0.90·n^−0.45·T^−0.11`.

## Reproduce

```sh
pip install -e ".[experiments,test]"
sh experiments/get_data.sh                                        # SIFT1M into data/sift (about 160 MB)
python experiments/run.py data/sift results/my_gpu.log 16384 0,1,2     # GPU table
python experiments/bench_sklearn.py data/sift results/my_cpu.log all   # CPU table
python experiments/summarize.py results/my_cpu.log
python experiments/faiss_baseline.py data/sift faiss_centroids.npy 16384   # about 13 minutes on 12 cores
python experiments/score_centroids.py data/sift faiss_centroids.npy
```

The follow-up experiments (`D=data/sift`):

```sh
python experiments/exact_neighbours.py $D learn_gt10.npy                       # exact neighbours of the 100k learn vectors
python experiments/ivf_recall.py $D ivf.log 16384 0,1,2,3,4 learn_gt10.npy      # search-index table (also 1024, 4096)
python experiments/summarize_ivf.py ivf.log
python experiments/pq_codebooks.py $D pq.log base 0,1,2                        # product quantization
python experiments/run_gmm.py $D gmm.log sift64 256 0,1,2 all                  # mixtures: sift64 | synth | skewed
python experiments/summarize_gmm.py gmm.log
python experiments/gmm_grid.py $D grid.log sift64 256 100000 0,1 grid          # settings grid; also: classic, rule
python experiments/gmm_sklearn_ref.py $D sk.log sift64 256 0                   # about 18 minutes
```

Faiss and PyTorch each ship their own OpenMP runtime and cannot share a process, which is why the Faiss baseline and
its scoring are two commands.

| File | What it is |
|---|---|
| `emeans_sklearn.py` | The scikit-learn estimator |
| `experiments/emeans_torch.py` | GPU version used for the first table |
| `experiments/emeans_v1.py` | First, unoptimized version (kept so the validation run can be repeated) |
| `experiments/bench_kernels.py` | Distance-kernel timings behind the optimized version |
| `experiments/ivf_recall.py`, `faiss_ivf_check.py` | Search-index evaluation and its check against a real Faiss index |
| `experiments/gmm_torch.py` | Full-pass EM and the online version for diagonal Gaussian mixtures |
| `experiments/pq_codebooks.py` | Product-quantization test |
| `results/` | Raw logs for every number above (`results/beyond/` for the follow-up experiments) |

## Notes

- Hardware: Apple M2 Max, 64 GB, 12 cores; PyTorch 2.15 (MPS), faiss-cpu 1.16.0, numpy 2.5, scipy 1.18.
- The paper's MSE values are on a different scale from mine, so only relative gaps are compared.
- A fused `addmm` distance kernel made the GPU code 3.8× faster than the broadcast form, with identical
  assignments. Half precision was another 13% faster but gave worse test error at 3 and 5 passes, so it is off.
- `results/v1_k16384.log` times include scoring passes; use `results/optimized_k16384.log` for timings.

## Credits and license

The method is the paper authors'. The paper is CC BY 4.0. SIFT1M is from the
[TEXMEX corpus](http://corpus-texmex.irisa.fr/) (Jégou, Douze, Schmid) and is not redistributed here.
Code in this repository: MIT.
