# e-means-repro

An independent check of the paper **"e-means: outperforming k-means (in one-tenth the epochs)"**
(Bessa, Mazaré, Szilvasy, Touvron, Picard, Douze, Jégou; [arXiv 2611.09501](https://arxiv.org/abs/2611.09501), Nov 2026),
plus a version of the algorithm that drops into scikit-learn.

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

## What I did not test

- Billion-vector runs, and the approximate (IVF) assignment the paper uses there.
- The paper's GPU comparison against Flash-KMeans, or any k-means++ baseline at 1M.
- Datasets other than SIFT.
- The CPU numbers at 1M are one seed each.

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

Faiss and PyTorch each ship their own OpenMP runtime and cannot share a process, which is why the Faiss baseline and
its scoring are two commands.

| File | What it is |
|---|---|
| `emeans_sklearn.py` | The scikit-learn estimator |
| `experiments/emeans_torch.py` | GPU version used for the first table |
| `experiments/emeans_v1.py` | First, unoptimized version (kept so the validation run can be repeated) |
| `experiments/bench_kernels.py` | Distance-kernel timings behind the optimized version |
| `results/` | Raw logs for every number above |

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
