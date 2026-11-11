"""Usage: python bench_sklearn.py <data dir> <log file> [big|small|all]
CPU-only comparison of EMeans (numpy/scipy) with scikit-learn's KMeans and MiniBatchKMeans.
MSE = mean squared distance to the nearest centre (inertia / n). Run from the repo root or with it on PYTHONPATH."""
import sys, os, json, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import load_digits, make_blobs
from emeans_sklearn import EMeans

D, out = sys.argv[1], sys.argv[2]; which = sys.argv[3] if len(sys.argv) > 3 else "all"
f = open(out, "a")
def log(**kw): s = json.dumps(kw); print(s, flush=True); f.write(s + "\n"); f.flush()
def fvecs(p):
    a = np.fromfile(p, dtype=np.int32); d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()
def fit(name, est, X, Xte=None, **extra):
    t = time.time(); est.fit(X); dt = time.time() - t
    r = dict(method=name, time=round(dt, 2), train_mse=round(est.inertia_ / len(X), 4), **extra)
    if Xte is not None: r["test_mse"] = round(-est.score(Xte) / len(Xte), 4)
    if hasattr(est, "n_iter_"): r["n_iter"] = int(est.n_iter_)
    log(**r)

if which in ("big", "all"):
    Xte = fvecs(D + "/sift_query.fvecs"); learn = fvecs(D + "/sift_learn.fvecs"); base = fvecs(D + "/sift_base.fvecs")
    for cfg, X, k, seeds, slow in (("sift100k_k1024", learn, 1024, (0, 1, 2), True), ("sift1M_k4096", base, 4096, (0,), False),
                                   ("sift1M_k16384", base, 16384, (0,), False)):
        for s in seeds:
            kw = dict(cfg=cfg, seed=s)
            fit("EMeans 2 epochs", EMeans(k, n_epochs=2, random_state=s), X, Xte, **kw)
            fit("EMeans 3 epochs", EMeans(k, random_state=s), X, Xte, **kw)
            fit("MiniBatchKMeans defaults", MiniBatchKMeans(k, random_state=s), X, Xte, **kw)
            fit("KMeans random init, 25 iters", KMeans(k, init="random", n_init=1, max_iter=25, tol=0, random_state=s), X, Xte, **kw)
            if slow: fit("KMeans defaults (k-means++)", KMeans(k, random_state=s), X, Xte, **kw)

if which in ("small", "all"):
    # the paper says its scaling laws are not tuned for tiny or low-dimensional data; check that honestly
    blobs = make_blobs(n_samples=10000, centers=8, n_features=2, random_state=0)[0]
    for cfg, X, k in (("digits_1797x64_k10", load_digits().data, 10), ("blobs_10000x2_k8", blobs, 8)):
        for s in range(10):
            kw = dict(cfg=cfg, seed=s)
            fit("EMeans 3 epochs", EMeans(k, random_state=s), X, **kw)
            fit("EMeans 10 epochs", EMeans(k, n_epochs=10, random_state=s), X, **kw)
            fit("KMeans defaults (k-means++)", KMeans(k, random_state=s), X, **kw)
