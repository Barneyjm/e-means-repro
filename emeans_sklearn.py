"""EMeans: e-means clustering (arXiv 2611.09501) as a scikit-learn estimator.

Needs only numpy, scipy and scikit-learn. An independent re-implementation written from the paper;
the authors' own code lives in Faiss.
"""
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import sparse
from joblib import cpu_count
from sklearn.base import BaseEstimator, ClassNamePrefixFeaturesOutMixin, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

__all__ = ["EMeans", "batch_law", "step_law"]


def batch_law(n, k, T, d=None):
    """Batch size from the paper's Table 2 (Eq. 7): nearest power of two, capped at n.

    One eighth of that above 2M samples (paper) and, when ``d`` is given, for 16 dimensions or fewer
    (the rule in the authors' released code).
    """
    B = 0.035 * k**0.13 * n**0.88 * T**0.81
    if n > 2_000_000 or (d is not None and d <= 16):
        B /= 8
    return int(min(max(2 ** round(math.log2(max(B, 1.0))), 1), n))


def step_law(n, k, T, B):
    """Peak step size from the paper's Table 2 (Eq. 6), capped at 1."""
    return min(0.824 * k**-0.46 * B**0.90 * n**-0.45 * T**-0.11, 1.0)


def _assign(X, C, pool=None, return_dist=False, chunk_bytes=2**25, bias=None):
    """Nearest centre for each row of X; optionally the squared distance to it.

    With ``bias`` (one value per centre) the choice minimises squared distance + bias; the distance returned
    is still the plain squared distance to the chosen centre.

    argmin |x - c|^2 = argmax (x.c - |c|^2 / 2). The work is cut into chunks small enough to stay in cache;
    numpy releases the GIL in matmul, the subtraction and argmax, so a thread pool runs the chunks in parallel.
    """
    half_sq = 0.5 * np.einsum("ij,ij->i", C, C)
    if bias is not None:
        half_sq = half_sq + 0.5 * bias.astype(C.dtype, copy=False)
    Ct = np.ascontiguousarray(C.T)
    n = X.shape[0]
    rows = int(min(max(chunk_bytes // (C.dtype.itemsize * C.shape[0]), 64), 4096))
    labels = np.empty(n, dtype=np.intp)
    dist = np.empty(n, dtype=X.dtype) if return_dist else None

    def job(i):
        xc = X[i:i + rows]
        G = xc @ Ct
        G -= half_sq
        lab = G.argmax(axis=1)
        labels[i:i + rows] = lab
        if return_dist:
            dist[i:i + rows] = np.einsum("ij,ij->i", xc, xc) - 2 * G[np.arange(len(lab)), lab]
            if bias is not None:
                dist[i:i + rows] -= bias[lab]

    starts = range(0, n, rows)
    if pool is None or len(starts) < 2:
        for i in starts:
            job(i)
    else:
        list(pool.map(job, starts))
    if return_dist:
        np.maximum(dist, 0, out=dist)
    return (labels, dist) if return_dist else labels


def _cluster_sums(X, labels, k):
    """Per-cluster counts and sums of the rows of X."""
    m = len(labels)
    onehot = sparse.csr_matrix((np.ones(m, dtype=X.dtype), (labels, np.arange(m))), shape=(k, m))
    return np.bincount(labels, minlength=k).astype(X.dtype), np.asarray(onehot @ X)


class EMeans(ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator):
    """Mini-batch k-means with exponential-moving-average updates ("e-means").

    Parameters
    ----------
    n_clusters : int, default=8
    n_epochs : int, default=3
        Passes over the data. The paper reports 2-3 epochs match 25 Lloyd iterations.
    batch_size : int or "auto", default="auto"
        "auto" uses the paper's batch scaling law (one eighth of it for 16 features or fewer, as in the
        authors' code).
    step_size : float or "auto", default="auto"
        Peak EMA step (1 - decay). "auto" uses the paper's step scaling law.
    init : "random" or array of shape (n_clusters, n_features), default="random"
    revive : bool, default=True
        Rebuild near-empty clusters every 5 batches by splitting a large donor cluster.
    balance : float, default=0.0
        Strength of a cluster-size penalty during training (not in the paper). A sample goes to the centre
        minimising ``squared distance + balance * D * (relative size - 1)``, where ``D`` is the running mean
        squared distance and relative size is the cluster's running share times ``n_clusters``. This is
        DeSieno's "conscience" (1988), the size penalty of balanced k-means. Around 0.3 gave more even
        clusters and cheaper inverted-file search on image descriptors; see EXTENSIONS.md for where it did not help.
    balance_end : float in [0, 1], default=0.0
        The penalty follows the cosine curve from ``balance`` down to ``balance_end * balance``.
        0 (default) fades it out, so training ends as plain k-means; 1 holds it to the end.
    compute_labels : bool, default=True
        Run one more assignment pass after training to set ``labels_`` and ``inertia_``.
    n_threads : int or None, default=None
        Threads for the assignment step. None uses every core.
    random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
    labels_ : ndarray of shape (n_samples,)
    inertia_ : float
        Sum of squared distances of samples to their nearest centre.
    n_iter_ : int
        Epochs run.
    n_steps_ : int
        Mini-batch updates run.
    batch_size_, step_size_ : the values actually used.
    """

    def __init__(self, n_clusters=8, *, n_epochs=3, batch_size="auto", step_size="auto", init="random",
                 revive=True, balance=0.0, balance_end=0.0, compute_labels=True, n_threads=None, random_state=None):
        self.n_clusters = n_clusters
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.init = init
        self.revive = revive
        self.balance = balance
        self.balance_end = balance_end
        self.compute_labels = compute_labels
        self.n_threads = n_threads
        self.random_state = random_state

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=[np.float64, np.float32], order="C")
        with self._pool() as pool:
            return self._fit(X, pool)

    def _pool(self):
        return ThreadPoolExecutor(self.n_threads or cpu_count())

    def _fit(self, X, pool):
        n, d = X.shape
        k, T = int(self.n_clusters), int(self.n_epochs)
        if k < 1 or T < 1:
            raise ValueError("n_clusters and n_epochs must be >= 1.")
        if n < k:
            raise ValueError(f"n_samples={n} should be >= n_clusters={k}.")
        rng = check_random_state(self.random_state)
        B = batch_law(n, k, T, d) if isinstance(self.batch_size, str) else int(min(max(self.batch_size, 1), n))
        a0 = step_law(n, k, T, B) if isinstance(self.step_size, str) else float(self.step_size)
        if not 0 < a0 <= 1:
            raise ValueError("step_size must be in (0, 1].")
        lam, lam_end = float(self.balance), float(self.balance_end)
        if lam < 0 or not 0 <= lam_end <= 1:
            raise ValueError("balance must be >= 0 and balance_end in [0, 1].")
        dt = X.dtype

        if isinstance(self.init, str):
            if self.init != "random":
                raise ValueError("init must be 'random' or an array of centres.")
            C = X[rng.choice(n, k, replace=False)].copy()
        else:
            C = np.array(self.init, dtype=dt, order="C")
            if C.shape != (k, d):
                raise ValueError(f"init has shape {C.shape}, expected {(k, d)}.")

        # centre j is S[j] / N[j]; both are decayed running totals (paper Eq. 2-4)
        N = np.full(k, 1e-10, dtype=dt)
        S = N[:, None] * C
        delta = 2.0**-7
        sign = np.where(np.arange(d) % 2 == 0, 1.0, -1.0).astype(dt)
        up, down = 1 + delta * sign, 1 - delta * sign
        nb = n // B
        M = T * nb
        last = np.empty(n, dtype=np.intp)
        t = 0
        err_sum, err_cnt = 0.0, 1e-10
        for epoch in range(T):
            perm = rng.permutation(n)
            for b in range(nb):
                ids = perm[b * B:(b + 1) * B]
                xb = X[ids]
                alpha = a0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / M)))  # cosine-annealed step (Eq. 5)
                t += 1
                if lam > 0:
                    bias = None
                    if t > 1:   # size penalty: lam * mean distortion * (relative size - 1), cosine-annealed
                        fade = lam_end + (1 - lam_end) * 0.5 * (1 + math.cos(math.pi * (t - 1) / M))
                        bias = (lam * fade * err_sum / err_cnt) * (N * (k / N.sum()) - 1)
                    a, dist = _assign(xb, C, pool, return_dist=True, bias=bias)
                    err_sum = (1 - alpha) * err_sum + alpha * float(dist.sum(dtype=np.float64))
                    err_cnt = (1 - alpha) * err_cnt + alpha * len(ids)
                else:
                    a = _assign(xb, C, pool)
                if epoch == T - 1:
                    last[ids] = a
                nj, sj = _cluster_sums(xb, a, k)
                if alpha >= 1.0:    # step capped at 1: replace only the clusters this batch touched (no 0/0)
                    hit = nj > 0
                    N[hit] = nj[hit]
                    S[hit] = sj[hit]
                else:
                    N *= 1 - alpha
                    N += alpha * nj
                    S *= 1 - alpha
                    S += alpha * sj
                if self.revive and t % 5 == 0:
                    self._revive(N, S, up, down, rng)
                C = S / N[:, None]
        rest = perm[nb * B:]
        if len(rest):
            last[rest] = _assign(X[rest], C, pool)

        # last-epoch-mean finalization: exact mean of the points each cluster won in the final epoch
        cnt, sums = _cluster_sums(X, last, k)
        nz = cnt > 0
        C[nz] = sums[nz] / cnt[nz, None]

        self.cluster_centers_ = C
        self._n_features_out = k
        self.n_iter_ = T
        self.n_steps_ = M
        self.batch_size_ = B
        self.step_size_ = a0
        if self.compute_labels:
            self.labels_, dist = _assign(X, C, pool, return_dist=True)
            self.inertia_ = float(dist.sum(dtype=np.float64))
        return self

    @staticmethod
    def _revive(N, S, up, down, rng, tau=0.1):
        """Give each near-empty cluster half of a donor cluster drawn in proportion to its count (paper App. H)."""
        dead_mask = N < tau * N.mean()
        if not dead_mask.any():
            return
        alive = np.flatnonzero(~dead_mask)
        dead = np.flatnonzero(dead_mask)[:len(alive)]
        p = N[alive].astype(np.float64)
        donors = rng.choice(alive, size=len(dead), replace=False, p=p / p.sum())
        centre = S[donors] / N[donors, None]
        half = N[donors] / 2
        N[dead] = half
        N[donors] = half
        S[dead] = centre * up * half[:, None]
        S[donors] = centre * down * half[:, None]

    def predict(self, X):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], order="C", reset=False)
        with self._pool() as pool:
            return _assign(X, self.cluster_centers_.astype(X.dtype, copy=False), pool)

    def transform(self, X):
        """Distances from each sample to each cluster centre."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], order="C", reset=False)
        return euclidean_distances(X, self.cluster_centers_.astype(X.dtype, copy=False))

    def score(self, X, y=None):
        """Negative sum of squared distances to the nearest centre (higher is better), as in KMeans."""
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], order="C", reset=False)
        with self._pool() as pool:
            _, dist = _assign(X, self.cluster_centers_.astype(X.dtype, copy=False), pool, return_dist=True)
        return -float(dist.sum(dtype=np.float64))
