"""OnlineGaussianMixture: a Gaussian mixture fitted in a few passes, as a scikit-learn estimator.

The e-means recipe (arXiv 2611.09501, written for k-means) applied to EM for mixtures with diagonal or spherical
covariances: mini-batch updates of decayed running totals, a cosine step size, re-seeding of starved components and a
final clean-up from the plain totals of the last pass. Needs only numpy, scipy and scikit-learn.
This is my own extension of the paper's method, not something the paper proposes or evaluates.
"""
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from joblib import cpu_count
from sklearn.base import BaseEstimator, DensityMixin
from sklearn.cluster import kmeans_plusplus
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted, validate_data

from emeans_sklearn import EMeans, batch_law, step_law

__all__ = ["OnlineGaussianMixture"]

_LOG2PI = math.log(2 * math.pi)


def _kernel(w, mu, var, dtype):
    """Fold the model into one (2d, K) matrix and a constant, so log w_k N(x | mu_k, var_k) = [x, x*x] @ W + const."""
    iv = 1.0 / var
    with np.errstate(divide="ignore"):
        const = np.log(w) - 0.5 * (np.log(var).sum(1) + (mu * mu * iv).sum(1) + mu.shape[1] * _LOG2PI)
    W = np.concatenate([(mu * iv).T, (-0.5 * iv).T])
    return np.ascontiguousarray(W, dtype=dtype), const.astype(dtype)


def _sweep(X, sel, shift, kern, pool, threads, mode, chunk_bytes=2**23):
    """Run the E-step over rows `sel` of X (None = all rows) in cache-sized chunks on a thread pool.

    mode "stats": responsibility-weighted totals (N, S, Q) in float64 and the summed log-likelihood
    mode "ll": log-likelihood per row;  "resp": responsibilities;  "label": most likely component
    numpy releases the GIL in matmul and in the element-wise loops, so the chunks run in parallel.
    """
    W, const = kern
    d, K = W.shape[0] // 2, W.shape[1]
    n = X.shape[0] if sel is None else len(sel)
    rows = int(min(max(chunk_bytes // (W.dtype.itemsize * max(K, 2 * d)), 64), 8192, max(-(-n // threads), 64)))

    def job(i):
        Z = np.empty((min(rows, n - i), 2 * d), dtype=W.dtype)
        x = Z[:, :d]
        x[:] = X[i:i + rows] if sel is None else X[sel[i:i + rows]]
        x -= shift
        np.multiply(x, x, out=Z[:, d:])
        L = Z @ W
        L += const
        if mode == "label":
            return L.argmax(axis=1)
        m = L.max(axis=1, keepdims=True)
        L -= m
        np.exp(L, out=L)
        s = L.sum(axis=1, keepdims=True)
        ll = (m + np.log(s))[:, 0]
        if mode == "ll":
            return ll
        L /= s
        if mode == "resp":
            return L
        return L.sum(axis=0, dtype=np.float64), L.T @ Z, float(ll.sum(dtype=np.float64))

    starts = range(0, n, rows)
    out = [job(i) for i in starts] if pool is None or len(starts) < 2 else list(pool.map(job, starts))
    if mode != "stats":
        return np.concatenate(out) if len(out) > 1 else out[0]
    N = np.zeros(K)
    SQ = np.zeros((K, 2 * d))
    ll = 0.0
    for a, b, c in out:
        N += a
        SQ += b
        ll += c
    return N, SQ[:, :d], SQ[:, d:], ll


class OnlineGaussianMixture(DensityMixin, BaseEstimator):
    """Gaussian mixture fitted by online EM with exponential-moving-average statistics.

    A stand-in for ``GaussianMixture(covariance_type="diag")`` on large data: it makes a fixed, small number of
    passes instead of iterating to convergence. Full and tied covariances are not supported.

    Parameters
    ----------
    n_components : int, default=1
    covariance_type : {"diag", "spherical"}, default="diag"
    n_epochs : int, default=3
        Passes over the data. One pass is usually not enough; three is the tested default.
    batch_size : int or "auto", default="auto"
        "auto" uses the e-means batch-size formula (fitted by its authors for k-means). With 16 features or fewer
        the formula's value is divided by 8, but not below 4 * n_components.
    step_size : float or "auto", default="auto"
        Peak step (1 - decay) of the running totals; it follows a cosine down to a tenth of this value.
        "auto" uses the e-means step-size formula.
    reg_covar : float, default=1e-6
        Added to every variance, as in GaussianMixture.
    init_params : {"auto", "random_from_data", "k-means++", "emeans"}, default="auto"
        Where the starting means come from: n_components distinct rows of X, k-means++ seeds (drawn from a
        random subset of at most max(20000, 50 * n_components) rows), or the centres of a 3-pass ``EMeans``
        clustering of X. "auto" uses k-means++ seeds up to 64 components and random rows above: with few,
        well separated clusters random rows often start two means in one cluster and the fit does not recover,
        while with hundreds of components the seeding costs more than it gains. Every component starts with
        weight 1 / n_components and the per-feature variance of the whole data set.
    means_init : array of shape (n_components, n_features), default=None
        Starting means; overrides init_params.
    revive : bool, default=True
        Every 5 updates, re-seed components whose weight fell below a tenth of the average by splitting a heavy
        component at +-0.5 standard deviations.
    n_threads : int or None, default=None
        Threads for the E-step. None uses every core.
    random_state : int, RandomState instance or None, default=None

    Attributes
    ----------
    weights_ : ndarray of shape (n_components,)
    means_ : ndarray of shape (n_components, n_features)
    covariances_ : ndarray of shape (n_components, n_features) for "diag", (n_components,) for "spherical"
    precisions_, precisions_cholesky_ : 1 / covariances_ and its square root, same shapes
    lower_bound_ : float
        Mean log-likelihood per training sample, measured during the last pass (so before the final clean-up).
    n_iter_ : int
        Passes run.
    n_steps_ : int
        Mini-batch updates run.
    n_revived_ : int
        Components re-seeded during the fit.
    batch_size_, step_size_ : the values actually used.

    Notes
    -----
    There is no convergence test and no ``converged_`` attribute: the fit always runs ``n_epochs`` passes.
    ``score`` returns the mean log-likelihood per sample, as GaussianMixture does.

    Where it pays off: hundreds of thousands of rows or more. In my tests on 1M rows of 64 features with 256
    components, 3 passes scored within 0.01 of GaussianMixture's defaults in under 1% of the time. With only a
    few hundred rows per component 3 passes are not enough (use 10 to 30), and below about 100,000 rows there is
    little or no time to gain. More passes do not always help, because the automatic batch size grows with the
    number of passes. See results/beyond/gmm_cpu_tables.md in the repository.
    """

    def __init__(self, n_components=1, *, covariance_type="diag", n_epochs=3, batch_size="auto", step_size="auto",
                 reg_covar=1e-6, init_params="auto", means_init=None, revive=True, n_threads=None,
                 random_state=None):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.step_size = step_size
        self.reg_covar = reg_covar
        self.init_params = init_params
        self.means_init = means_init
        self.revive = revive
        self.n_threads = n_threads
        self.random_state = random_state

    # ------------------------------------------------------------------ fitting
    def _params(self, N, S, Q):
        """Weights, means and variances (always (K, d)) from running totals."""
        Ns = np.maximum(N, 1e-30)[:, None]
        mu = S / Ns
        var = np.maximum(Q / Ns - mu * mu, 0.0)
        if self.covariance_type == "spherical":
            var = np.broadcast_to(var.mean(axis=1, keepdims=True), var.shape)
        return N / N.sum(), mu, var + self.reg_covar

    def fit(self, X, y=None):
        X = validate_data(self, X, dtype=[np.float64, np.float32], order="C", ensure_min_samples=2)
        n, d = X.shape
        K, T = int(self.n_components), int(self.n_epochs)
        if K < 1 or T < 1:
            raise ValueError("n_components and n_epochs must be >= 1.")
        if n < K:
            raise ValueError(f"n_samples={n} should be >= n_components={K}.")
        if self.covariance_type not in ("diag", "spherical"):
            raise ValueError("covariance_type must be 'diag' or 'spherical'; full and tied are not supported.")
        if self.reg_covar < 0:
            raise ValueError("reg_covar must be >= 0.")
        if self.init_params not in ("auto", "random_from_data", "k-means++", "emeans"):
            raise ValueError("init_params must be 'auto', 'random_from_data', 'k-means++' or 'emeans'.")
        start = self.init_params if self.init_params != "auto" else "k-means++" if K <= 64 else "random_from_data"
        rng = check_random_state(self.random_state)
        if isinstance(self.batch_size, str):
            B = batch_law(n, K, T, d)        # one eighth of the formula for 16 features or fewer, as in EMeans
            if d <= 16:                      # ... but keep a few rows per component
                B = max(B, min(4 * K, batch_law(n, K, T)))
        else:
            B = int(min(max(self.batch_size, 1), n))
        a0 = step_law(n, K, T, B) if isinstance(self.step_size, str) else float(self.step_size)
        if not 0 < a0 <= 1:
            raise ValueError("step_size must be in (0, 1].")
        threads = self.n_threads or cpu_count()
        dt, reg = X.dtype, self.reg_covar

        # work on centred data: keeps mean^2 small next to the variance, which matters in float32
        tot = np.zeros(d)
        sq = np.zeros(d)
        for i in range(0, n, 65536):
            xc = X[i:i + 65536].astype(np.float64)
            tot += xc.sum(0)
            sq += np.einsum("ij,ij->j", xc, xc)
        shift64 = tot / n
        gvar = np.maximum(sq / n - shift64 * shift64, 0.0)
        shift = shift64.astype(dt)

        if self.means_init is None and start == "emeans":
            seed = rng.randint(np.iinfo(np.int32).max)
            km = EMeans(K, compute_labels=False, n_threads=self.n_threads, random_state=seed).fit(X)
            mu = km.cluster_centers_.astype(np.float64) - shift64
        elif self.means_init is None and start == "k-means++":
            sub = X if n <= max(20000, 50 * K) else X[rng.choice(n, max(20000, 50 * K), replace=False)]
            mu = kmeans_plusplus(sub, K, random_state=rng)[0].astype(np.float64) - shift64
        elif self.means_init is None:
            mu = X[rng.choice(n, K, replace=False)].astype(np.float64) - shift64
        else:
            mu = np.array(self.means_init, dtype=np.float64)
            if mu.shape != (K, d):
                raise ValueError(f"means_init has shape {mu.shape}, expected {(K, d)}.")
            mu = mu - shift64
        # three decayed running totals per component: weight N, sum S, sum of squares Q
        N = np.full(K, 1e-10)
        S = N[:, None] * mu
        Q = N[:, None] * (gvar[None, :] + mu * mu)
        w, mu, var = self._params(N, S, Q)
        w = np.full(K, 1.0 / K)
        sign = np.where(np.arange(d) % 2 == 0, 1.0, -1.0)
        nb = n // B
        M = T * nb
        fN, fS, fQ, fll, fn = np.zeros(K), np.zeros((K, d)), np.zeros((K, d)), 0.0, 0
        t = revived = 0
        with ThreadPoolExecutor(threads) as pool:
            for epoch in range(T):
                perm = rng.permutation(n)
                for b in range(nb):
                    ids = perm[b * B:(b + 1) * B]
                    alpha = self._step(t, M, a0)
                    t += 1
                    bN, bS, bQ, ll = _sweep(X, ids, shift, _kernel(w, mu, var, dt), pool, threads, "stats")
                    if epoch == T - 1:
                        fN += bN
                        fS += bS
                        fQ += bQ
                        fll += ll
                        fn += len(ids)
                    if alpha >= 1.0:    # step capped at 1: replace only the components this batch reached, as in EMeans
                        hit = bN > 1e-8 * bN.sum()
                        N[hit], S[hit], Q[hit] = bN[hit], bS[hit], bQ[hit]
                    else:
                        for run, batch in ((N, bN), (S, bS), (Q, bQ)):
                            run *= 1 - alpha
                            run += alpha * batch
                    if self.revive and t % 5 == 0:
                        revived += self._revive(N, S, Q, sign, rng)
                    self._after_update(t, M, N, S, Q)
                    w, mu, var = self._params(N, S, Q)
            rest = perm[nb * B:]
            if len(rest):
                bN, bS, bQ, ll = _sweep(X, rest, shift, _kernel(w, mu, var, dt), pool, threads, "stats")
                fN += bN
                fS += bS
                fQ += bQ
                fll += ll
                fn += len(rest)
        w, mu, var = self._params(*self._final_totals((N, S, Q), (fN, fS, fQ)))

        self.weights_ = w
        self.means_ = mu + shift64
        self.covariances_ = var[:, 0].copy() if self.covariance_type == "spherical" else np.array(var)
        self.precisions_ = 1.0 / self.covariances_
        self.precisions_cholesky_ = np.sqrt(self.precisions_)
        self.lower_bound_ = fll / fn
        self.n_iter_ = T
        self.n_steps_ = M
        self.n_revived_ = revived
        self.batch_size_ = B
        self.step_size_ = a0
        return self

    # The three methods below are the recipe's schedule and finish; subclasses can swap them to try other rules.
    def _step(self, t, M, a0):
        """Step size of update t out of M: a cosine from a0 down to a0 / 10."""
        return a0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / M)))

    def _after_update(self, t, M, N, S, Q):
        """Called after update t with the running totals."""

    def _final_totals(self, running, last_pass):
        """Final clean-up: use the plain (undecayed) totals of the last pass. A component that received nothing
        in that pass gets weight zero."""
        fN, fS, fQ = last_pass
        keep = fN > 1e-8 * fN.sum()
        return np.where(keep, fN, 0.0), np.where(keep[:, None], fS, 0.0), np.where(keep[:, None], fQ, 0.0)

    def _revive(self, N, S, Q, sign, rng, tau=0.1, size=0.5):
        """Give each starved component half of a heavy one, drawn in proportion to weight; the two halves are
        placed at mean +- size * standard deviation with alternating signs across features."""
        dead_mask = N < tau * N.mean()
        if not dead_mask.any():
            return 0
        alive = np.flatnonzero(~dead_mask)
        dead = np.flatnonzero(dead_mask)[:len(alive)]
        donors = rng.choice(alive, size=len(dead), replace=False, p=N[alive] / N[alive].sum())
        _, m, v = self._params(N[donors], S[donors], Q[donors])
        half = N[donors] / 2
        off = size * np.sqrt(v) * sign
        for idx, m2 in ((dead, m + off), (donors, m - off)):
            N[idx] = half
            S[idx] = half[:, None] * m2
            Q[idx] = half[:, None] * (v - self.reg_covar + m2 * m2)
        return len(dead)

    # ------------------------------------------------------------------ inference
    def _infer(self, X, mode):
        check_is_fitted(self)
        X = validate_data(self, X, dtype=[np.float64, np.float32], order="C", reset=False)
        var = self.covariances_[:, None] if self.covariance_type == "spherical" else self.covariances_
        var = np.broadcast_to(var, self.means_.shape)
        shift = self.weights_ @ self.means_    # any shift gives the same model; this one keeps float32 well conditioned
        kern = _kernel(self.weights_, self.means_ - shift, var, X.dtype)
        threads = self.n_threads or cpu_count()
        with ThreadPoolExecutor(threads) as pool:
            return _sweep(X, None, shift.astype(X.dtype), kern, pool, threads, mode)

    def score_samples(self, X):
        """Log-likelihood of each sample."""
        return self._infer(X, "ll")

    def score(self, X, y=None):
        """Mean log-likelihood per sample."""
        return float(self.score_samples(X).mean(dtype=np.float64))

    def predict_proba(self, X):
        """Posterior probability of each component for each sample."""
        return self._infer(X, "resp")

    def predict(self, X):
        """Most likely component for each sample."""
        return self._infer(X, "label")

    def fit_predict(self, X, y=None):
        return self.fit(X).predict(X)

    def sample(self, n_samples=1):
        """Draw samples from the fitted mixture. Returns (X, component labels), grouped by component."""
        check_is_fitted(self)
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1.")
        rng = check_random_state(self.random_state)
        counts = rng.multinomial(n_samples, self.weights_)
        std = np.sqrt(self.covariances_)
        if self.covariance_type == "spherical":
            std = std[:, None]
        d = self.means_.shape[1]
        X = np.vstack([m + rng.standard_normal((c, d)) * s for m, s, c in zip(self.means_, std, counts)])
        return X, np.repeat(np.arange(len(counts)), counts)

    def _n_parameters(self):
        K, d = self.means_.shape
        return K * d + (K if self.covariance_type == "spherical" else K * d) + K - 1

    def bic(self, X):
        """Bayesian information criterion on X (lower is better)."""
        return -2 * self.score(X) * X.shape[0] + self._n_parameters() * math.log(X.shape[0])

    def aic(self, X):
        """Akaike information criterion on X (lower is better)."""
        return -2 * self.score(X) * X.shape[0] + 2 * self._n_parameters()
