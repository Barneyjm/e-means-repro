"""Usage: python gmm_sklearn_ref.py <data dir> <log file> <dataset> <K> <seeds>
scikit-learn's GaussianMixture as users would run it (CPU, float64, defaults: k-means start, up to 100 EM passes),
plus the k-means++ start. Reference for wall time and for whether a better start closes the gap to online EM."""
import sys, json, time, warnings, numpy as np
from sklearn.mixture import GaussianMixture
from run_gmm import sift64, synth
from gmm_torch import loglik
warnings.filterwarnings("ignore")
D, outp, ds, K = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); seeds = [int(s) for s in sys.argv[5].split(",")]
X, H, _ = sift64(D) if ds == "sift64" else synth(1.0, 10) if ds == "synth" else synth(1.5, 1)
X = X.astype(np.float64); f = open(outp, "a")
for seed in seeds:
    for init in ("kmeans", "k-means++"):
        t = time.time(); g = GaussianMixture(K, covariance_type="diag", init_params=init, random_state=seed).fit(X); dt = time.time() - t
        s = json.dumps(dict(ds=ds, K=K, m="sklearn", init=init, seed=seed, time=round(dt, 1), passes=int(g.n_iter_), converged=bool(g.converged_),
                            test=round(loglik(H, g.weights_, g.means_, g.covariances_), 5), n_small=int((g.weights_ * K < 0.1).sum())))
        print(s, flush=True); f.write(s + "\n"); f.flush()
