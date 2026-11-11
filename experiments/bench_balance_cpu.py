"""Usage: python bench_balance_cpu.py <base .npy> <queries .npy> <log file> <out dir> <k> <seeds>
CPU only, scikit-learn API: KMeans and MiniBatchKMeans as shipped, against EMeans with and without the size penalty.
Logs time, train/test MSE and cluster-size balance, and saves each set of centroids as
<out dir>/<data>_k<k>_<name>_s<seed>.npy for faiss_ivf_bench.py."""
import sys, os, json, time, warnings, numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from emeans_sklearn import EMeans, _assign
warnings.filterwarnings("ignore")
X = np.ascontiguousarray(np.load(sys.argv[1]), dtype=np.float32); Q = np.ascontiguousarray(np.load(sys.argv[2]), dtype=np.float32)[:10000]
outp, od, k = sys.argv[3], sys.argv[4], int(sys.argv[5]); seeds = [int(s) for s in sys.argv[6].split(",")]; tag = os.path.basename(sys.argv[1]).split(".")[0]
methods = [("sk_kmeans", lambda s: KMeans(k, random_state=s)), ("sk_kmeans_random25", lambda s: KMeans(k, init="random", n_init=1, max_iter=25, random_state=s)),
           ("sk_minibatch", lambda s: MiniBatchKMeans(k, random_state=s)), ("emeans3", lambda s: EMeans(k, n_epochs=3, random_state=s)),
           ("emeans3_bal0.3", lambda s: EMeans(k, n_epochs=3, balance=0.3, random_state=s)), ("emeans5_bal0.3", lambda s: EMeans(k, n_epochs=5, balance=0.3, random_state=s)),
           ("emeans3_bal0.1held", lambda s: EMeans(k, n_epochs=3, balance=0.1, balance_end=1.0, random_state=s))]
f = open(outp, "a")
for seed in seeds:
    for name, make in methods:
        t = time.time(); m = make(seed).fit(X); dt = time.time() - t; C = np.ascontiguousarray(m.cluster_centers_, dtype=np.float32)
        lab, dist = _assign(X, C, return_dist=True); cnt = np.bincount(lab, minlength=k).astype(float); _, dq = _assign(Q, C, return_dist=True)
        np.save(f"{od}/{tag}_k{k}_{name}_s{seed}.npy", C)
        s = json.dumps(dict(data=tag, n=len(X), k=k, m=name, seed=seed, time=round(dt, 2), iters=int(getattr(m, "n_iter_", 0)), train=float(dist.mean(dtype=np.float64)), test=float(dq.mean(dtype=np.float64)),
                            imbalance=round(float(k * (cnt ** 2).sum() / len(X) ** 2), 4), lt10=int((cnt < 10).sum()), max_list=int(cnt.max())))
        print(s, flush=True); f.write(s + "\n"); f.flush()
