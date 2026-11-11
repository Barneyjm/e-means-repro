"""Usage: python centroid_sets.py <base .npy> <queries .npy> <log file> <out dir> <k> <seeds> <variant> [...]
Centroids for the real-index check, CPU only. Variants:
  faiss<T>          Faiss's own k-means, T iterations, no subsampling
  mine<T>[+<lam>]   EMeans from emeans_sklearn.py, T passes; +<lam> adds the size penalty of EXTENSIONS.md section 5, faded out
                    over training (append h to hold it instead)
  auth<T>[+<lam>]   the authors' released NumPy e-means (faiss/contrib/e_means, not in a Faiss release yet), their defaults.
                    Needs EMEANS_AUTHORS=<path to their e_means directory>. With +<lam>, one method of their class is
                    overridden to add the same penalty to the assignment step; their files are not edited.
Logs time, train / held-out MSE and list balance; saves <out dir>/<data>_k<k>_<variant>_s<seed>.npy for faiss_ivf_bench.py."""
import sys, os, json, math, time, importlib, numpy as np, faiss
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from emeans_sklearn import EMeans

def authors(lam, held):
    pkg = os.path.abspath(os.environ["EMEANS_AUTHORS"]); sys.path.insert(0, os.path.dirname(pkg))
    NumpyEmeans = importlib.import_module(os.path.basename(pkg)).NumpyEmeans
    if not lam: return NumpyEmeans
    step_at = importlib.import_module(os.path.basename(pkg) + ".base")._scheduled_alpha
    class Balanced(NumpyEmeans):
        """Assign each row to argmin |x - c_j|^2 + lam * fade * D * (k * n_j / n - 1): D = running mean distortion, n_j = the
        class's own running counts. Exact with faiss.knn: every centroid gets one extra coordinate sqrt(bias_j)."""
        _cur, _es, _ec, _bs, _bn, _bias = -1, 0.0, 1e-10, 0.0, 0, None
        def _assign_rows(self, x, centroids):
            total = self.recipe["total_steps"]
            if self._cur != self._step:                      # first chunk of a new batch
                if self._cur >= 0:
                    a = step_at(self.recipe["alpha0"], self._cur, total); self._es = (1 - a) * self._es + a * self._bs; self._ec = (1 - a) * self._ec + a * self._bn
                self._cur, self._bs, self._bn, self._bias = self._step, 0.0, 0, None
                if self._step > 0:
                    fade = held + (1 - held) * 0.5 * (1 + math.cos(math.pi * min(self._step, total) / total)); u = self._usage.astype(np.float64)
                    b = lam * fade * (self._es / self._ec) * (u * (self.k / u.sum()) - 1); self._bias = (b - b.min()).astype(np.float32)
                    self._caug = np.ascontiguousarray(np.hstack([centroids, np.sqrt(self._bias)[:, None]]), dtype=np.float32)
            if self._bias is None: d, l = faiss.knn(x, centroids, 1); d, l = d.ravel(), l.ravel()
            else:
                d, l = faiss.knn(np.ascontiguousarray(np.hstack([x, np.zeros((len(x), 1), np.float32)])), self._caug, 1); l = l.ravel(); d = d.ravel() - self._bias[l]
            self._bs += float(d.sum(dtype=np.float64)); self._bn += len(x)
            return d, l
    return Balanced

def train(X, k, v, seed):
    name, _, pen = v.partition("+"); held = float(pen.endswith("h")); lam = float(pen.rstrip("h")) if pen else 0.0
    if name.startswith("faiss"):
        km = faiss.Kmeans(X.shape[1], k, niter=int(name[5:]), seed=seed + 1, max_points_per_centroid=10**9); km.train(X); return km.centroids
    T = int(name[4:])
    if name.startswith("mine"): return EMeans(k, n_epochs=T, balance=lam, balance_end=held, random_state=seed).fit(X).cluster_centers_
    return authors(lam, held)(X.shape[1], k, nepochs=T, seed=seed + 1, capture_assignments=False).train(X)[0]

if __name__ == "__main__":
    X = np.ascontiguousarray(np.load(sys.argv[1]), dtype=np.float32); Q = np.ascontiguousarray(np.load(sys.argv[2]), dtype=np.float32)[:10000]
    outp, od, k = sys.argv[3], sys.argv[4], int(sys.argv[5]); seeds = [int(s) for s in sys.argv[6].split(",")]; tag = os.path.basename(sys.argv[1]).split(".")[0]
    done = {(r["m"], r["seed"]) for r in map(json.loads, open(outp))} if os.path.exists(outp) else set(); f = open(outp, "a")
    for seed in seeds:
        for v in sys.argv[7:]:
            if (v, seed) in done: continue
            t = time.time(); C = np.ascontiguousarray(train(X, k, v, seed), dtype=np.float32); dt = time.time() - t
            dist, lab = faiss.knn(X, C, 1); cnt = np.bincount(lab.ravel(), minlength=k).astype(float); dq, _ = faiss.knn(Q, C, 1)
            np.save(f"{od}/{tag}_k{k}_{v}_s{seed}.npy", C)
            s = json.dumps(dict(data=tag, n=len(X), d=X.shape[1], k=k, m=v, seed=seed, time=round(dt, 2), train=float(dist.mean(dtype=np.float64)), test=float(dq.mean(dtype=np.float64)),
                                imbalance=round(float(k * (cnt ** 2).sum() / len(X) ** 2), 4), lt10=int((cnt < 10).sum()), max_list=int(cnt.max())))
            print(s, flush=True); f.write(s + "\n"); f.flush()
