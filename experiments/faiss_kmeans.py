"""Usage: python faiss_kmeans.py <base .npy|.fvecs> <out prefix> <k> <seeds> [iterations ...]
Faiss's own k-means (CPU) as an outside baseline: saves <out prefix>_faiss<iterations>_s<seed>.npy. No subsampling
(max_points_per_centroid is lifted); with `ivf` as an iteration count it uses what an IndexIVF trains by default
(10 iterations, at most 256 points per centroid)."""
import sys, time, numpy as np, faiss
def load(p):
    if p.endswith(".npy"): return np.ascontiguousarray(np.load(p), dtype=np.float32)
    a = np.fromfile(p, dtype=np.int32); return a.reshape(-1, a[0] + 1)[:, 1:].view(np.float32).copy()
X = load(sys.argv[1]); k = int(sys.argv[3])
for seed in [int(s) for s in sys.argv[4].split(",")]:
    for it in sys.argv[5:] or ["25"]:
        km = faiss.Kmeans(X.shape[1], k, niter=10, seed=seed + 1) if it == "ivf" else faiss.Kmeans(X.shape[1], k, niter=int(it), seed=seed + 1, max_points_per_centroid=10**9)
        t = time.time(); km.train(X); print(f"faiss k-means {it} seed {seed}: {time.time() - t:.1f} s", flush=True); np.save(f"{sys.argv[2]}_faiss{it}_s{seed}.npy", km.centroids)
