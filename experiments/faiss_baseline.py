"""Independent reference: Faiss CPU Lloyd k-means, 25 iterations, no subsampling.
Run in its own process (faiss and torch each ship an OpenMP runtime), then score with score_centroids.py.
Usage: python faiss_baseline.py <data dir> <centroids.npy> <k>"""
import sys, time, numpy as np, faiss
data, out, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
a = np.fromfile(data + "/sift_base.fvecs", dtype=np.int32); d = a[0]
X = a.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()
km = faiss.Kmeans(int(d), k, niter=25, seed=0, max_points_per_centroid=10**9, verbose=True)
t = time.time(); km.train(X); dt = time.time() - t
print(f"faiss-cpu lloyd25: {faiss.omp_get_max_threads()} threads, {dt:.1f} s", flush=True)
np.save(out, km.centroids)
