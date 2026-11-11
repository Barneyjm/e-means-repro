"""Usage: python faiss_ivf_check.py <data dir> <centroids.npy>. Cross-check of ivf_recall.py: build a real Faiss
IndexIVFFlat on the given centroids and print recall of the true nearest neighbour and vectors scanned per query."""
import sys, numpy as np, faiss
def fvecs(p):
    a = np.fromfile(p, dtype=np.int32); d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()
D = sys.argv[1]; C = np.load(sys.argv[2]); X = fvecs(D + "/sift_base.fvecs"); Q = fvecs(D + "/sift_query.fvecs")
a = np.fromfile(D + "/sift_groundtruth.ivecs", dtype=np.int32); gt = a.reshape(-1, a[0] + 1)[:, 1:]
quant = faiss.IndexFlatL2(X.shape[1]); quant.add(C)
index = faiss.IndexIVFFlat(quant, X.shape[1], len(C)); index.train(X[:1000]); index.add(X)   # quantizer is pre-filled, so train() keeps the centroids
for p in (1, 8, 32):
    index.nprobe = p; faiss.cvar.indexIVF_stats.reset(); _, I = index.search(Q, 10)
    print(f"nprobe {p}: r1 {(I == gt[:, :1]).any(1).mean():.4f}  scanned {faiss.cvar.indexIVF_stats.ndis / len(Q):.1f}")
