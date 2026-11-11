"""Usage: python faiss_ivf_bench.py <base> <queries> <neighbours.npy> <log file> <centroids.npy> [more centroids ...]
A real Faiss IndexIVFFlat per set of centroids: recall, distance computations and query time over a sweep of nprobe,
and the cost needed for 90 / 95 / 99 % recall (interpolated). Independent of the torch code (Faiss and PyTorch cannot
share a process), so it also checks the recall arithmetic in ivf_recall.py. FAISS_NQ=<n> limits the number of queries."""
import sys, os, json, time, numpy as np, faiss
def load(p):
    if p.endswith(".npy"): return np.ascontiguousarray(np.load(p), dtype=np.float32)
    a = np.fromfile(p, dtype=np.int32); return a.reshape(-1, a[0] + 1)[:, 1:].view(np.float32).copy()
X = load(sys.argv[1]); gt = np.load(sys.argv[3])[:int(os.environ.get("FAISS_NQ", 10**9)), :10]; Q = load(sys.argv[2])[:len(gt)]; f = open(sys.argv[4], "a")
for cp in sys.argv[5:]:
    C = np.load(cp); quant = faiss.IndexFlatL2(X.shape[1]); quant.add(C)
    index = faiss.IndexIVFFlat(quant, X.shape[1], len(C)); index.train(X[:1000]); index.add(X)   # quantizer is pre-filled, so train() keeps the centroids
    probes = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]; r1, r10, nd, ms = [], [], [], []
    for p in probes:
        index.nprobe = p; faiss.cvar.indexIVF_stats.reset(); t = time.time(); _, I = index.search(Q, 10); dt = time.time() - t
        r1.append(float((I == gt[:, :1]).any(1).mean())); r10.append(float(np.mean([len(set(a) & set(b)) for a, b in zip(I[:2000], gt[:2000])]) / 10))
        nd.append(faiss.cvar.indexIVF_stats.ndis / len(Q)); ms.append(1e6 * dt / len(Q))
    out = dict(centroids=cp.split("/")[-1], k=len(C), nq=len(Q), probes=probes, r1=r1, r10=r10, ndis=nd, us_per_query=ms)
    for name, r in (("r1", r1), ("r10", r10)):
        for tgt in (0.9, 0.95, 0.99):
            ok = r[-1] >= tgt; out[f"ndis_{name}>={tgt}"] = round(float(np.interp(tgt, r, nd)), 1) if ok else None; out[f"us_{name}>={tgt}"] = round(float(np.interp(tgt, r, ms)), 2) if ok else None
    s = json.dumps(out); print(s, flush=True); f.write(s + "\n"); f.flush()
