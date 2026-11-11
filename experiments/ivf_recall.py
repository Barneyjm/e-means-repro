"""Does a lower clustering error give a better search index?

Usage: python ivf_recall.py <data dir> <log file> <k> <seeds, comma separated> [neighbours.npy]

Builds an inverted-file (IVF) index from each set of centroids: every base vector goes in the list of its
nearest centroid, and a query scans the lists of its `nprobe` nearest centroids. With exact distances inside
the lists, a true neighbour is found exactly when its list is scanned, so recall at every nprobe follows from
the rank of that list. Cost = number of base vectors scanned. Reports recall at fixed nprobe and the cost
needed to reach fixed recall (the fair comparison: uneven lists make the same nprobe cost more).

Queries: the official 10k with their ground truth, or, given a file from exact_neighbours.py, the 100k 'learn'
vectors (10x the queries, so about a third of the noise).
"""
import sys, json, numpy as np, torch
from common import fvecs, dev
from emeans_torch import lloyd, emeans, Assigner, score

def ivecs(p):
    a = np.fromfile(p, dtype=np.int32); d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def evaluate(Xd, Qd, gt, C, P=256, ch=5000):
    n, k = Xd.shape[0], len(C); P = min(P, k); Cd = torch.from_numpy(C).to(dev)
    a = Assigner()(Xd, Cd); sizes = torch.bincount(a, minlength=k).cpu(); a = a.cpu()
    cn = (Cd * Cd).sum(1); Ct = Cd.T.contiguous()
    h1 = torch.zeros(P + 1, dtype=torch.int64); h10 = torch.zeros(P + 1, dtype=torch.int64); cost = torch.zeros(P, dtype=torch.float64)
    for i in range(0, len(Qd), ch):
        q = Qd[i:i+ch]
        order = torch.topk(torch.addmm(cn[None].expand(len(q), -1), q, Ct, alpha=-2), P, dim=1, largest=False).indices.cpu()  # nearest lists, in order
        gcell = a[gt[i:i+ch, :10]]                                               # lists holding the 10 true neighbours
        hit = order[:, :, None] == gcell[:, None, :]
        rank = torch.where(hit.any(1), hit.int().argmax(1), torch.full_like(gcell, P))
        h1 += torch.bincount(rank[:, 0], minlength=P + 1); h10 += torch.bincount(rank.flatten(), minlength=P + 1)
        cost += sizes[order].cumsum(1).double().sum(0)
    r1 = (h1[:P].cumsum(0) / len(Qd)).numpy(); r10 = (h10[:P].cumsum(0) / (10 * len(Qd))).numpy(); cost = (cost / len(Qd)).numpy()
    out = dict(imbalance=round(float(k * (sizes.double() ** 2).sum() / n ** 2), 4), max_list=int(sizes.max()), empty=int((sizes == 0).sum()))
    for p in (1, 2, 4, 8, 16, 32, 64, 128):
        if p <= P: out.update({f"r1@{p}": round(float(r1[p-1]), 5), f"r10@{p}": round(float(r10[p-1]), 5), f"cost@{p}": round(float(cost[p-1]), 1)})
    for name, r in (("r1", r1), ("r10", r10)):
        for tgt in (0.9, 0.95, 0.99):
            out[f"cost_{name}>={tgt}"] = round(float(np.interp(tgt, r, cost)), 1) if r[-1] >= tgt else None
    return out

if __name__ == "__main__":
    D, outp, k = sys.argv[1], sys.argv[2], int(sys.argv[3]); seeds = [int(s) for s in sys.argv[4].split(",")]
    X = fvecs(D + "/sift_base.fvecs"); Xte = fvecs(D + "/sift_query.fvecs"); Xd = torch.from_numpy(X).to(dev)
    if len(sys.argv) > 5:
        gt = torch.from_numpy(np.load(sys.argv[5])).long(); Qd = torch.from_numpy(fvecs(D + "/sift_learn.fvecs")[:len(gt)]).to(dev); qs = "learn"
    else:
        gt = torch.from_numpy(ivecs(D + "/sift_groundtruth.ivecs")).long(); Qd = torch.from_numpy(Xte).to(dev); qs = "official"
    f = open(outp, "a")
    def log(m, seed, dt, C):
        s = json.dumps(dict(k=k, queries=qs, nq=len(gt), m=m, seed=seed, time=round(dt, 2), **score(Xd, X, Xte, C), **evaluate(Xd, Qd, gt, C)))
        print(s, flush=True); f.write(s + "\n"); f.flush()
    emeans(Xd, X, k, 1, 99)                                                  # warm up GPU kernels
    for seed in seeds:
        for it in (10, 25):
            C, dt = lloyd(Xd, X, k, it, seed); log(f"lloyd{it}", seed, dt, C)
        for T in (1, 2, 3, 5):
            C, dt, info = emeans(Xd, X, k, T, seed); log(f"emeans{T}", seed, dt, C)
