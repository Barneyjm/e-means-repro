"""Search-index quality for any dataset and any list of methods.

Usage: python ivf_sweep.py <base> <queries> <neighbours.npy> <log file> <k> <seeds> <methods>
  base, queries   .fvecs or .npy float32 arrays; neighbours.npy = exact nearest base ids per query (exact_neighbours.py)
  methods         comma separated, e.g. lloyd10,lloyd25,lloyd100,emeans1,emeans3,emeans10
Test MSE is measured on the first 10,000 queries. See ivf_recall.py for how recall and scan cost are computed.
"""
import sys, os, json, numpy as np, torch
from common import fvecs, dev
from emeans_torch import lloyd, emeans, Assigner
from ivf_recall import evaluate

def score(Xd, Xnp, Xte, Cf, ch=50_000):
    """train / test MSE in float64, in chunks so that wide data fits in memory"""
    idx = Assigner()(Xd, torch.from_numpy(Cf).to(dev)).cpu(); C = torch.from_numpy(Cf).double(); tot = 0.0
    for i in range(0, len(Xnp), ch): tot += float(((torch.from_numpy(Xnp[i:i+ch]).double() - C[idx[i:i+ch]]) ** 2).sum())
    T = torch.from_numpy(Xte).double(); cn = (C * C).sum(1); te = torch.cat([torch.min((T[i:i+2048] ** 2).sum(1)[:, None] + cn[None] - 2 * T[i:i+2048] @ C.T, 1).values for i in range(0, len(T), 2048)])
    return dict(train=tot / len(Xnp), test=float(te.mean()), dead_lt10=int((torch.bincount(idx, minlength=len(Cf)) < 10).sum()))

def load(p): return fvecs(p) if p.endswith(".fvecs") else np.ascontiguousarray(np.load(p), dtype=np.float32)

if __name__ == "__main__":
    bp, qp, gp, outp, k = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    seeds = [int(s) for s in sys.argv[6].split(",")]; methods = sys.argv[7].split(",")
    X = load(bp); gt = torch.from_numpy(np.load(gp)).long(); Q = load(qp)[:len(gt)]
    Xd = torch.from_numpy(X).to(dev); Qd = torch.from_numpy(Q).to(dev); Xte = Q[:10000]
    done = {(r["k"], r["m"], r["seed"]) for r in map(json.loads, open(outp))} if os.path.exists(outp) else set()   # resume
    f = open(outp, "a")
    emeans(Xd, X, k, 1, 99)                                                  # warm up GPU kernels
    for seed in seeds:
        for m in methods:
            if (k, m, seed) in done: continue
            if m.startswith("lloyd"): C, dt = lloyd(Xd, X, k, int(m[5:]), seed); info = {}
            else: C, dt, info = emeans(Xd, X, k, int(m[6:]), seed)
            s = json.dumps(dict(data=bp.split("/")[-1], n=len(X), d=X.shape[1], k=k, nq=len(gt), m=m, seed=seed, time=round(dt, 2), **info,
                                **score(Xd, X, Xte, C), **evaluate(Xd, Qd, gt, C)))
            print(s, flush=True); f.write(s + "\n"); f.flush()
