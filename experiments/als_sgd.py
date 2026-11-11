"""Usage: python als_sgd.py <ratings.npy> <log file> <K> <seeds> [epochs=20] [batch=8192] [lrs=0.01,0.03,0.1]

Rough reference only: plain mini-batch SGD matrix factorization (no biases, no momentum), same split, same centring and
the same held-out RMSE as als_run.py. Per rating: p += lr (e q - reg p), q += lr (e p - reg q), gradients of a batch
summed per row; lr follows a cosine to 10 %. A small grid over lr and reg, best reported; it is not a tuned SGD system.
"""
import sys, json, math, time, torch
from common import dev, sync
from als_torch import Data, rmse
path, outp, K = sys.argv[1], sys.argv[2], int(sys.argv[3]); seeds = [int(s) for s in sys.argv[4].split(",")]
E = int(sys.argv[5]) if len(sys.argv) > 5 else 20; B = int(sys.argv[6]) if len(sys.argv) > 6 else 8192
LRS = [float(x) for x in sys.argv[7].split(",")] if len(sys.argv) > 7 else (0.01, 0.03, 0.1); f = open(outp, "a")
for seed in seeds:
    d = Data(path, seed); tr = torch.from_numpy(~d.te).to(dev); u = torch.from_numpy(d.u).to(dev)[tr]; i = torch.from_numpy(d.i).to(dev)[tr]; r = torch.from_numpy(d.r).to(dev)[tr]; n = len(r)
    for lr in LRS:
        for reg in (0.02, 0.05):
            g = torch.Generator().manual_seed(seed); P = (0.1 * torch.randn(d.nu, K, generator=g)).to(dev); Q = (0.1 * torch.randn(d.ni, K, generator=g)).to(dev)
            sync(); t0 = time.time(); spent = 0.0; curve = []; steps = E * (n // B); t = 0
            for ep in range(E):
                perm = torch.randperm(n, generator=g).to(dev)
                for bi in range(n // B):
                    ids = perm[bi * B:(bi + 1) * B]; ub, ib = u[ids], i[ids]; p, q = P[ub], Q[ib]; e = (r[ids] - (p * q).sum(1))[:, None]
                    a = lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / steps))); t += 1
                    P.index_add_(0, ub, e * q - reg * p, alpha=a); Q.index_add_(0, ib, e * p - reg * q, alpha=a)
                sync(); spent += time.time() - t0; curve.append(round(rmse(d, P, Q), 5)); t0 = time.time()
                if not math.isfinite(curve[-1]): break
            s = json.dumps(dict(data=path.split("/")[-1].split(".")[0], K=K, m="sgd", seed=seed, lr=lr, reg=reg, batch=B, epochs=len(curve), time=round(spent, 2), test=curve[-1], best=min(curve), curve=curve))
            print(s, flush=True); f.write(s + "\n"); f.flush()
