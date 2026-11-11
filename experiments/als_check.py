"""Usage: python als_check.py <ml1m.npy>. Full ALS in als_torch.py against a plain numpy float64 loop, same start, same split."""
import sys, numpy as np, torch
from als_torch import Data, als, init, rmse
d = Data(sys.argv[1], 0); K, lam, sweeps = 16, 0.05, 5; tr = ~d.te; u, i, r = d.u[tr], d.i[tr], d.r[tr].astype(np.float64)
Q = init(d, K, 0)[0].cpu().numpy().astype(np.float64); P = np.zeros((d.nu, K))
def half(F, key, oth, nk):
    out = np.zeros((nk, F.shape[1])); o = np.argsort(key, kind="stable"); cnt = np.bincount(key, minlength=nk); s = 0
    for e in range(nk):
        rows = o[s:s + cnt[e]]; s += cnt[e]; G = F[oth[rows]]
        out[e] = np.linalg.solve(G.T @ G + lam * max(cnt[e], 1) * np.eye(F.shape[1]), G.T @ r[rows])
    return out
ref = []
for it in range(sweeps):
    P = half(Q, u, i, d.nu); Q = half(P, i, u, d.ni)
    ref.append(float(np.sqrt((((P[d.u[d.te]] * Q[d.i[d.te]]).sum(1) - d.r[d.te]) ** 2).mean())))
mine = []; als(d, K, sweeps, 0, lam, lambda it, s, P, Q: mine.append(rmse(d, P, Q)))
print("numpy  ", [round(x, 5) for x in ref]); print("als_torch", [round(x, 5) for x in mine]); print("max abs difference", max(abs(a - b) for a, b in zip(ref, mine)))
