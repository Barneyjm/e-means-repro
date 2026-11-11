"""Usage: python pq_codebooks.py <data dir> <log file> <train: base|learn> <seeds, comma separated> [lowd|bal]
(lowd: only the low-dimension batch rule; bal: only that rule plus the faded size penalty)

Product quantization: cut each 128-dim SIFT vector into 16 pieces of 8 dims and learn a 256-entry codebook per piece
(the classic 16-byte code). Low dimension and few clusters, i.e. outside the range the paper's formulas were fitted on.
Score = reconstruction error on held-out vectors (summed over the 16 pieces), Lloyd 25 vs e-means.
"""
import sys, json, numpy as np, torch
from common import fvecs, dev, mse, laws
from emeans_torch import lloyd, emeans
D, outp, tr = sys.argv[1], sys.argv[2], sys.argv[3]; seeds = [int(s) for s in sys.argv[4].split(",")]; lowd_only = len(sys.argv) > 5; bal_only = len(sys.argv) > 5 and sys.argv[5] == 'bal'
X = fvecs(D + f"/sift_{tr}.fvecs"); Xte = fvecs(D + "/sift_query.fvecs"); Xd = torch.from_numpy(X).to(dev)
M, k = 16, 256; ds = X.shape[1] // M
f = open(outp, "a")
def run(name, seed, fit):
    t = tr_e = te_e = 0.0; info = {}
    for j in range(M):
        sl = slice(j * ds, (j + 1) * ds); Xs = np.ascontiguousarray(X[:, sl]); res = fit(Xd[:, sl].contiguous(), Xs, seed * 100 + j)
        t += res[1]; tr_e += mse(Xs, res[0]); te_e += mse(np.ascontiguousarray(Xte[:, sl]), res[0]); info = res[2] if len(res) > 2 else {}
    s = json.dumps(dict(train_set=tr, n=len(X), m=name, seed=seed, time=round(t, 2), train=round(tr_e, 3), test=round(te_e, 3), **info)); print(s, flush=True); f.write(s + "\n"); f.flush()
emeans(Xd[:, :ds].contiguous(), np.ascontiguousarray(X[:, :ds]), k, 1, 99)
for seed in seeds:
    if not lowd_only:
        for it in (10, 25, 100): run(f"lloyd{it}", seed, lambda xd, xs, sd: lloyd(xd, xs, k, it, sd))
        for T in (1, 2, 3, 5, 10): run(f"emeans{T}", seed, lambda xd, xs, sd: emeans(xd, xs, k, T, sd))
    if bal_only:                                                             # size penalty faded out (emeans_lab), on the low-dimension batch rule
        from emeans_lab import emeans_x
        for T in (3, 5):
            Bq, aq = laws(len(X), k, T, d=ds); run(f"emeans{T}_lowd_bal0.3", seed, lambda xd, xs, sd: emeans_x(xd, xs, k, T, sd, B=Bq, a0=aq, consc=0.3))
        continue
    for T in (1, 2, 3, 5, 10):                                               # authors' code: 1/8 of the batch size when d <= 16
        Bq, aq = laws(len(X), k, T, d=ds); run(f"emeans{T}_lowd", seed, lambda xd, xs, sd: emeans(xd, xs, k, T, sd, B=Bq, a0=aq))
