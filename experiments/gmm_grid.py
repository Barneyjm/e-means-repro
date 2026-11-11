"""Usage: python gmm_grid.py <data dir> <log file> <dataset> <K> <n rows> <seeds> [grid|classic|rule]

Do the paper's batch-size and step-size formulas (fitted for k-means) carry over to mixtures?
grid     batch size x peak step x passes, around the formula's choice
classic  the classic online-EM step rule (t+2)^-kappa given its best shot: small batches, with and without revival
rule     the formula as published vs the same memory window (batch / step) with 4x smaller batches, the direction
         the 100k grid favours, at 1/2/3/5/10 passes on the full data
"""
import sys, json, numpy as np, torch
from common import dev, laws
from gmm_torch import em, online_em, loglik
from run_gmm import sift64, synth

D, outp, ds, K, n = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
seeds = [int(s) for s in sys.argv[6].split(",")]; mode = sys.argv[7] if len(sys.argv) > 7 else "grid"
X, H, _ = sift64(D) if ds == "sift64" else synth(1.0, 10) if ds == "synth" else synth(1.5, 1)
X = X[:n]; Xd = torch.from_numpy(X).to(dev); X2d = Xd * Xd
f = open(outp, "a")
def log(**kw): s = json.dumps(dict(ds=ds, K=K, n=n, **kw)); print(s, flush=True); f.write(s + "\n"); f.flush()
def shifted(n, K, T):
    B, a = laws(n, K, T); return max(B // 4, 256), a / 4
online_em(Xd, X2d, K, 1, 99)
for seed in seeds:
    if mode == "grid":
        for T in (1, 3, 10):
            B0, a0 = laws(n, K, T); log(m="law", T=T, seed=seed, B=B0, a0=round(a0, 3))
            for bm in (1/32, 1/16, 1/8, 1/4, 1/2, 1, 2):
                for a in (0.03, 0.06, 0.125, 0.25, 0.5, 1.0):
                    B = int(B0 * bm)
                    if B < 128: continue
                    p, dt, info = online_em(Xd, X2d, K, T, seed, B=B, a0=a)
                    log(m="grid", T=T, seed=seed, bm=bm, time=round(dt, 2), test=round(loglik(H, *p), 5), **{**info, "a0": a})
    elif mode == "classic":
        T = 3; B0, _ = laws(n, K, T)
        for bm in (1/32, 1/8, 1/2, 1):
            for kp in (0.5, 0.6, 0.75, 0.9):
                for rv in (False, True):
                    p, dt, info = online_em(Xd, X2d, K, T, seed, B=int(B0 * bm), schedule="stepwise", kappa=kp, revive=rv, final="none")
                    log(m="classic", T=T, seed=seed, bm=bm, kappa=kp, revive=rv, time=round(dt, 2), test=round(loglik(H, *p), 5), **info)
    else:
        curve = []
        def hook(it, spent, p):
            curve.append(loglik(H, *p))
            if it in (10, 25, 50, 100, 200): log(m="em", iters=it, seed=seed, time=round(spent, 2), test=round(curve[-1], 5))
        em(Xd, X2d, K, 200, seed, hook)
        for T in (1, 2, 3, 5, 10):
            for name, (B, a) in (("published", laws(n, K, T)), ("shifted", shifted(n, K, T))):
                p, dt, info = online_em(Xd, X2d, K, T, seed, B=B, a0=a); te = loglik(H, *p)
                log(m="rule", rule=name, T=T, seed=seed, time=round(dt, 2), test=round(te, 5),
                    em_passes_to_match=next((i + 1 for i, v in enumerate(curve) if v >= te), None), **info)
