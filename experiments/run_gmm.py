"""Usage: python run_gmm.py <data dir> <log file> <dataset> <K> <seeds, comma separated> [main|ablate|all]

dataset: sift64  SIFT1M projected to 64 whitened PCA dimensions (the classic Fisher-vector setup); held-out = the 100k learn set
         synth   1M draws from a known 256-component mixture, mildly uneven weights; the true model gives the ceiling
         skewed  same, heavily uneven weights (most true components are under 10% of the mean weight)
Score = mean log-likelihood per held-out row, exact float64. Times are training only.
"""
import sys, json, numpy as np, torch
from common import fvecs, dev
from gmm_torch import em, online_em, loglik

def sift64(D, dims=64):
    X = fvecs(D + "/sift_base.fvecs").astype(np.float64); H = fvecs(D + "/sift_learn.fvecs").astype(np.float64)
    m = X.mean(0); ev, V = np.linalg.eigh(np.cov((X - m).T)); P = V[:, ::-1][:, :dims] / np.sqrt(ev[::-1][:dims])
    return ((X - m) @ P).astype(np.float32), ((H - m) @ P).astype(np.float32), None

def synth(power, shift, n=1_000_000, nte=100_000, K=256, d=64):
    rng = np.random.default_rng(123)
    w = (np.arange(K) + shift) ** -power; w /= w.sum()
    mu = rng.normal(0, 0.7, (K, d)); var = np.exp(rng.uniform(np.log(0.3), np.log(2.0), (K, d)))
    z = rng.choice(K, n + nte, p=w); X = (mu[z] + rng.standard_normal((n + nte, d)) * np.sqrt(var[z])).astype(np.float32)
    return X[:n], X[n:], (w, mu, var)

if __name__ == "__main__":
    D, outp, ds, K = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    seeds = [int(s) for s in sys.argv[5].split(",")]; mode = sys.argv[6] if len(sys.argv) > 6 else "all"
    X, H, truth = sift64(D) if ds == "sift64" else synth(1.0, 10) if ds == "synth" else synth(1.5, 1)
    Xd = torch.from_numpy(X).to(dev); X2d = Xd * Xd
    f = open(outp, "a")
    def log(**kw): s = json.dumps(dict(ds=ds, K=K, **kw)); print(s, flush=True); f.write(s + "\n"); f.flush()
    def weights(p): w = np.sort(p[0]); return dict(w_min_rel=round(float(w[0] * K), 4), n_small=int((w * K < 0.1).sum()))
    if truth: log(m="truth", test=round(loglik(H, *truth), 5), train=round(loglik(X, *truth), 5), **weights(truth))
    online_em(Xd, X2d, K, 1, 99)                                             # warm up GPU kernels
    for seed in seeds:
        curve = []
        if mode in ("main", "all"):
            def hook(it, spent, p):
                curve.append(loglik(H, *p))
                if it in (10, 25, 50, 100, 200):
                    log(m="em", iters=it, seed=seed, time=round(spent, 2), test=round(curve[-1], 5), train=round(loglik(X, *p), 5), **weights(p))
            em(Xd, X2d, K, 200, seed, hook)
            log(m="em_curve", seed=seed, test=[round(v, 5) for v in curve])
            for T in (1, 2, 3, 5, 10):
                p, dt, info = online_em(Xd, X2d, K, T, seed); te = loglik(H, *p)
                match = next((i + 1 for i, v in enumerate(curve) if v >= te), None)   # EM passes needed to do as well
                log(m="online", T=T, seed=seed, time=round(dt, 2), test=round(te, 5), train=round(loglik(X, *p), 5), em_passes_to_match=match, **info, **weights(p))
        if mode in ("ablate", "all"):
            T = 3
            variants = [("default", {}), ("constant step", dict(schedule="const")), ("no revival", dict(revive=False)),
                        ("paper-style split (mean*(1±2^-7))", dict(split="delta")), ("no last-pass average", dict(final="none")),
                        ("revival threshold 0.01", dict(tau=0.01))]
            variants += [(f"classic online EM, step (t+2)^-{kp}", dict(schedule="stepwise", kappa=kp, revive=False, final="none")) for kp in (0.5, 0.6, 0.75, 1.0)]
            variants += [(f"classic step (t+2)^-0.6 + last-pass average", dict(schedule="stepwise", kappa=0.6, revive=False))]
            B0, a0 = online_em.__globals__["laws"](len(X), K, T)
            variants += [(f"B x{bm}, step x{am}", dict(B=int(B0 * bm), a0=min(a0 * am, 1.0))) for bm in (0.125, 0.25, 1, 4) for am in (0.25, 0.5, 1) if (bm, am) != (1, 1)]
            for name, kw in variants:
                p, dt, info = online_em(Xd, X2d, K, T, seed, **kw)
                log(m="ablate", variant=name, T=T, seed=seed, time=round(dt, 2), test=round(loglik(H, *p), 5), **info, **weights(p))
