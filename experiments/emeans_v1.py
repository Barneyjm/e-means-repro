"""First, straightforward re-implementation of e-means (arXiv 2611.09501, Alg. 1 + App. H) and Lloyd k-means.
Both use the same assignment kernel, so one epoch costs the same for each. Slow; see emeans.py for the fast one."""
import math, numpy as np, torch
from common import dev, mse, split_pattern, laws

def assign(X, C, chunk=8192):
    """nearest-centroid index for each row of X (float32 on device)"""
    cn = (C * C).sum(1)
    out = torch.empty(X.shape[0], dtype=torch.int64, device=X.device)
    for i in range(0, X.shape[0], chunk):
        xc = X[i:i+chunk]
        out[i:i+chunk] = torch.argmin(cn[None, :] - 2 * xc @ C.T, dim=1)
    return out

def final_train_mse(Xd, Xnp, Cnp):
    idx = assign(Xd, torch.from_numpy(Cnp).to(dev)).cpu().numpy()
    return mse(Xnp, Cnp, idx)

# ---------------- Lloyd ----------------
def lloyd(Xd, Xnp, Xte, k, iters, seed, eval_at=(), log=None):
    rng = np.random.default_rng(seed); n, d = Xnp.shape
    C = Xnp[rng.choice(n, k, replace=False)].astype(np.float64)
    up, dn = split_pattern(d, 1/1024)  # faiss EPS
    res = {}
    for it in range(1, iters + 1):
        a = assign(Xd, torch.from_numpy(C.astype(np.float32)).to(dev))
        cnt = torch.zeros(k, device=dev).index_add_(0, a, torch.ones(n, device=dev)).cpu().numpy().astype(np.float64)
        S = torch.zeros(k, d, device=dev).index_add_(0, a, Xd).cpu().numpy().astype(np.float64)
        nz = cnt > 0; C[nz] = S[nz] / cnt[nz, None]
        # faiss-style empty-cluster split
        for j in np.where(cnt == 0)[0]:
            p = np.maximum(cnt - 1, 0); p = p / p.sum()
            jj = rng.choice(k, p=p)
            C[j] = C[jj] * up; C[jj] = C[jj] * dn
            cnt[j] = cnt[jj] / 2; cnt[jj] -= cnt[j]
        if it in eval_at:
            Cf = C.astype(np.float32)
            res[it] = dict(train=final_train_mse(Xd, Xnp, Cf), test=mse(Xte, Cf), empty_fixed=int((cnt==0).sum()))
            if log: log(f"  lloyd seed{seed} it{it}: {res[it]}")
    return res

# ---------------- e-means ----------------
def emeans(Xd, Xnp, Xte, k, T, seed, revival="split", finalize="lastmean", B=None, a0=None, cosine=True, tau=0.1):
    rng = np.random.default_rng(seed); g = torch.Generator(device="cpu").manual_seed(seed)
    n, d = Xnp.shape
    if B is None or a0 is None:
        B_, a0_ = laws(n, k, T); B = B or B_; a0 = a0 or a0_
    M = T * (n // B)
    C0 = torch.from_numpy(Xnp[rng.choice(n, k, replace=False)]).to(dev)
    N = torch.full((k,), 1e-10, device=dev)
    Sm = N[:, None] * C0
    up, dn = (torch.from_numpy(v).float().to(dev) for v in split_pattern(d, 2**-7))  # BF16 eps
    C = C0.clone()
    t = 0; last_a = torch.empty(n, dtype=torch.int64, device=dev); ones = torch.ones(B, device=dev)
    for ep in range(T):
        perm = torch.randperm(n, generator=g).to(dev)
        for bi in range(n // B):
            ids = perm[bi*B:(bi+1)*B]; xb = Xd[ids]
            alpha = a0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / M))) if cosine else a0
            gam = 1 - alpha; t += 1
            a = assign(xb, C)
            if ep == T - 1: last_a[ids] = a
            nj = torch.zeros(k, device=dev).index_add_(0, a, ones)
            sj = torch.zeros(k, d, device=dev).index_add_(0, a, xb)
            N = gam * N + alpha * nj; Sm = gam * Sm + alpha * sj
            C = Sm / N[:, None]
            if t % 5 == 0 and revival != "off":
                dead = torch.where(N < tau * N.mean())[0]
                if len(dead):
                    if revival == "split":
                        Nc = N.cpu().numpy().astype(np.float64)
                        alive = np.setdiff1d(np.arange(k), dead.cpu().numpy())
                        p = Nc[alive] / Nc[alive].sum()
                        donors = torch.from_numpy(rng.choice(alive, size=len(dead), replace=False, p=p)).to(dev)
                        cd = C[donors]
                        half = N[donors] / 2
                        C[dead] = cd * up; C[donors] = cd * dn
                        N[dead] = half; N[donors] = half
                        Sm[dead] = C[dead] * half[:, None]; Sm[donors] = C[donors] * half[:, None]
                    else:  # resample from batch (VQ-VAE style)
                        r = xb[torch.randint(0, B, (len(dead),), generator=g).to(dev)]
                        C[dead] = r; Sm[dead] = r * N[dead][:, None]
    if n % B:  # points never seen in last epoch's batches: assign with final centroids
        rest = perm[(n // B) * B:]; last_a[rest] = assign(Xd[rest], C)
    Ce = C.clone()
    if finalize == "lastmean":
        cnt = torch.zeros(k, device=dev).index_add_(0, last_a, torch.ones(n, device=dev))
        Ssum = torch.zeros(k, d, device=dev).index_add_(0, last_a, Xd)
        nz = cnt > 0; Ce[nz] = Ssum[nz] / cnt[nz, None]
    Cf = Ce.cpu().numpy()
    Cnt = torch.zeros(k, device=dev).index_add_(0, assign(Xd, Ce), torch.ones(n, device=dev)).cpu().numpy()
    return dict(B=B, a0=round(a0, 4), M=M, train=final_train_mse(Xd, Xnp, Cf), test=mse(Xte, Cf), dead_lt10=int((Cnt < 10).sum()))
