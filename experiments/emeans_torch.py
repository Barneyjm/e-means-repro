"""Optimized e-means / Lloyd (same fused assignment kernel for both). Training time excludes scoring."""
import numpy as np, torch, time, math
from common import fvecs, dev, mse, split_pattern, laws
CH = 16384

class Assigner:
    """nearest centroid via one fused addmm per chunk; half=True uses fp16 on data pre-scaled to |x|<=1"""
    def __init__(self, half=False, scale=1.0): self.half, self.scale = half, scale
    def prep(self, X): return (X * self.scale).half() if self.half else X
    def __call__(self, Xp, C):
        Cc = C * self.scale if self.half else C
        cn = (Cc * Cc).sum(1); Ct = Cc.T.contiguous()
        if self.half: cn, Ct = cn.half(), Ct.half()
        out = [torch.argmin(torch.addmm(cn[None, :].expand(min(CH, Xp.shape[0] - i), -1), Xp[i:i+CH], Ct, alpha=-2), dim=1)
               for i in range(0, Xp.shape[0], CH)]
        return out[0] if len(out) == 1 else torch.cat(out)

def score(Xd, Xnp, Xte, Cf):
    idx = Assigner()(Xd, torch.from_numpy(Cf).to(dev)).cpu().numpy()
    cnt = np.bincount(idx, minlength=len(Cf))
    return dict(train=mse(Xnp, Cf, idx), test=mse(Xte, Cf), dead_lt10=int((cnt < 10).sum()))

def lloyd(Xd, Xnp, k, iters, seed):
    rng = np.random.default_rng(seed); n, d = Xnp.shape; A = Assigner()
    t0 = time.time()
    C = torch.from_numpy(Xnp[rng.choice(n, k, replace=False)]).to(dev)
    up, dn = (torch.from_numpy(v).float().to(dev) for v in split_pattern(d, 1/1024))
    for it in range(iters):
        a = A(Xd, C)
        cnt = torch.bincount(a, minlength=k).float()
        S = torch.zeros(k, d, device=dev).index_add_(0, a, Xd)
        nz = cnt > 0; C[nz] = S[nz] / cnt[nz, None]
        empty = torch.where(~nz)[0].cpu().numpy()
        if len(empty):
            c = cnt.cpu().numpy().astype(np.float64)
            for j in empty:
                p = np.maximum(c - 1, 0); jj = rng.choice(k, p=p / p.sum())
                C[j] = C[jj] * up; C[jj] = C[jj] * dn; c[j] = c[jj] / 2; c[jj] -= c[j]
    Cf = C.cpu().numpy()
    return Cf, time.time() - t0

def emeans(Xd, Xnp, k, T, seed, half=False, tau=0.1, B=None, a0=None):
    rng = np.random.default_rng(seed); g = torch.Generator().manual_seed(seed)
    n, d = Xnp.shape
    Bl, al = laws(n, k, T); B = B or Bl; a0 = a0 or al
    A = Assigner(half, 1.0 / float(np.abs(Xnp[:10000]).max())) if half else Assigner()
    t0 = time.time()
    Xp = A.prep(Xd)
    nb = n // B; M = T * nb
    C = torch.from_numpy(Xnp[rng.choice(n, k, replace=False)]).to(dev)
    N = torch.full((k,), 1e-10, device=dev); Sm = N[:, None] * C
    up, dn = (torch.from_numpy(v).float().to(dev) for v in split_pattern(d, 2**-7))
    last_a = torch.empty(n, dtype=torch.int64, device=dev); t = 0
    for ep in range(T):
        perm = torch.randperm(n, generator=g).to(dev)
        for bi in range(nb):
            ids = perm[bi*B:(bi+1)*B]; xb = Xd[ids]
            alpha = a0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / M))); t += 1
            a = A(Xp[ids] if half else xb, C)
            if ep == T - 1: last_a[ids] = a
            N.mul_(1 - alpha).add_(torch.bincount(a, minlength=k), alpha=alpha)
            Sm.mul_(1 - alpha).add_(torch.zeros(k, d, device=dev).index_add_(0, a, xb), alpha=alpha)
            if t % 5 == 0:
                deadm = N < tau * N.mean()
                if bool(deadm.any()):
                    Nc = N.cpu().numpy().astype(np.float64); dm = deadm.cpu().numpy()
                    dead = np.where(dm)[0]; alive = np.where(~dm)[0]
                    donors = rng.choice(alive, size=len(dead), replace=False, p=Nc[alive] / Nc[alive].sum())
                    dead, donors = torch.from_numpy(dead).to(dev), torch.from_numpy(donors).to(dev)
                    cd = Sm[donors] / N[donors, None]; half_n = N[donors] / 2
                    N[dead] = half_n; N[donors] = half_n
                    Sm[dead] = cd * up * half_n[:, None]; Sm[donors] = cd * dn * half_n[:, None]
            C = Sm / N[:, None]
    if n % B:
        rest = perm[nb * B:]; last_a[rest] = A(Xp[rest] if half else Xd[rest], C)
    cnt = torch.bincount(last_a, minlength=k).float()
    Ssum = torch.zeros(k, d, device=dev).index_add_(0, last_a, Xd)
    nz = cnt > 0; C[nz] = Ssum[nz] / cnt[nz, None]
    Cf = C.cpu().numpy()
    return Cf, time.time() - t0, dict(B=B, a0=round(a0, 3), updates=M)
