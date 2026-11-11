"""Explicit-feedback ALS (ALS-WR ridge: lam * number of ratings) and the e-means recipe applied to it. Same kernels for both.

Full ALS, per sweep: all users solved exactly from the item factors, then all items from the user factors.
Online: mini-batches of users. Per batch: solve those users exactly against the current item factors (user factors are
never stored, like assignments in k-means), add their ratings to per-item decayed normal-equation totals
(A_i = sum p p^T, b_i = sum r p, c_i = count) and re-solve the touched items, q_i = (A_i + lam c_i I)^-1 b_i.
The solve does not change when (A, b, c) are scaled together, so untouched items keep their factor and the totals need
no rescaling to full-data size. Step: cosine a0 -> 0.1 a0. End: items re-solved from the plain totals of the last pass,
then all users solved once (needed for prediction in any case; counted in the time).
"""
import math, time, numpy as np, torch
from common import dev, sync, laws

BUDGET, SPREAD = 3_000_000, 3.0          # normal-equation kernel: padded ratings per chunk, longest / shortest list in a chunk

class Side:
    """ratings grouped by one side: entity e owns rows start[e] : start[e] + cnt[e] of (oth, r)"""
    def __init__(self, key, oth, r, nk):
        o = np.argsort(key, kind="stable"); cnt = np.bincount(key, minlength=nk)
        self.oth, self.r = torch.from_numpy(oth[o]).to(dev), torch.from_numpy(r[o]).to(dev)
        self.cnt = torch.from_numpy(cnt).to(dev); self.start = torch.from_numpy(np.cumsum(cnt) - cnt).to(dev)

class Data:
    """<path>.npy rows (user id, item id, rating); a random `holdout` share of ratings is the test set; ratings are centred"""
    def __init__(self, path, seed, holdout=0.1):
        a = np.load(path); te = np.random.default_rng(1000 + seed).random(len(a)) < holdout
        u = np.unique(a[:, 0].astype(np.int64), return_inverse=True)[1]; i = np.unique(a[:, 1].astype(np.int64), return_inverse=True)[1]
        self.nu, self.ni = int(u.max()) + 1, int(i.max()) + 1; self.mean = float(a[~te, 2].mean()); r = (a[:, 2] - self.mean).astype(np.float32)
        self.u, self.i, self.r, self.te = u, i, r, te
        self.tu, self.ti, self.tr = (torch.from_numpy(x[te]).to(dev) for x in (u, i, r))
        self.U = Side(u[~te], i[~te], r[~te], self.nu); self.I = Side(i[~te], u[~te], r[~te], self.ni)

def neq(F, oth, r, start, cnt):
    """A = sum f f^T and b = sum r f over each entity's rows start[e] : start[e] + cnt[e] of (oth, r); f = F[oth].
    Entities are sorted by length and padded in chunks of similar length, so the work is one gather and one bmm per chunk."""
    m, K = len(cnt), F.shape[1]; A = torch.zeros(m, K, K, device=dev); b = torch.zeros(m, K, device=dev)
    order = torch.argsort(cnt); c = cnt[order].cpu().numpy(); lo = int(np.searchsorted(c, 1)); last = len(oth) - 1
    while lo < m:
        hi = min(lo + max(1, int(np.searchsorted(np.arange(1, m - lo + 1) * c[lo:], BUDGET, side="right"))), int(np.searchsorted(c, SPREAD * c[lo], side="right")))
        e = order[lo:hi]; ar = torch.arange(int(c[hi - 1]), device=dev)
        idx = (start[e][:, None] + ar[None]).clamp_(max=last); G = F[oth[idx]] * (ar[None] < cnt[e][:, None])[..., None]
        Gt = G.transpose(1, 2); A[e] = Gt @ G; b[e] = (Gt @ r[idx][..., None])[..., 0]; lo = hi
    return A, b

def solve(A, b, ridge):
    A = A + ridge[:, None, None] * torch.eye(A.shape[1], device=dev)
    return torch.cholesky_solve(b[..., None], torch.linalg.cholesky(A))[..., 0]

def users(d, Q, lam, ids=None):
    U = d.U; st, cn = (U.start, U.cnt) if ids is None else (U.start[ids], U.cnt[ids])
    return solve(*neq(Q, U.oth, U.r, st, cn), lam * cn.clamp(min=1).float())

def rmse(d, P, Q): return float(((P[d.tu] * Q[d.ti]).sum(1) - d.tr).pow(2).mean().sqrt())

def objective(d, P, Q, lam, ch=2_000_000):
    """training RMSE and the ALS-WR objective per training rating: (squared error + lam (sum n_u |p_u|^2 + sum n_i |q_i|^2)) / n"""
    U = d.U; u = torch.repeat_interleave(torch.arange(d.nu, device=dev), U.cnt); n = len(u)
    sse = sum(float(((P[u[i:i+ch]] * Q[U.oth[i:i+ch]]).sum(1) - U.r[i:i+ch]).pow(2).sum()) for i in range(0, n, ch))
    pen = float((U.cnt * P.pow(2).sum(1)).sum() + (d.I.cnt * Q.pow(2).sum(1)).sum())
    return (sse / n) ** 0.5, (sse + lam * pen) / n

def init(d, K, seed):
    g = torch.Generator().manual_seed(seed); return (0.1 * torch.randn(d.ni, K, generator=g)).to(dev), g

def als(d, K, sweeps, seed, lam, hook=None):
    """hook(sweep, seconds so far, P, Q) is called after every sweep; its own time is not counted"""
    Q, _ = init(d, K, seed); I = d.I; sync(); t0 = time.time(); spent = 0.0
    for it in range(sweeps):
        P = users(d, Q, lam); Q = solve(*neq(P, I.oth, I.r, I.start, I.cnt), lam * I.cnt.clamp(min=1).float())
        if hook: sync(); spent += time.time() - t0; hook(it + 1, spent, P, Q); t0 = time.time()
    sync(); return P, Q, spent + time.time() - t0

def online(d, K, T, seed, lam, B=None, a0=None, schedule="cosine", kappa=0.6, decay="all", final="last_epoch", resolve=1):
    """schedule: cosine | constant | stepwise ((t+2)^-kappa). decay: all (every item's totals fade each batch, as in e-means)
    | touched (only items rated in the batch fade). final: last_epoch | none. resolve: re-solve items every `resolve` batches.
    Returns P, Q, seconds, info."""
    Q, g = init(d, K, seed); n, U = d.nu, d.U; Bl, al = laws(n, d.ni, T); B = B or Bl; a0 = a0 or al; nb = max(n // B, 1); M = T * nb
    sync(); t0 = time.time()
    A = torch.zeros(d.ni, K, K, device=dev); b = torch.zeros(d.ni, K, device=dev); c = torch.zeros(d.ni, device=dev)
    keep = final == "last_epoch"; LA, Lb = (torch.zeros_like(A), torch.zeros_like(b)) if keep else (None, None)
    t = 0; dirty = torch.zeros(d.ni, dtype=torch.bool, device=dev); big = 0.0
    for ep in range(T):
        perm = torch.randperm(n, generator=g).to(dev)
        for bi in range(nb):
            ids = perm[bi * B:(bi + 1) * B] if bi < nb - 1 else perm[bi * B:]
            alpha = a0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / M))) if schedule == "cosine" else a0 if schedule == "constant" else min(1.0, (t + 2) ** -kappa)
            t += 1; st, cn = U.start[ids], U.cnt[ids]
            Pb = solve(*neq(Q, U.oth, U.r, st, cn), lam * cn.clamp(min=1).float())
            lu = torch.repeat_interleave(torch.arange(len(ids), device=dev), cn)                     # the batch's ratings ...
            pos = torch.arange(len(lu), device=dev) + torch.repeat_interleave(st - (torch.cumsum(cn, 0) - cn), cn)
            its, o = torch.sort(U.oth[pos]); items, ic = torch.unique_consecutive(its, return_counts=True)   # ... regrouped by item
            Ab, bb = neq(Pb, lu[o], U.r[pos][o], torch.cumsum(ic, 0) - ic, ic)
            if decay == "all": A.mul_(1 - alpha).index_add_(0, items, Ab, alpha=alpha); b.mul_(1 - alpha).index_add_(0, items, bb, alpha=alpha); c.mul_(1 - alpha).index_add_(0, items, ic.float(), alpha=alpha)
            else: A[items] = (1 - alpha) * A[items] + alpha * Ab; b[items] = (1 - alpha) * b[items] + alpha * bb; c[items] = (1 - alpha) * c[items] + alpha * ic
            if keep and ep == T - 1: LA.index_add_(0, items, Ab); Lb.index_add_(0, items, bb)
            dirty[items] = True
            if t % resolve == 0 or t == M:
                it = torch.where(dirty & (c > 1e-3))[0]; Q[it] = solve(A[it], b[it], lam * c[it]); dirty[:] = False   # a step of 1 wipes the totals of items absent from that batch
    if keep: nz = torch.where(d.I.cnt > 0)[0]; Q[nz] = solve(LA[nz], Lb[nz], lam * d.I.cnt[nz].float())
    Q[d.I.cnt == 0] = 0                                    # items with no training rating: predict the mean, as full ALS does
    P = users(d, Q, lam); sync()
    return P, Q, time.time() - t0, dict(B=B, a0=round(a0, 4), updates=M)
