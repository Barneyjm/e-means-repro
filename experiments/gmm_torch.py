"""Diagonal Gaussian mixtures: standard full-pass EM, and online EM built with the e-means recipe.

Both use the same E-step kernel, so a comparison is about the algorithm, not the arithmetic.
State for the online version is three decayed running totals per component (weight N, sum S, sum of squares Q),
the Gaussian-mixture analogue of e-means' (count, sum).
"""
import math, time, numpy as np, torch
from common import dev, sync, laws

REG = 1e-6          # added to every variance, same as scikit-learn's reg_covar default
CH = 32768
LOG2PI = math.log(2 * math.pi)

def to_params(N, S, Q):
    Ns = N.clamp_min(1e-30)
    mu = S / Ns[:, None]
    return N / N.sum(), mu, (Q / Ns[:, None] - mu * mu).clamp_min(0) + REG

def estep(X, X2, w, mu, var):
    """responsibilities (rows × K) and per-row log-likelihood, from two matrix products"""
    iv = 1 / var
    const = torch.log(w) - 0.5 * (torch.log(var).sum(1) + (mu * mu * iv).sum(1) + X.shape[1] * LOG2PI)
    L = torch.addmm(const[None, :].expand(X.shape[0], -1), X2, iv.T, alpha=-0.5).addmm_(X, (mu * iv).T)
    ll = torch.logsumexp(L, 1)
    return torch.exp(L - ll[:, None]), ll

def stats(X, X2, w, mu, var):
    """sufficient statistics of a block of rows under the current model, chunked"""
    K, d = mu.shape
    N = torch.zeros(K, device=X.device, dtype=X.dtype); S = torch.zeros(K, d, device=X.device, dtype=X.dtype); Q = torch.zeros_like(S)
    ll = torch.zeros((), device=X.device, dtype=X.dtype)
    for i in range(0, X.shape[0], CH):
        R, l = estep(X[i:i+CH], X2[i:i+CH], w, mu, var)
        N += R.sum(0); S += R.T @ X[i:i+CH]; Q += R.T @ X2[i:i+CH]; ll += l.sum()
    return N, S, Q, ll

def loglik(Xnp, w, mu, var):
    """exact float64 mean log-likelihood per row on CPU (the score everything is compared on)"""
    X = torch.from_numpy(Xnp).double(); p = [torch.as_tensor(np.asarray(v)).double() for v in (w, mu, var)]
    return float(sum(estep(X[i:i+CH], X[i:i+CH] ** 2, *p)[1].sum() for i in range(0, len(X), CH)) / len(X))

def init(Xd, K, seed):
    rng = np.random.default_rng(seed)
    idx = torch.from_numpy(rng.choice(Xd.shape[0], K, replace=False)).to(Xd.device)
    return torch.full((K,), 1.0 / K, device=Xd.device), Xd[idx].clone(), Xd.var(0)[None, :].expand(K, -1).clone() + REG, rng

def cpu(*p): return [v.detach().cpu().numpy() for v in p]

def em(Xd, X2d, K, iters, seed, hook=None):
    """Standard EM: one full pass over the data per update. hook(iteration, seconds so far, params) is not timed."""
    w, mu, var, _ = init(Xd, K, seed); spent = 0.0
    for it in range(1, iters + 1):
        t0 = time.time()
        w, mu, var = to_params(*stats(Xd, X2d, w, mu, var)[:3]); sync()
        spent += time.time() - t0
        if hook: hook(it, spent, cpu(w, mu, var))
    return cpu(w, mu, var), spent

def online_em(Xd, X2d, K, T, seed, B=None, a0=None, schedule="cosine", kappa=0.6, revive=True, tau=0.1,
              split="sigma", split_size=0.5, final="last_epoch"):
    """Online EM with the e-means ingredients, each switchable:
    schedule  'cosine' (paper), 'const', or 'stepwise' = (t+2)^-kappa, the classic online-EM step rule
    revive    every 5 updates, components with N < tau*mean(N) are re-seeded by splitting a heavy component
    split     'sigma': children at mean ± split_size*std;  'delta': the paper's mean*(1 ± 2^-7)
    final     'last_epoch': parameters from plain (undecayed) totals of the last pass; 'none': the running totals
    """
    n, d = Xd.shape; g = torch.Generator().manual_seed(seed)
    Bl, al = laws(n, K, T); B = B or Bl; a0 = a0 or al
    t0 = time.time()
    w, mu, var, rng = init(Xd, K, seed)
    N = torch.full((K,), 1e-10, device=dev); S = N[:, None] * mu; Q = N[:, None] * (var - REG + mu * mu)
    sgn = torch.where(torch.arange(d, device=dev) % 2 == 0, 1.0, -1.0)
    nb = n // B; M = T * nb; t = 0; revived = 0
    fN = torch.zeros(K, device=dev); fS = torch.zeros(K, d, device=dev); fQ = torch.zeros(K, d, device=dev)
    for ep in range(T):
        perm = torch.randperm(n, generator=g).to(dev)
        for bi in range(nb):
            ids = perm[bi*B:(bi+1)*B]
            if schedule == "cosine": alpha = a0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / M)))
            elif schedule == "const": alpha = a0
            else: alpha = (t + 2) ** -kappa
            t += 1
            bN, bS, bQ, _ = stats(Xd[ids], X2d[ids], w, mu, var)
            if ep == T - 1: fN += bN; fS += bS; fQ += bQ
            N.mul_(1 - alpha).add_(bN, alpha=alpha); S.mul_(1 - alpha).add_(bS, alpha=alpha); Q.mul_(1 - alpha).add_(bQ, alpha=alpha)
            if revive and t % 5 == 0:
                deadm = N < tau * N.mean()
                if bool(deadm.any()):
                    Nc = N.cpu().numpy().astype(np.float64); dm = deadm.cpu().numpy()
                    dead = np.where(dm)[0]; alive = np.where(~dm)[0]
                    donors = rng.choice(alive, size=min(len(dead), len(alive)), replace=False, p=Nc[alive] / Nc[alive].sum())
                    dead = torch.from_numpy(dead[:len(donors)]).to(dev); donors = torch.from_numpy(donors).to(dev)
                    _, m_, v_ = to_params(N[donors], S[donors], Q[donors]); half = N[donors] / 2
                    off = split_size * v_.sqrt() * sgn if split == "sigma" else m_ * (2 ** -7) * sgn
                    for idx, m2 in ((dead, m_ + off), (donors, m_ - off)):
                        N[idx] = half; S[idx] = half[:, None] * m2; Q[idx] = half[:, None] * (v_ - REG + m2 * m2)
                    revived += len(dead)
            w, mu, var = to_params(N, S, Q)
    if final == "last_epoch":
        if n % B:
            rest = perm[nb*B:]; bN, bS, bQ, _ = stats(Xd[rest], X2d[rest], w, mu, var); fN += bN; fS += bS; fQ += bQ
        keep = fN > 1e-8 * fN.sum()                 # a component that saw nothing in the last pass keeps its running value
        N = torch.where(keep, fN, N * 0); S = torch.where(keep[:, None], fS, S * 0); Q = torch.where(keep[:, None], fQ, Q * 0)
        w, mu, var = to_params(N, S, Q)
    sync()
    return cpu(w, mu, var), time.time() - t0, dict(B=B, a0=round(a0, 3), updates=M, revived=revived)
