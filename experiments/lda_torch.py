"""Variational LDA on the GPU: full-batch EM and an online version built from the e-means ingredients.

Same E-step kernel for both (dense batch x vocabulary blocks, cold start, the update of Hoffman, Blei & Bach 2010 /
scikit-learn). The online version keeps decayed topic-word totals S (lambda = eta + S), so one update is
S <- (1 - step) S + step * (n / batch) * batch totals, which is Hoffman's rule with rho = step. What changes with the
e-means recipe: large batches from the paper's formula, the cosine step curve, a timed repair of dead and duplicate
topics, and a final exact M-step from the undecayed totals of the last pass.
"""
import math, time, numpy as np, torch
from common import dev, sync, laws
CH = 4096; EPS = float(np.finfo(np.float64).eps)

def load(path, n=None):
    """-> train counts as a dense uint8 (n, V) tensor on the device, held-out observed / scored halves as dense float32"""
    z = np.load(path); out = []
    for nm in ("X", "Ho", "Ht"):
        shape = tuple(z[nm + "_shape"]); ip = z[nm + "_indptr"]; rows = shape[0] if nm != "X" or n is None else min(n, shape[0])
        nnz = ip[rows]; r = torch.from_numpy(np.repeat(np.arange(rows), np.diff(ip[:rows + 1]))).to(dev); c = torch.from_numpy(z[nm + "_indices"][:nnz].astype(np.int64)).to(dev)
        v = torch.from_numpy(z[nm + "_data"][:nnz]); D = torch.zeros(rows, shape[1], dtype=torch.uint8 if nm == "X" else torch.float32, device=dev)
        D[r, c] = (v.clamp(max=255).to(torch.uint8) if nm == "X" else v).to(dev); out.append(D)
    return out

def dexp(a): return torch.exp(torch.digamma(a) - torch.digamma(a.sum(1, keepdim=True)))

def estep(X, eb, alpha, tol=1e-3, max_iter=100, strict=False, need_ss=True, gamma0=None):
    """X (B, V) counts (uint8 or float), eb = exp(E[log beta]) (K, V). Cold start. Stops when the mean change of gamma,
    averaged over the block, is below tol; strict=True instead freezes each document when its own mean change is below
    tol (scikit-learn); gamma0 replaces the start at 1. The count / phinorm ratio is only ever written at the non-zero counts of the block."""
    lin = X.reshape(-1).nonzero().squeeze(1); xv = X.reshape(-1)[lin].float(); R = torch.zeros(X.shape, device=X.device); Rf = R.view(-1)
    gamma = torch.ones(X.shape[0], eb.shape[0], device=X.device) if gamma0 is None else gamma0; et = dexp(gamma); ebT = eb.T.contiguous()
    active = torch.ones(X.shape[0], dtype=torch.bool, device=X.device); it = 0
    for it in range(1, max_iter + 1):
        Rf[lin] = xv / ((et @ eb).view(-1)[lin] + EPS); new = alpha + et * (R @ ebT); change = (new - gamma).abs().mean(1)
        if strict:
            gamma = torch.where(active[:, None], new, gamma); active = active & (change >= tol)
            if it % 5 == 0 and not bool(active.any()): break
        else:
            gamma = new
            if it % 5 == 0 and float(change.mean()) < tol: break
        et = dexp(gamma)
    et = dexp(gamma)
    if not need_ss: return gamma, None, it
    Rf[lin] = xv / ((et @ eb).view(-1)[lin] + EPS)
    return gamma, eb * (et.T @ R), it

def totals(Xu, ids, eb, alpha, **kw):
    """summed batch totals over the documents `ids` (a tensor of row numbers, or a slice)"""
    ss = torch.zeros_like(eb); n = Xu.shape[0] if isinstance(ids, slice) else len(ids); its = 0
    for i in range(0, n, CH):
        _, s, it = estep(Xu[i:i+CH] if isinstance(ids, slice) else Xu[ids[i:i+CH]], eb, alpha, **kw); ss += s; its += it
    return ss, its

def perplexity(lam, Ho, Ht, alpha):
    """document completion: fit each held-out document's topic mix on its observed half, score the other half"""
    eb = dexp(lam); beta = lam / lam.sum(1, keepdim=True); ll = tok = 0.0
    for i in range(0, Ho.shape[0], CH):
        g, _, _ = estep(Ho[i:i+CH], eb, alpha, tol=1e-4, strict=True, need_ss=False)
        ll += float((Ht[i:i+CH] * torch.log((g / g.sum(1, keepdim=True)) @ beta + 1e-30)).sum(1).cpu().double().sum())
        tok += float(Ht[i:i+CH].sum(1).cpu().double().sum())
    return math.exp(-ll / tok)

def init(K, V, seed, Xu=None, mix=None):
    """scikit-learn's start: lambda ~ Gamma(100, 0.01), from the same generator and in the same order. With mix = s
    (the analogue of starting k-means from data points): topic k = (1 - s) x one random document's word frequencies
    + s x the corpus frequencies, with the same total mass V as the default start."""
    if mix is None: return torch.from_numpy(np.random.RandomState(seed).gamma(100.0, 0.01, (K, V)).astype(np.float32)).to(dev)
    rng = np.random.default_rng(seed); n = Xu.shape[0]; d = Xu[torch.from_numpy(rng.choice(n, K, replace=False)).to(dev)].float()
    u = Xu[torch.from_numpy(rng.choice(n, min(n, 20000), replace=False)).to(dev)].sum(0, dtype=torch.float32); u = u / u.sum()
    return V * ((1 - mix) * d / d.sum(1, keepdim=True) + mix * u[None]) + 1e-3

def em(Xu, K, iters, seed, alpha, eta, hook=None, mix=None, **kw):
    n, V = Xu.shape; lam = init(K, V, seed, Xu, mix); spent = 0.0
    for it in range(iters):
        sync(); t0 = time.time(); ss, _ = totals(Xu, slice(None), dexp(lam), alpha, **kw); lam = eta + ss; sync(); spent += time.time() - t0
        if hook: hook(it + 1, spent, lam)
    return lam, spent

def repair(S, eta, rng, tau, dup, delta):
    """Topics whose total is below tau x the mean are dead. Of two topics whose word distributions have cosine above
    `dup`, the smaller is folded into the larger and counted as dead. Each dead topic takes half of a large topic (picked
    with probability proportional to its total); the halves are pushed apart by a random +-delta pattern over words."""
    K, V = S.shape; N = S.sum(1); nd = 0
    if dup < 1:
        P = S + eta; P = P / P.norm(dim=1, keepdim=True); G = (P @ P.T).triu(1); pairs = (G > dup).nonzero().cpu().numpy(); Nc = N.cpu().numpy(); gone = set()
        for i, j in pairs:
            if i in gone or j in gone: continue
            a, b = (i, j) if Nc[i] >= Nc[j] else (j, i); S[a] += S[b]; S[b] = 0; gone.add(b); nd += 1
        N = S.sum(1)
    deadm = N < tau * N.mean()
    if not bool(deadm.any()): return 0, nd
    Nc = N.cpu().numpy().astype(np.float64); dm = deadm.cpu().numpy(); dead = np.where(dm)[0]; alive = np.where(~dm)[0]
    if len(dead) > len(alive): dead = dead[np.argsort(Nc[dead])[:len(alive)]]
    donors = rng.choice(alive, size=len(dead), replace=False, p=Nc[alive] / Nc[alive].sum())
    s = torch.from_numpy(rng.choice([-1.0, 1.0], size=(len(dead), V)).astype(np.float32)).to(S.device) * delta
    dead, donors = torch.from_numpy(dead).to(S.device), torch.from_numpy(donors).to(S.device); h = S[donors] / 2
    S[dead] = h * (1 + s); S[donors] = h * (1 - s)
    return len(dead), nd

def online(Xu, K, T, seed, alpha, eta, B=None, a0=None, schedule="cosine", kappa=0.7, tau0=10.0, revive=True, tau=0.1, dup=0.9,
           delta=0.5, every=5, final="last_epoch", mix=None, hook=None, **kw):
    """returns lambda, seconds, info. mix: see init. schedule: cosine (a0 -> 0.1 a0) | constant | hoffman (step = (tau0 + t)^-kappa, t = 1, 2, ...)"""
    rng = np.random.default_rng(seed); g = torch.Generator().manual_seed(seed); n, V = Xu.shape
    Bl, al = laws(n, K, T); B = min(B or Bl, n); a0 = a0 or al; nb = n // B; M = T * nb
    S = init(K, V, seed, Xu, mix) - eta; S.clamp_(min=0); L = torch.zeros_like(S); t = 0; rev = dups = its = 0
    sync(); t0 = time.time()
    for ep in range(T):
        perm = torch.randperm(n, generator=g).to(dev)
        for bi in range(nb):
            ids = perm[bi*B:(bi+1)*B]
            step = a0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / M))) if schedule == "cosine" else a0 if schedule == "constant" else (tau0 + t + 1) ** -kappa
            t += 1; ss, i = totals(Xu, ids, dexp(S + eta), alpha, **kw); its += i
            if ep == T - 1 and final == "last_epoch": L += ss
            S.mul_(1 - step).add_(ss, alpha=step * n / len(ids))
            if revive and t % every == 0: r, d = repair(S, eta, rng, tau, dup, delta); rev += r; dups += d
            if hook: hook(t, S + eta)                                        # diagnostics only: its cost is inside the returned time
    if final == "last_epoch":
        if n % B: ss, _ = totals(Xu, perm[nb*B:], dexp(S + eta), alpha, **kw); L += ss
        S = L
    lam = S + eta; sync()
    return lam, time.time() - t0, dict(B=B, a0=round(a0, 3), updates=M, revived=rev, merged=dups, estep_iters=its)

def health(lam, eta, tau=0.1, dup=0.9):
    """dead topics (total below tau x mean) and duplicate pairs (cosine above dup)"""
    S = lam - eta; N = S.sum(1); P = lam / lam.norm(dim=1, keepdim=True)
    return dict(dead=int((N < tau * N.mean()).sum()), dup_pairs=int(((P @ P.T).triu(1) > dup).sum()), top_share=round(float(N.max() / N.sum()), 4))
