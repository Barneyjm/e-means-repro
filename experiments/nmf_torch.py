"""NMF (squared error) from sufficient statistics: full-batch HALS and an online version built from the e-means ingredients.

Both use the same kernel: nonnegative least squares by coordinate descent (HALS) on small Gram matrices. Online keeps
decayed statistics G = mean of w w^T (K x K) and C = mean of x^T w (V x K), as in online dictionary learning (Mairal et
al. 2010), and re-solves the components from them after every batch. The e-means recipe adds: large batches and the
cosine step from the paper's formulas, a timed repair of dead components, and a final solve from the undecayed
statistics of the last pass. Runs on any device; the experiments here ran on the CPU (EMEANS_DEVICE=cpu).
"""
import math, time, numpy as np, torch
from common import dev, laws

def hals(W, C, G, sweeps):
    """in place: each row w of W moves towards argmin ||x - w H||^2, w >= 0, given C = X H^T and G = H H^T"""
    d = G.diagonal().clamp(min=1e-12)
    for _ in range(sweeps):
        for k in range(G.shape[0]): W[:, k].add_((C[:, k] - W @ G[:, k]) / d[k]).clamp_(min=0)
    return W

def codes(X, Ht, sweeps=10):
    return hals(torch.zeros(X.shape[0], Ht.shape[1], device=X.device), X @ Ht, Ht.T @ Ht, sweeps)

def error(X, Ht, ch=8192):
    """relative squared reconstruction error of held-out rows, codes solved by 50 cold HALS sweeps"""
    num = den = 0.0
    for i in range(0, X.shape[0], ch):
        x = X[i:i+ch]; W = codes(x, Ht, 50); num += float(((x - W @ Ht.T) ** 2).sum()); den += float((x ** 2).sum())
    return num / den

def init(X, K, seed):
    """scikit-learn's init='random': sqrt(mean(X) / K) x |standard normal|"""
    g = torch.Generator().manual_seed(seed); return (math.sqrt(float(X.mean()) / K) * torch.randn(X.shape[1], K, generator=g).abs()).to(X.device)

def batch(X, K, iters, seed, hook=None, sweeps=4):
    Ht = init(X, K, seed); W = None; spent = 0.0
    for it in range(iters):
        t0 = time.time(); C = X @ Ht; G = Ht.T @ Ht
        W = codes(X, Ht) if W is None else hals(W, C, G, sweeps)
        hals(Ht, X.T @ W, W.T @ W, sweeps); spent += time.time() - t0
        if hook: hook(it + 1, spent, Ht)
    return Ht, spent

def repair(Ht, C, G, rng, tau, delta):
    """a component is dead when its share of the reconstruction energy, G_kk ||h_k||^2, is below tau x the mean; it takes
    half of a large component (picked in proportion to energy), the two halves pushed apart by a random +-delta pattern"""
    e = (G.diagonal() * (Ht ** 2).sum(0)).cpu().numpy().astype(np.float64); dm = e < tau * e.mean()
    if not dm.any(): return 0
    dead = np.where(dm)[0]; alive = np.where(~dm)[0]
    if len(dead) > len(alive): dead = dead[np.argsort(e[dead])[:len(alive)]]
    donors = rng.choice(alive, size=len(dead), replace=False, p=e[alive] / e[alive].sum())
    for k, j in zip(dead.tolist(), donors.tolist()):
        s = torch.from_numpy(rng.choice([-1.0, 1.0], size=Ht.shape[0]).astype(np.float32)).to(Ht.device) * delta
        Ht[:, k] = Ht[:, j] * (1 + s); Ht[:, j] *= (1 - s); C[:, j] /= 2; C[:, k] = C[:, j]
        G[j, :] /= 2; G[:, j] /= 2; G[k, :] = G[j, :]; G[:, k] = G[:, j]; G[k, k] = G[j, j]
    return len(dead)

def online(X, K, T, seed, B=None, a0=None, schedule="cosine", kappa=0.5, tau0=1.0, revive=True, tau=0.1, delta=0.5, every=5,
           final="last_epoch", sweeps=4, final_sweeps=10):
    """returns H^T (V x K), seconds, info. schedule: cosine (a0 -> 0.1 a0) | constant | power (step = (tau0 + t)^-kappa,
    t = 0, 1, ...; kappa = 1, tau0 = 1 is the plain running average of all batches so far)"""
    rng = np.random.default_rng(seed); g = torch.Generator().manual_seed(seed); n, V = X.shape
    Bl, al = laws(n, K, T); B = min(B or Bl, n); a0 = a0 or al; nb = n // B; M = T * nb
    Ht = init(X, K, seed); G = torch.zeros(K, K, device=X.device); C = torch.zeros(V, K, device=X.device); GL = torch.zeros_like(G); CL = torch.zeros_like(C); t = rev = 0
    t0 = time.time()
    for ep in range(T):
        perm = torch.randperm(n, generator=g).to(X.device)
        for bi in range(nb):
            x = X[perm[bi*B:(bi+1)*B]]
            step = a0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / M))) if schedule == "cosine" else a0 if schedule == "constant" else (tau0 + t) ** -kappa
            if t == 0: step = 1.0                                               # the statistics start empty
            t += 1; W = codes(x, Ht); g2 = W.T @ W / len(x); c2 = x.T @ W / len(x)
            if ep == T - 1 and final == "last_epoch": GL += g2; CL += c2
            G.mul_(1 - step).add_(g2, alpha=step); C.mul_(1 - step).add_(c2, alpha=step)
            if revive and t % every == 0: rev += repair(Ht, C, G, rng, tau, delta)
            hals(Ht, C, G, sweeps)
    if final == "last_epoch": hals(Ht, CL, GL, final_sweeps)
    return Ht, time.time() - t0, dict(B=B, a0=round(a0, 3), updates=M, revived=rev)

def health(Ht, X, tau=0.1):
    W = codes(X[:8192], Ht); e = (W ** 2).mean(0) * (Ht ** 2).sum(0)
    return dict(dead=int((e < tau * e.mean()).sum()), zero=int((Ht.sum(0) == 0).sum()))
