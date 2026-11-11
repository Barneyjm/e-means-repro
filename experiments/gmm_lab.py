"""Online EM for diagonal Gaussian mixtures (gmm_torch.online_em) with two old ideas as options. With the defaults
it is the same algorithm as gmm_torch.online_em.

  beta0    deterministic-annealing EM (Ueda & Nakano 1998; Rose 1990): responsibilities are softmax(beta * log p),
           with beta rising from beta0 to 1 over the first anneal_frac of the run (flatter, less committed early on)
  wtemp0   mixture weights enter the E-step as w^g, g rising from wtemp0 to 1: an equal-prior start that removes the
           rich-get-richer pull of large components (the soft-assignment cousin of the conscience in emeans_lab)

Usage: python gmm_lab.py <data dir> <log file> <dataset: sift64|synth|skewed> <K> <seeds> <variants file>
Variants file: `name | key=value, ...` with keys T, Bx, ax (multiples of the formula's batch size and step) and the
options above."""
import sys, os, json, math, time, numpy as np, torch
from common import dev, sync, laws
from gmm_torch import REG, CH, LOG2PI, to_params, init, cpu, loglik

def stats_x(X, X2, w, mu, var, beta=1.0, g=1.0):
    K, d = mu.shape; iv = 1 / var
    const = g * torch.log(w) - 0.5 * (torch.log(var).sum(1) + (mu * mu * iv).sum(1) + d * LOG2PI)
    N = torch.zeros(K, device=X.device); S = torch.zeros(K, d, device=X.device); Q = torch.zeros_like(S)
    for i in range(0, X.shape[0], CH):
        L = torch.addmm(const[None, :].expand(min(CH, X.shape[0] - i), -1), X2[i:i+CH], iv.T, alpha=-0.5).addmm_(X[i:i+CH], (mu * iv).T)
        R = torch.softmax(L * beta if beta != 1.0 else L, 1); N += R.sum(0); S += R.T @ X[i:i+CH]; Q += R.T @ X2[i:i+CH]
    return N, S, Q

def online_em_x(Xd, X2d, K, T, seed, B=None, a0=None, tau=0.1, split_size=0.5, beta0=1.0, wtemp0=1.0, anneal_frac=0.5):
    n, d = Xd.shape; g_ = torch.Generator().manual_seed(seed); Bl, al = laws(n, K, T); B = B or Bl; a0 = a0 or al
    t0 = time.time(); w, mu, var, rng = init(Xd, K, seed)
    N = torch.full((K,), 1e-10, device=dev); S = N[:, None] * mu; Q = N[:, None] * (var - REG + mu * mu)
    sgn = torch.where(torch.arange(d, device=dev) % 2 == 0, 1.0, -1.0); nb = n // B; M = T * nb; t = 0; revived = 0
    fN = torch.zeros(K, device=dev); fS = torch.zeros(K, d, device=dev); fQ = torch.zeros(K, d, device=dev)
    for ep in range(T):
        perm = torch.randperm(n, generator=g_).to(dev)
        for bi in range(nb):
            ids = perm[bi*B:(bi+1)*B]; u = t / M; alpha = a0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * u))); t += 1
            r = min(u / anneal_frac, 1.0); beta = beta0 + (1 - beta0) * r; gw = wtemp0 + (1 - wtemp0) * r
            bN, bS, bQ = stats_x(Xd[ids], X2d[ids], w, mu, var, beta, gw)
            if ep == T - 1: fN += bN; fS += bS; fQ += bQ
            N.mul_(1 - alpha).add_(bN, alpha=alpha); S.mul_(1 - alpha).add_(bS, alpha=alpha); Q.mul_(1 - alpha).add_(bQ, alpha=alpha)
            if tau > 0 and t % 5 == 0:
                deadm = N < tau * N.mean()
                if bool(deadm.any()):
                    Nc = N.cpu().numpy().astype(np.float64); dm = deadm.cpu().numpy(); dead = np.where(dm)[0]; alive = np.where(~dm)[0]
                    donors = rng.choice(alive, size=min(len(dead), len(alive)), replace=False, p=Nc[alive] / Nc[alive].sum())
                    dead = torch.from_numpy(dead[:len(donors)]).to(dev); donors = torch.from_numpy(donors).to(dev)
                    _, m_, v_ = to_params(N[donors], S[donors], Q[donors]); half = N[donors] / 2; off = split_size * v_.sqrt() * sgn
                    for idx, m2 in ((dead, m_ + off), (donors, m_ - off)):
                        N[idx] = half; S[idx] = half[:, None] * m2; Q[idx] = half[:, None] * (v_ - REG + m2 * m2)
                    revived += len(dead)
            w, mu, var = to_params(N, S, Q)
    if n % B:
        rest = perm[nb*B:]; bN, bS, bQ = stats_x(Xd[rest], X2d[rest], w, mu, var); fN += bN; fS += bS; fQ += bQ
    keep = fN > 1e-8 * fN.sum()
    w, mu, var = to_params(torch.where(keep, fN, N * 0), torch.where(keep[:, None], fS, S * 0), torch.where(keep[:, None], fQ, Q * 0)); sync()
    return cpu(w, mu, var), time.time() - t0, dict(B=B, a0=round(a0, 3), updates=M, revived=revived)

if __name__ == "__main__":
    from run_gmm import sift64, synth
    from run_lab import parse
    D, outp, ds, K = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]); seeds = [int(s) for s in sys.argv[5].split(",")]
    variants = [parse(l) for l in open(sys.argv[6]) if l.strip() and not l.startswith("#")]
    X, H, truth = sift64(D) if ds == "sift64" else synth(1.0, 10) if ds == "synth" else synth(1.5, 1)
    Xd = torch.from_numpy(X).to(dev); X2d = Xd * Xd
    done = {(r["ds"], r["K"], r["v"], r["seed"]) for r in map(json.loads, open(outp))} if os.path.exists(outp) else set()
    f = open(outp, "a"); online_em_x(Xd, X2d, K, 1, 99, beta0=0.5)
    for seed in seeds:
        for name, kw in variants:
            if (ds, K, name, seed) in done: continue
            kw = dict(kw); T = kw.pop("T", 3); Bl, al = laws(len(X), K, T)
            if "Bx" in kw: kw["B"] = max(256, int(Bl * kw.pop("Bx")))
            if "ax" in kw: kw["a0"] = min(al * kw.pop("ax"), 1.0)
            p, dt, info = online_em_x(Xd, X2d, K, T, seed, **kw)
            s = json.dumps(dict(ds=ds, K=K, v=name, seed=seed, T=T, time=round(dt, 2), test=round(loglik(H, *p), 5), n_small=int((p[0] * K < 0.1).sum()), **info))
            print(s, flush=True); f.write(s + "\n"); f.flush()
