"""e-means with switchable extras: old ideas from the vector-quantization and neural-codebook literature that the
paper's recipe does not use. With every option at its default this is exactly `emeans_torch.emeans`
(checked by lab_check.py). Options, and where each idea comes from:

  sched, floor   shape of the step curve (cosine to floor*a0 as in the paper, linear, exp, const)
  revive         what a starved centroid becomes: "split" (paper: half of a donor), "point" (a random vector of the
                 current batch: the VQ-VAE / Jukebox codebook reset), "d2" (a batch vector drawn in proportion to its
                 squared error: k-means++ seeding applied online)
  donor          which centroid gives up half: "count" (paper: biggest) or "error" (largest summed distortion: the
                 equal-distortion principle of Gersho 1979 and LBG-U, Fritzke 1997)
  util_tau       also relocate centroids whose removal would cost little (utility below util_tau * mean): LBG-U
  consc          conscience / frequency-sensitive learning (DeSieno 1988, Ahalt 1990): busy centroids are handicapped
                 when assigning, by consc * mean distortion * (relative size - 1) (or log relative size). The linear form
                 is the assignment step for the objective MSE + consc * mean distortion * imbalance / 2, where imbalance
                 = k * sum(n_j^2) / n^2 is what a search index pays for uneven lists. The strength follows the cosine
                 curve down to consc_end * consc (0 = annealed away, 1 or consc_const = held)
  soft_m         neural-gas style soft start (Martinetz 1993): each vector also pulls its 2nd..m-th nearest centroid
                 with rank-decaying weight, during the first soft_frac of the run
  bgrow          growing batches ("increase the batch size instead of decaying the step", Smith 2018; nested
                 mini-batches, Newling 2016): start at B/bgrow and grow to B over the first half, step scaled alike
  nest           Nesterov-style look-ahead: assign against C + nest * (C - previous C)
  init           "random" (paper) or "pp" (k-means++ D^2 sampling on a subsample, in chunks)
  C0             start from given centroids (used by `ladder`: LBG codebook doubling, Linde-Buzo-Gray 1980)
  polish         finish with this many exact Lloyd iterations (online first, batch last: Bottou & Bengio 1995)
"""
import numpy as np, torch, time, math
from common import dev, split_pattern, laws
CH = 16384

def nearest(X, C, bias=None, top=1):
    """index of the nearest centroid(s) and squared distance minus |x|^2 (plus bias, if given)"""
    cn = (C * C).sum(1); Ct = C.T.contiguous()
    if bias is not None: cn = cn + bias
    I, V = [], []
    for i in range(0, X.shape[0], CH):
        D = torch.addmm(cn[None, :].expand(min(CH, X.shape[0] - i), -1), X[i:i+CH], Ct, alpha=-2)
        v, a = torch.min(D, dim=1) if top == 1 else torch.topk(D, top, dim=1, largest=False)
        I.append(a); V.append(v)
    return (I[0], V[0]) if len(I) == 1 else (torch.cat(I), torch.cat(V))

def seed_pp(Xd, k, rng, per_centroid=8):
    """k-means++ D^2 sampling on a subsample, k/256 centroids per step"""
    n = Xd.shape[0]; m = min(n, per_centroid * k); sub = Xd[torch.from_numpy(rng.choice(n, m, replace=False)).to(dev)]
    sn = (sub * sub).sum(1); ids = [int(rng.integers(m))]; mind = None; step = max(1, k // 256); new = ids
    while True:
        c = sub[torch.tensor(new, device=dev)]; dist = (sn[:, None] + (c * c).sum(1)[None] - 2 * sub @ c.T).min(1).values
        mind = dist if mind is None else torch.minimum(mind, dist)
        if len(ids) >= k: break
        p = mind.clamp_min(0).cpu().double().numpy(); p[ids] = 0
        new = rng.choice(m, min(step, k - len(ids)), replace=False, p=p / p.sum()).tolist(); ids += new
    return sub[torch.tensor(ids, device=dev)].clone()

def lloyd_from(Xd, C, iters, rng, k):
    d = Xd.shape[1]; up, dn = (torch.from_numpy(v).float().to(dev) for v in split_pattern(d, 1/1024))
    for _ in range(iters):
        a, _ = nearest(Xd, C); cnt = torch.bincount(a, minlength=k).float()
        S = torch.zeros(k, d, device=dev).index_add_(0, a, Xd); nz = cnt > 0; C[nz] = S[nz] / cnt[nz, None]
        empty = torch.where(~nz)[0].cpu().numpy()
        if len(empty):
            c = cnt.cpu().numpy().astype(np.float64)
            for j in empty:
                p = np.maximum(c - 1, 0); jj = rng.choice(k, p=p / p.sum())
                C[j] = C[jj] * up; C[jj] = C[jj] * dn; c[j] = c[jj] / 2; c[jj] -= c[j]
    return C

def lloyd_x(Xd, Xnp, k, iters, seed, consc=0.0, init="random"):
    """Lloyd, optionally from a k-means++ start (init="pp") and optionally with the linear size penalty (a naive
    balanced k-means: sizes and mean distortion come from the previous iteration)"""
    rng = np.random.default_rng(seed); n, d = Xd.shape; t0 = time.time(); xn = (Xd * Xd).sum(1)
    C = seed_pp(Xd, k, rng) if init == "pp" else torch.from_numpy(Xnp[rng.choice(n, k, replace=False)]).to(dev); bias = None
    up, dn = (torch.from_numpy(v).float().to(dev) for v in split_pattern(d, 1/1024))
    for _ in range(iters):
        a, v = nearest(Xd, C, bias); cnt = torch.bincount(a, minlength=k).float()
        dbar = float((v - (bias[a] if bias is not None else 0) + xn).clamp_min(0).mean())
        S = torch.zeros(k, d, device=dev).index_add_(0, a, Xd); nz = cnt > 0; C[nz] = S[nz] / cnt[nz, None]
        empty = torch.where(~nz)[0].cpu().numpy()
        if len(empty):
            c = cnt.cpu().numpy().astype(np.float64)
            for j in empty:
                p = np.maximum(c - 1, 0); jj = rng.choice(k, p=p / p.sum())
                C[j] = C[jj] * up; C[jj] = C[jj] * dn; c[j] = c[jj] / 2; c[jj] -= c[j]
            cnt = torch.from_numpy(c).float().to(dev)
        if consc: bias = consc * dbar * (cnt * (k / n) - 1)
    return C.cpu().numpy(), time.time() - t0, {}

def emeans_x(Xd, Xnp, k, T, seed, B=None, a0=None, tau=0.1, sched="cosine", floor=0.1, revive="split", donor="count",
             util_tau=0.0, consc=0.0, consc_const=False, consc_end=0.0, consc_form="linear", soft_m=1, soft_frac=0.2, bgrow=1, nest=0.0, init="random",
             C0=None, polish=0, final="last", every=5):
    rng = np.random.default_rng(seed); g = torch.Generator().manual_seed(seed)
    n, d = Xd.shape; Bl, al = laws(n, k, T); B = B or Bl; a0 = a0 or al
    need_err = donor == "error" or revive == "d2" or consc > 0; top = 2 if util_tau > 0 else 1
    t0 = time.time()
    if C0 is not None: C = C0.clone()
    elif init == "pp": C = seed_pp(Xd, k, rng)
    else: C = torch.from_numpy(Xnp[rng.choice(n, k, replace=False)]).to(dev)
    xn = (Xd * Xd).sum(1) if need_err else None
    N = torch.full((k,), 1e-10, device=dev); Sm = N[:, None] * C
    E = torch.zeros(k, device=dev); U = torch.zeros(k, device=dev); Cprev = C
    up, dn = (torch.from_numpy(v).float().to(dev) for v in split_pattern(d, 2**-7))
    last_a = torch.empty(n, dtype=torch.int64, device=dev); done = 0; total = T * (n // B) * B; t = 0; nrev = 0
    def curve(u):
        if sched == "cosine": return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * u))
        if sched == "linear": return floor + (1 - floor) * (1 - u)
        if sched == "exp": return floor ** u
        return 1.0
    for ep in range(T):
        perm = torch.randperm(n, generator=g).to(dev); pos = 0
        while True:
            u = done / total
            bs = B if bgrow == 1 else max(256, int(B * bgrow ** -(1 - min(2 * u, 1))) // 256 * 256)
            if pos + bs > n: break
            ids = perm[pos:pos+bs]; xb = Xd[ids]; pos += bs; done += bs
            alpha = min(a0 * curve(u) * (bs / B), 1.0); t += 1
            bias = None
            if consc > 0 and t > 1:
                rel = N * (k / N.sum()); end = 1.0 if consc_const else consc_end
                lam = consc * (end + (1 - end) * 0.5 * (1 + math.cos(math.pi * u))) * float(E.sum() / N.sum())
                bias = lam * (rel - 1 if consc_form == "linear" else torch.log(rel.clamp_min(1e-3)))
            Ca = C + nest * (C - Cprev) if nest else C
            soft = soft_m > 1 and u < soft_frac
            a, v = nearest(xb, Ca, bias, soft_m if soft else top)
            a1 = a if a.dim() == 1 else a[:, 0]
            if ep == T - 1: last_a[ids] = a1
            if soft:
                lam = max(1.0 - u / soft_frac, 0.05); w = torch.exp(-torch.arange(soft_m, device=dev) / lam); w = w / w.sum()
                cnt = torch.zeros(k, device=dev); sums = torch.zeros(k, d, device=dev)
                for r in range(soft_m):
                    cnt.add_(torch.bincount(a[:, r], minlength=k), alpha=float(w[r])); sums.index_add_(0, a[:, r], xb * w[r])
            else:
                cnt = torch.bincount(a1, minlength=k); sums = torch.zeros(k, d, device=dev).index_add_(0, a1, xb)
            if alpha >= 1.0: nzb = cnt > 0; N[nzb] = cnt[nzb].float(); Sm[nzb] = sums[nzb]
            else: N.mul_(1 - alpha).add_(cnt, alpha=alpha); Sm.mul_(1 - alpha).add_(sums, alpha=alpha)
            if need_err or top == 2:
                v1 = v if v.dim() == 1 else v[:, 0]
                if need_err:
                    e = (v1 - (bias[a1] if bias is not None else 0) + xn[ids]).clamp_min(0)
                    E.mul_(1 - alpha).add_(torch.zeros(k, device=dev).index_add_(0, a1, e), alpha=alpha)
                if top == 2 and not soft:
                    U.mul_(1 - alpha).add_(torch.zeros(k, device=dev).index_add_(0, a1, v[:, 1] - v[:, 0]), alpha=alpha)
            if t % every == 0:
                deadm = N < tau * N.mean()
                if util_tau > 0 and not soft: deadm |= U < util_tau * 0.5 * (1 + math.cos(math.pi * u)) * U.mean()
                if bool(deadm.any()):
                    Nc = N.cpu().numpy().astype(np.float64); dm = deadm.cpu().numpy()
                    dead = np.where(dm)[0]; alive = np.where(~dm)[0]
                    if len(dead) > len(alive): dead = dead[np.argsort(Nc[dead])[:len(alive)]]
                    nrev += len(dead)
                    if revive == "split":
                        w = Nc[alive] if donor == "count" else E.cpu().numpy().astype(np.float64)[alive] + 1e-30
                        donors = rng.choice(alive, size=len(dead), replace=False, p=w / w.sum())
                        dead, donors = torch.from_numpy(dead).to(dev), torch.from_numpy(donors).to(dev)
                        cd = Sm[donors] / N[donors, None]; half_n = N[donors] / 2
                        N[dead] = half_n; N[donors] = half_n
                        Sm[dead] = cd * up * half_n[:, None]; Sm[donors] = cd * dn * half_n[:, None]
                        E[dead] = E[donors] / 2; E[donors] = E[dead]; U[dead] = U.mean(); U[donors] = U.mean()
                    else:
                        p = e.cpu().double().numpy() + 1e-30 if revive == "d2" else np.ones(bs)
                        pick = torch.from_numpy(rng.choice(bs, size=min(len(dead), bs), replace=False, p=p / p.sum())).to(dev)
                        dead = torch.from_numpy(dead[:len(pick)]).to(dev); nn_ = tau * float(N.mean())
                        N[dead] = nn_; Sm[dead] = xb[pick] * nn_; E[dead] = 0; U[dead] = U.mean()
            Cprev = C; C = Sm / N[:, None]
    if final == "last":
        if pos < n: rest = perm[pos:]; last_a[rest] = nearest(Xd[rest], C)[0]
        cnt = torch.bincount(last_a, minlength=k).float()
        Ssum = torch.zeros(k, d, device=dev).index_add_(0, last_a, Xd); nz = cnt > 0; C[nz] = Ssum[nz] / cnt[nz, None]
    if polish: C = lloyd_from(Xd, C, polish, rng, k)
    Cf = C.cpu().numpy()
    return Cf, time.time() - t0, dict(B=B, a0=round(a0, 3), updates=t, revived=nrev)

def ladder(Xd, Xnp, k, T, seed, levels=3, phi=0.5, **kw):
    """LBG codebook doubling: one cheap pass at k/2^levels, then split every centroid and repeat, on a phi*n subsample;
    then T ordinary passes at full k. Extra cost is about phi * (1/2 + 1/4 + ...) of one pass."""
    rng = np.random.default_rng(seed + 1000); n, d = Xd.shape; t0 = time.time()
    sub = Xd[torch.from_numpy(rng.choice(n, int(phi * n), replace=False)).to(dev)]
    up, dn = (torch.from_numpy(v).float().to(dev) for v in split_pattern(d, 2**-7)); C = None
    for j in range(levels, 0, -1):
        ks = k >> j; C0 = sub[torch.from_numpy(rng.choice(len(sub), ks, replace=False)).to(dev)] if C is None else C
        Cs, _, _ = emeans_x(sub, None, ks, 1, seed, C0=C0); Cs = torch.from_numpy(Cs).to(dev); C = torch.cat([Cs * up, Cs * dn])
    Cf, _, info = emeans_x(Xd, Xnp, k, T, seed, C0=C, **kw)
    return Cf, time.time() - t0, info
