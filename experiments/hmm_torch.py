"""Hidden Markov models with categorical emissions: full-pass Baum-Welch, and an online version built with the
e-means recipe. Both use the same batched, scaled forward-backward kernel.

Training data is a table of fixed-length chunks (rows). Every chunk is treated as its own sequence that starts from
the learned initial distribution; no state is carried across chunk boundaries. With buf > 0 each row carries buf
extra symbols of real context at both ends, which take part in forward-backward but not in the counts.
State for the online version is three decayed running totals of expected counts: initial state (K), transitions
(K x K) and emissions (K x V), the HMM analogue of e-means' (count, sum).
"""
import math, time, numpy as np, torch
from common import dev, sync, laws

TINY = 1e-30

def to_params(I, Tr, E):
    return I / I.sum().clamp_min(TINY), Tr / Tr.sum(1, keepdim=True).clamp_min(TINY), E / E.sum(1, keepdim=True).clamp_min(TINY)

def forward(xt, pi, A, Bm):
    """xt: (L, rows) symbols. Returns scaled alphas (L, rows, K), scale factors c (L, rows, 1), emissions em (L, rows, K)"""
    L, n = xt.shape; em = Bm.T[xt]
    al = torch.empty_like(em); c = torch.empty(L, n, 1, device=xt.device, dtype=em.dtype)
    a = pi[None, :] * em[0]
    for t in range(L):
        if t: a = (a @ A) * em[t]
        c[t] = a.sum(1, keepdim=True).clamp_min(TINY); a = a / c[t]; al[t] = a
    return al, c, em

def stats(x, pi, A, Bm, ch, buf=0):
    """expected counts of a block of chunks under the current model (summed over rows) and its log-likelihood"""
    K, V = Bm.shape; I = torch.zeros(K, device=x.device, dtype=A.dtype); Tr = torch.zeros(K, K, device=x.device, dtype=A.dtype)
    Et = torch.zeros(V, K, device=x.device, dtype=A.dtype); ll = 0.0; At = A.T.contiguous()
    for i in range(0, x.shape[0], ch):
        xt = x[i:i+ch].T.contiguous(); L, n = xt.shape; lo, hi = buf, L - buf
        al, c, em = forward(xt, pi, A, Bm)
        em.div_(c)                                           # em[t] is now b_t / c_t
        beta = torch.empty_like(al); beta[L-1] = 1.0
        for t in range(L - 1, 0, -1):
            em[t].mul_(beta[t]); beta[t-1] = em[t] @ At      # em[t] becomes b_t * beta_t / c_t
        Tr += al[lo:hi-1].reshape(-1, K).T @ em[lo+1:hi].reshape(-1, K)
        al.mul_(beta)                                        # state posteriors
        I += al[lo].sum(0); Et.index_add_(0, xt[lo:hi].reshape(-1), al[lo:hi].reshape(-1, K)); ll += float(torch.log(c[lo:hi]).sum())
    return I, Tr * A, Et.T.contiguous(), ll

def bits(xe, warm, pi, A, Bm, ch=8192):
    """held-out bits per symbol. xe: (rows, warm + L) on the device; the first `warm` symbols of each row only set
    the state. Forward pass in the model's precision, scale factors summed in float64."""
    tot = 0.0; cnt = 0; BmT = Bm.T.contiguous()
    for i in range(0, xe.shape[0], ch):
        xt = xe[i:i+ch].T.contiguous(); acc = []
        a = pi[None, :] * BmT[xt[0]]
        for t in range(xt.shape[0]):
            if t: a = (a @ A) * BmT[xt[t]]
            c = a.sum(1, keepdim=True).clamp_min(TINY); a = a / c
            if t >= warm: acc.append(torch.log(c).sum())
        tot += float(torch.stack(acc).cpu().double().sum()); cnt += (xt.shape[0] - warm) * xt.shape[1]
    return -tot / cnt / math.log(2)

def init(K, V, seed, device=dev, dtype=torch.float32):
    """uniform initial distribution; every transition and emission row is a draw from a flat Dirichlet.
    (A start closer to uniform, rows = normalised uniform(0.5, 1.5), kept Baum-Welch at the unigram score for 30 passes.)"""
    rng = np.random.default_rng(seed)
    A = torch.from_numpy(rng.dirichlet(np.ones(K), K)).to(device=device, dtype=dtype); Bm = torch.from_numpy(rng.dirichlet(np.ones(V), K)).to(device=device, dtype=dtype)
    return torch.full((K,), 1.0 / K, device=device, dtype=dtype), A, Bm, rng

def chunk_rows(K, L=256): return max(256, min(32768, 2 ** int(math.log2(7e7 / (L * K)))))     # rows per kernel call, by memory

def baum_welch(X, K, V, iters, seed, hook=None, start=None, buf=0):
    """Standard Baum-Welch: one full pass per update. hook(iteration, seconds so far, params, train loglik) is not timed."""
    pi, A, Bm, _ = start or init(K, V, seed, X.device); spent = 0.0; ch = chunk_rows(K, X.shape[1])
    for it in range(1, iters + 1):
        t0 = time.time()
        I, Tr, E, ll = stats(X, pi, A, Bm, ch, buf); pi, A, Bm = to_params(I, Tr, E); sync()
        spent += time.time() - t0
        if hook: hook(it, spent, (pi, A, Bm), ll)
    return (pi, A, Bm), spent

def split_states(I, Tr, E, dead, donors, rng, size, trans=False):
    """Re-seed states. Each takes half of a donor state's counts: same outgoing row, half of the incoming column,
    half of the initial mass. That alone leaves the model's likelihood unchanged (two identical copies of a state).
    The emission rows of the two halves are then pushed apart by factors (1 + size*s) and (1 - size*s), with a random
    sign s per symbol and per split; trans=True does the same to their outgoing transition rows."""
    def push(M, idx_a, idx_b):
        h = M[idx_b]; s = torch.from_numpy(rng.choice([-1.0, 1.0], size=tuple(h.shape))).to(device=M.device, dtype=M.dtype) * size
        up = h * (1 + s); dn = h * (1 - s); tot = h.sum(1, keepdim=True)
        M[idx_a] = up * (tot / up.sum(1, keepdim=True).clamp_min(TINY)); M[idx_b] = dn * (tot / dn.sum(1, keepdim=True).clamp_min(TINY))
    Tr[dead] = Tr[donors] / 2; Tr[donors] /= 2
    Tr[:, dead] = Tr[:, donors] / 2; Tr[:, donors] /= 2
    I[dead] = I[donors] / 2; I[donors] /= 2
    E[dead] = E[donors] / 2; E[donors] /= 2
    push(E, dead, donors)
    if trans: push(Tr, dead, donors)

def online(X, K, V, T, seed, B=None, a0=None, schedule="cosine", kappa=0.6, revive=True, tau=0.1, split_size=0.5,
           final=True, start=None, grow=None, buf=0, split_trans=False):
    """Online Baum-Welch with the e-means ingredients, each switchable:
    schedule  'cosine' (a0 down to 0.1*a0), 'const', or 'stepwise' = (t+2)^-kappa, the textbook online-EM step rule
    revive    every 5 updates, states whose expected occupancy is below tau * mean are re-seeded by split_states
    final     parameters from the plain (undecayed) expected counts of the last pass instead of the running totals
    grow      optional (K0, fractions): start with K0 states and double by splitting every state at the given
              fractions of the run (state-splitting, see hmm_run.py); K must equal K0 * 2**len(fractions)
    """
    n = X.shape[0]; g = torch.Generator().manual_seed(seed)
    Bl, al = laws(n, K, T); B = B or Bl; a0 = a0 or al
    t0 = time.time()
    Kc = grow[0] if grow else K
    pi, A, Bm, rng = start or init(Kc, V, seed, X.device)
    I = pi * 1e-10; Tr = A * 1e-10; E = Bm * 1e-10
    nb = n // B; M = T * nb; t = 0; revived = 0; fin = None
    grow_at = [int(round(f * M)) for f in grow[1]] if grow else []
    for ep in range(T):
        perm = torch.randperm(n, generator=g).to(X.device)
        for bi in range(nb):
            if t in grow_at:                                 # double the state count: every state is split in two
                idx = torch.arange(Kc, device=X.device); z = lambda *s: torch.zeros(*s, device=X.device)
                I = torch.cat([I, z(Kc)]); E = torch.cat([E, z(Kc, V)]); Tr = torch.cat([torch.cat([Tr, z(Kc, Kc)], 1), z(Kc, 2 * Kc)], 0)
                split_states(I, Tr, E, idx + Kc, idx, rng, split_size, split_trans); Kc *= 2; pi, A, Bm = to_params(I, Tr, E)
            xb = X[perm[bi*B:(bi+1)*B]]
            if schedule == "cosine": alpha = a0 * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / M)))
            elif schedule == "const": alpha = a0
            else: alpha = (t + 2) ** -kappa
            t += 1
            bI, bT, bE, _ = stats(xb, pi, A, Bm, chunk_rows(Kc, X.shape[1]), buf)
            if ep == T - 1 and final:
                fin = [bI.clone(), bT.clone(), bE.clone()] if fin is None else [f + b for f, b in zip(fin, (bI, bT, bE))]
            s = alpha / len(xb)
            I.mul_(1 - alpha).add_(bI, alpha=s); Tr.mul_(1 - alpha).add_(bT, alpha=s); E.mul_(1 - alpha).add_(bE, alpha=s)
            if revive and t % 5 == 0:
                occ = E.sum(1); deadm = occ < tau * occ.mean()
                if bool(deadm.any()):
                    oc = occ.cpu().numpy().astype(np.float64); dm = deadm.cpu().numpy()
                    dead = np.where(dm)[0]; alive = np.where(~dm)[0]
                    donors = rng.choice(alive, size=min(len(dead), len(alive)), replace=False, p=oc[alive] / oc[alive].sum())
                    dead = torch.from_numpy(dead[:len(donors)]).to(X.device); donors = torch.from_numpy(donors).to(X.device)
                    split_states(I, Tr, E, dead, donors, rng, split_size, split_trans); revived += len(dead)
            pi, A, Bm = to_params(I, Tr, E)
    assert Kc == K
    if final and fin is not None:
        if n % B:
            bI, bT, bE, _ = stats(X[perm[nb*B:]], pi, A, Bm, chunk_rows(K, X.shape[1]), buf); fin = [f + b for f, b in zip(fin, (bI, bT, bE))]
        keep = fin[2].sum(1) > 1e-8 * fin[2].sum()           # a state that saw nothing in the last pass keeps its running value
        pi = fin[0] / fin[0].sum(); _, A2, B2 = to_params(*fin)
        A = torch.where(keep[:, None], A2, A); Bm = torch.where(keep[:, None], B2, Bm)
    sync()
    return (pi, A, Bm), time.time() - t0, dict(B=B, a0=round(a0, 3), updates=M, revived=revived)
