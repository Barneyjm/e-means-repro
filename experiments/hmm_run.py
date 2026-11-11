"""Usage: python hmm_run.py <text8.npy | synth> <log file> <K> <seed> <mode> [train symbols, default 20000000] [passes list]

Data: text8 (27 symbols; first <train symbols> for training, last 5,000,000 held out), or 'synth': 2.5M symbols drawn
from a known random 64-state HMM with sparse transitions and peaked emissions, 1M held out.
Training rows are 256 symbols. Score = held-out bits per symbol; held-out rows are 1,024 symbols, each scored after a
128-symbol warm-up (real preceding text), so chunk boundaries do not affect the score.

Modes (each appends JSON lines to the log):
  bw <passes> <seconds>  Baum-Welch curve, resumable: stops after <seconds> of wall time and continues when run again
  main      online version at 1, 2, 3, 5, 10 passes with the paper's formulas (n = number of rows)
  main_sym  the same with the formulas fed n = number of symbols; the batch size they return (in symbols) is
            converted to rows. Optional passes list, e.g. 2,3,5
  ablate    3 passes with one ingredient changed at a time
  (environment: HMM_VARIANTS='a|b' limits ablate to the named variants, HMM_KAPPAS=0.5,0.6 limits textbook)
  textbook  step (t+2)^-kappa, no clean-up, with and without revival, at the formula's batch size (n = rows), 1/4 and 1/16 of it
  grid      3 passes, batch size x peak step around the formula's choice
  grow      state splitting: start with K/4 states and double twice early in the run (splits also perturb transitions)
  buffer    32 symbols of real context on both sides of every training row (Baum-Welch 40 passes, online 1/2/3/5)
  short     rows of 64 symbols with 16 symbols of context on both sides instead of rows of 256 (same runs as buffer)
"""
import sys, os, json, time, math, numpy as np, torch
from common import dev, laws
from hmm_torch import init, bits, baum_welch, online, forward

L, WARM, EL = 256, 128, 1024

def text8(path, ntrain, buf=0, L=256):
    x = np.load(path, mmap_mode="r"); w = np.lib.stride_tricks.sliding_window_view
    ntrain -= ntrain % L
    tr = w(np.ascontiguousarray(x[:ntrain + 2 * buf]), L + 2 * buf)[::L] if buf else np.ascontiguousarray(x[:ntrain]).reshape(-1, L)
    te = w(np.ascontiguousarray(x[-5_000_000 - WARM:]), EL + WARM)[::EL]
    return tr.astype(np.int64), te.astype(np.int64), 27, None

def synth(K=64, V=27, ntrain=2_500_000, ntest=1_000_000, seed=1234, ca=0.03, cb=0.03):
    rng = np.random.default_rng(seed); g = torch.Generator().manual_seed(seed)
    A = torch.from_numpy(rng.dirichlet(np.full(K, ca), K)); Bm = torch.from_numpy(rng.dirichlet(np.full(V, cb), K))
    pi = torch.full((K,), 1.0 / K, dtype=torch.float64)
    for _ in range(2000): pi = pi @ A                          # stationary distribution
    def draw(rows, length):
        z = torch.multinomial(pi, rows, True, generator=g); out = torch.empty(rows, length, dtype=torch.int64)
        for t in range(length):
            if t: z = torch.multinomial(A[z], 1, generator=g)[:, 0]
            out[:, t] = torch.multinomial(Bm[z], 1, generator=g)[:, 0]
        return out.numpy()
    return draw(ntrain // L, L), draw(ntest // EL, EL + WARM), V, (pi, A, Bm)

if __name__ == "__main__":
    src, outp, K, seed, mode = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    ds = "synth" if src == "synth" else "text8"
    if mode == "bw": passes, budget = int(sys.argv[6]), float(sys.argv[7]); ntrain = int(sys.argv[8]) if len(sys.argv) > 8 else 20_000_000
    else: ntrain = int(sys.argv[6]) if len(sys.argv) > 6 else 20_000_000
    buf = 32 if mode == "buffer" else 16 if mode == "short" else 0
    if mode == "short": L = 64
    Xn, Xe, V, truth = synth() if ds == "synth" else text8(src, ntrain, buf, L)
    X = torch.from_numpy(Xn).to(dev); xe = torch.from_numpy(Xe).to(dev); n = len(X)
    f = open(outp, "a")
    def log(**kw): s = json.dumps(dict(ds=ds, K=K, n=n, seed=seed, **kw)); print(s, flush=True); f.write(s + "\n"); f.flush()
    def run(m, T, **kw):
        p, dt, info = online(X, K, V, T, seed, buf=buf, **kw); log(m=m, T=T, time=round(dt, 2), bits=round(bits(xe, WARM, *p), 5), **info)
    B0, a00 = laws(n, K, 3)
    def sym_laws(T): Bs, a = laws(n * L, K, T); return dict(B=max(1, Bs // L), a0=a)
    Ts = [int(t) for t in sys.argv[7].split(",")] if mode != "bw" and len(sys.argv) > 7 else (1, 2, 3, 5, 10)
    if mode == "bw":
        ck = f"{outp}.{ds}_{K}_{seed}.pt"; st = torch.load(ck, map_location=dev) if os.path.exists(ck) else None
        if st is None and seed == 0:                           # reference scores, logged once
            flat = Xn.reshape(-1); e = Xe[:, WARM:].reshape(-1); prev = Xe[:, WARM-1:-1].reshape(-1)
            u = np.bincount(flat, minlength=V) + 1.0; u /= u.sum(); bi = np.ones((V, V)); np.add.at(bi, (flat[:-1], flat[1:]), 1); bi /= bi.sum(1, keepdims=True)
            log(m="reference", unigram=round(float(-np.log2(u[e]).mean()), 5), bigram=round(float(-np.log2(bi[prev, e]).mean()), 5),
                truth=round(bits(xe, WARM, *[v.float().to(dev) for v in truth]), 5) if truth else None)
        done = st["it"] if st else 0; spent0 = st["spent"] if st else 0.0; t_start = time.time()
        class Stop(Exception): pass
        def hook(it, spent, p, ll):
            log(m="bw", it=done + it, time=round(spent0 + spent, 2), bits=round(bits(xe, WARM, *p), 5), train_bits=round(-ll / (n * L) / math.log(2), 5))
            stop = time.time() - t_start > budget or done + it >= passes
            if stop or (done + it) % 10 == 0: torch.save(dict(it=done + it, spent=spent0 + spent, p=[v.cpu() for v in p]), ck)
            if stop: raise Stop
        try:
            if done < passes: baum_welch(X, K, V, passes - done, seed, hook, start=[v.to(dev) for v in st["p"]] + [None] if st else None)
        except Stop: pass
    elif mode == "main":
        online(X[:2048], K, V, 1, 99, B=1024)                  # warm up GPU kernels
        for T in Ts: run("online", T)
    elif mode == "main_sym":
        for T in Ts: run("online_sym", T, **sym_laws(T))
    elif mode == "ablate":
        for name, kw in (("default", {}), ("constant step", dict(schedule="const")), ("no revival", dict(revive=False)), ("no last-pass clean-up", dict(final=False)),
                         ("tiny split (1 +- 2^-7)", dict(split_size=2 ** -7)), ("B/4, step/4", dict(B=B0 // 4, a0=a00 / 4)),
                         ("n = symbols", sym_laws(3)), ("n = symbols, constant step", dict(schedule="const", **sym_laws(3))),
                         ("n = symbols, no revival", dict(revive=False, **sym_laws(3))), ("n = symbols, no last-pass clean-up", dict(final=False, **sym_laws(3)))):
            if os.environ.get("HMM_VARIANTS") and name not in os.environ["HMM_VARIANTS"].split("|"): continue
            p, dt, info = online(X, K, V, 3, seed, **kw); log(m="ablate", T=3, variant=name, time=round(dt, 2), bits=round(bits(xe, WARM, *p), 5), **info)
    elif mode == "textbook":
        kap = [float(v) for v in os.environ.get("HMM_KAPPAS", "0.5,0.6,0.75,0.9").split(",")]
        for bm, kappas in ((1, kap), (4, kap), (16, kap)):
            for kp in kappas:
                for rv in (False, True):
                    p, dt, info = online(X, K, V, 3, seed, B=B0 // bm, schedule="stepwise", kappa=kp, revive=rv, final=False)
                    log(m="textbook", T=3, bdiv=bm, kappa=kp, revive=rv, time=round(dt, 2), bits=round(bits(xe, WARM, *p), 5), **info)
    elif mode == "grid":
        log(m="law", T=3, B=B0, a0=round(a00, 3))
        for bm, steps in ((1, (1.0, 0.5, 0.25)), (4, (1.0, 0.5, 0.25, 0.125)), (16, (1.0, 0.5, 0.25, 0.125, 0.06)), (64, (0.25,))):
            for a in steps:
                p, dt, info = online(X, K, V, 3, seed, B=B0 // bm, a0=a); log(m="grid", T=3, bdiv=bm, time=round(dt, 2), bits=round(bits(xe, WARM, *p), 5), **{**info, "a0": a})
    elif mode == "grow":
        for T in (3, 5):
            for name, fr in (("fixed", None), ("grow at 1/6, 1/3", (1 / 6, 1 / 3)), ("grow at 1/10, 1/5", (0.1, 0.2))):
                p, dt, info = online(X, K, V, T, seed, grow=(K // 4, fr) if fr else None, split_trans=True)
                log(m="grow", T=T, variant=name, time=round(dt, 2), bits=round(bits(xe, WARM, *p), 5), **info)
    elif mode in ("buffer", "short"):
        cur = []; _, dt = baum_welch(X, K, V, 40, seed, lambda it, s, p, ll: cur.append(round(bits(xe, WARM, *p), 5)), buf=buf)
        log(m=mode + "_bw", curve=cur, time=round(dt, 2), rows=L, buf=buf)
        for T in (1, 2, 3, 5): run(mode + "_online", T)
