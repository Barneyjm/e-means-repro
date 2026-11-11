"""Usage: python bench_gmm_sklearn.py <X.npy> <heldout.npy> <log file> <mode> [seeds] [K list | passes]

CPU-only comparison of OnlineGaussianMixture (numpy/scipy) with scikit-learn's GaussianMixture, diagonal covariances.
X / heldout: float32 arrays, e.g. SIFT1M reduced to 64 whitened PCA dimensions (run_gmm.sift64) and its 100k learn set.
Score = mean log-likelihood per held-out row, computed in float64 by the same function for every model.

modes: big_mine     1M rows, K = 256 and 1024: online at 1/2/3/5/10 passes, published formulas and batch/4 + step/4
       big_sk_random  1M rows, K = 256, first seed: GaussianMixture from the same random start, run up to <passes> passes;
                      state is kept next to the log, so calling it with 25, then 60, then 100 continues one run
       big_sk_default 1M rows, K = 256, first seed: GaussianMixture with its defaults (k-means start, to convergence)
       mid          first 100k rows, K = 64 and 256: both estimators
       cm           textbook online EM (Cappe & Moulines 2009: step t^-0.6, no revival, Polyak averaging over the second
                    half) at several batch sizes, K = 256, on the first <rows> rows (last argument)
       init         random rows vs k-means++ seeds vs an e-means clustering as the starting means; last argument = rows,K[,K...]
       passes       more passes; last argument = rows,K,passes[,passes...]
       threads      1M rows, K = 256: online, 3 passes, 1 thread vs all
       small        iris / wine / digits / blobs / a synthetic mixture, plus two low-dimensional cuts of X; 10 random splits each
       summary      <X.npy> is instead the folder holding gmm_cpu_*.log; prints the markdown tables
"""
import sys, os, json, time, pickle, warnings
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from sklearn.mixture import GaussianMixture
from online_gmm_sklearn import OnlineGaussianMixture
from emeans_sklearn import batch_law, step_law
warnings.filterwarnings("ignore")

xp, hp, out, mode = sys.argv[1:5]; seeds = [int(s) for s in sys.argv[5].split(",")] if len(sys.argv) > 5 else [0, 1, 2]
extra = [int(s) for s in sys.argv[6].split(",")] if len(sys.argv) > 6 else None
f = open(out, "a")
def log(**kw): s = json.dumps(kw); print(s, flush=True); f.write(s + "\n"); f.flush()
LOG2PI = np.log(2 * np.pi)

def heldout(H, w, mu, var):
    """exact float64 mean log-likelihood per row"""
    H = H.astype(np.float64); tot = 0.0; iv = 1 / var
    with np.errstate(divide="ignore"): const = np.log(w) - 0.5 * (np.log(var).sum(1) + (mu * mu * iv).sum(1) + H.shape[1] * LOG2PI)
    for i in range(0, len(H), 8192):
        x = H[i:i + 8192]; L = x @ (mu * iv).T - 0.5 * (x * x) @ iv.T + const; m = L.max(1); tot += (m + np.log(np.exp(L - m[:, None]).sum(1))).sum()
    return tot / len(H)

def diag(g): return g.weights_, g.means_, np.broadcast_to(g.covariances_[:, None] if g.covariances_.ndim == 1 else g.covariances_, g.means_.shape)

def online(X, H, K, T, seed, rule="published", **kw):
    kw.setdefault("init_params", "random_from_data")    # the estimator's default above 64 components; explicit so every K is comparable
    if rule == "quarter":                                   # same memory window, 4x smaller batches
        B = batch_law(len(X), K, T); kw.update(batch_size=max(B // 4, 32), step_size=step_law(len(X), K, T, B) / 4)
    t = time.time(); g = OnlineGaussianMixture(K, n_epochs=T, random_state=seed, **kw).fit(X); dt = time.time() - t
    return dict(m="online", rule=rule, T=T, seed=seed, time=round(dt, 2), test=round(heldout(H, *diag(g)), 5), B=g.batch_size_, a0=round(g.step_size_, 3),
                revived=g.n_revived_, small=int((g.weights_ * K < 0.1).sum()))

class CappeMoulines(OnlineGaussianMixture):
    """Textbook online EM on the same kernel: step gamma0 * t^-0.6, running totals averaged over the second half of the run."""
    def _step(self, t, M, a0): return min(a0 * (t + 1) ** -0.6, 1.0)
    def _after_update(self, t, M, N, S, Q):
        if t == 1: self._avg, self._navg = [np.zeros_like(N), np.zeros_like(S), np.zeros_like(Q)], 0
        if t > M // 2:
            for a, b in zip(self._avg, (N, S, Q)): a += b
            self._navg += 1
    def _final_totals(self, running, last_pass): return [a / self._navg for a in self._avg]

def same_start(X, K, seed):
    """the start OnlineGaussianMixture uses: K random rows, global variance, equal weights"""
    rng = np.random.RandomState(seed); idx = rng.choice(len(X), K, replace=False)
    return dict(weights_init=np.full(K, 1 / K, dtype=X.dtype), means_init=X[idx],                 # same dtype as X, or scikit-learn
                precisions_init=np.tile(1 / (X.var(0, dtype=np.float64) + 1e-6), (K, 1)).astype(X.dtype))   # switches to float64

def sk_random(X, H, K, seed, stops, state=None, **tag):
    """full-pass EM from the same kind of start; tol=0 so it runs exactly the passes asked for"""
    if state and os.path.exists(state): g, spent, done = pickle.load(open(state, "rb"))
    else: g = GaussianMixture(K, covariance_type="diag", tol=0, warm_start=True, max_iter=stops[0], **same_start(X, K, seed)); spent = 0.0; done = 0
    for it in stops:
        if it <= done: continue
        g.max_iter = it - done; t = time.time(); g.fit(X); spent += time.time() - t; done = it
        log(m="sklearn", init="same random start", passes=it, seed=seed, time=round(spent, 2), test=round(heldout(H, *diag(g)), 5), small=int((g.weights_ * K < 0.1).sum()), K=K, **tag)
        if state: pickle.dump((g, spent, done), open(state, "wb"))

def sk_default(X, H, K, seed, **tag):
    t = time.time(); g = GaussianMixture(K, covariance_type="diag", random_state=seed).fit(X); dt = time.time() - t
    log(m="sklearn", init="defaults (k-means start)", passes=int(g.n_iter_), converged=bool(g.converged_), seed=seed, time=round(dt, 2),
        test=round(heldout(H, *diag(g)), 5), small=int((g.weights_ * K < 0.1).sum()), K=K, **tag)

if mode not in ("small", "summary"):
    X = np.load(xp); H = np.load(hp)
    OnlineGaussianMixture(8, random_state=0).fit(X[:20000])            # warm up threads and BLAS
if mode == "big_mine":
    for K in extra or (256, 1024):
        for seed in seeds:
            for T in (1, 2, 3, 5, 10):
                for rule in ("published", "quarter"): log(cfg=f"1M_K{K}", K=K, **online(X, H, K, T, seed, rule))
elif mode == "big_sk_random":
    sk_random(X, H, 256, seeds[0], extra, state=out + ".skstate.pkl", cfg="1M_K256")
elif mode == "big_sk_default":
    sk_default(X, H, 256, seeds[0], cfg="1M_K256")
elif mode == "mid":
    X = X[:100_000]
    for K in extra or (64, 256):
        for seed in seeds:
            for T in (1, 2, 3, 5, 10):
                for rule in ("published", "quarter"): log(cfg=f"100k_K{K}", K=K, **online(X, H, K, T, seed, rule))
            sk_random(X, H, K, seed, (25, 100), cfg=f"100k_K{K}")
            sk_default(X, H, K, seed, cfg=f"100k_K{K}")
elif mode == "cm":
    X = X[:extra[0]]; K = 256; tag = dict(cfg=f"{'1M' if len(X) == 1_000_000 else str(len(X) // 1000) + 'k'}_K{K}", K=K)
    for seed in seeds:
        for T in (3, 10):
            log(**tag, **online(X, H, K, T, seed))
            for div in (1, 8, 32, 128):
                for avg in (True, False):
                    B = max(batch_law(len(X), K, T) // div, 64); t = time.time()
                    g = (CappeMoulines if avg else type("NoAvg", (CappeMoulines,), dict(_final_totals=lambda self, running, last_pass: running)))(
                        K, n_epochs=T, batch_size=B, step_size=1.0, revive=False, random_state=seed).fit(X); dt = time.time() - t
                    log(m="cappe_moulines", averaging=avg, T=T, seed=seed, B=B, time=round(dt, 2), test=round(heldout(H, *diag(g)), 5), small=int((g.weights_ * K < 0.1).sum()), **tag)
elif mode == "init":
    n = extra[0]; X = X[:n]
    for K in extra[1:]:
        for seed in seeds:
            for T in (1, 2, 3, 5):
                for ip in ("random_from_data", "k-means++", "emeans"):
                    r = online(X, H, K, T, seed, init_params=ip); r.update(init=ip, cfg=f"{'1M' if n == 1_000_000 else str(n // 1000) + 'k'}_K{K}", K=K); log(**r)
elif mode == "mid_sk_random":
    X = X[:100_000]
    for K in extra or (64, 256):
        for seed in seeds: sk_random(X, H, K, seed, (25, 100), cfg=f"100k_K{K}")
elif mode == "passes":
    n, K = extra[:2]; X = X[:n]
    for seed in seeds:
        for T in extra[2:]:
            for ip in ("random_from_data", "k-means++"):
                r = online(X, H, K, T, seed, init_params=ip); r.update(init=ip, cfg=f"{'1M' if n == 1_000_000 else str(n // 1000) + 'k'}_K{K}", K=K); log(**r)
elif mode == "threads":
    for th in (1, 4, None):
        for seed in seeds: log(cfg="1M_K256", K=256, threads=th or "all", **online(X, H, 256, 3, seed, n_threads=th))
elif mode == "small":
    from sklearn.datasets import load_digits, load_iris, load_wine, make_blobs
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    def mixture(n=60000, K=24, d=16, seed=0):
        rng = np.random.RandomState(seed); mu = rng.normal(0, 1.0, (K, d)); sd = np.exp(rng.uniform(np.log(0.4), np.log(1.5), (K, d))); z = rng.randint(K, size=n)
        return (mu[z] + rng.standard_normal((n, d)) * sd[z]).astype(np.float32)
    sets = [("iris 150x4, K=3", load_iris().data, 3), ("wine 178x13, K=3", StandardScaler().fit_transform(load_wine().data), 3),
            ("digits (PCA 16) 1797x16, K=10", PCA(16, whiten=True, random_state=0).fit_transform(load_digits().data), 10),
            ("blobs 10000x2, K=8", make_blobs(n_samples=10000, centers=8, n_features=2, random_state=0)[0], 8),
            ("well-separated blobs 20000x20, K=12", make_blobs(n_samples=20000, centers=12, n_features=20, random_state=0)[0], 12),
            ("blobs 100000x2, K=30", make_blobs(n_samples=100000, centers=30, n_features=2, cluster_std=0.6, random_state=1)[0], 30),
            ("synthetic mixture 60000x16, K=24", mixture(), 24)]
    if xp != "-":
        S = np.load(xp)[:120_000]; sets += [("SIFT first 8 PCA dims 120000x8, K=64", S[:, :8], 64), ("SIFT first 16 PCA dims 120000x16, K=64", S[:, :16], 64)]
    for cfg, D, K in sets:
        big = len(D) > 20000
        for seed in range(10):
            p = np.random.RandomState(seed).permutation(len(D)); cut = int(0.8 * len(D)); X, H = D[p[:cut]], D[p[cut:]]
            law3, law10 = batch_law(len(X), K, 3), batch_law(len(X), K, 10)
            ests = [("GaussianMixture defaults", GaussianMixture(K, covariance_type="diag", random_state=seed))]
            if not big: ests.append(("GaussianMixture n_init=5", GaussianMixture(K, covariance_type="diag", n_init=5, random_state=seed)))
            rnd = dict(init_params="random_from_data", random_state=seed)      # every K here is <= 64, so the default start is k-means++
            ests += [("Online 3 passes (defaults)", OnlineGaussianMixture(K, random_state=seed)),
                     ("Online 10 passes (defaults)", OnlineGaussianMixture(K, n_epochs=10, random_state=seed)),
                     ("Online 3 passes, random start", OnlineGaussianMixture(K, **rnd)), ("Online 10 passes, random start", OnlineGaussianMixture(K, n_epochs=10, **rnd)),
                     ("Online 3 passes, e-means start", OnlineGaussianMixture(K, init_params="emeans", random_state=seed)),
                     ("Online 3 passes, random start, formula batch size", OnlineGaussianMixture(K, batch_size=law3, **rnd)),
                     ("Online 10 passes, random start, formula batch size", OnlineGaussianMixture(K, n_epochs=10, batch_size=law10, **rnd)),
                     ("Online 10 passes, random start, no revival", OnlineGaussianMixture(K, n_epochs=10, revive=False, **rnd))]
            for name, est in ests:
                t = time.time(); est.fit(X); dt = time.time() - t
                log(cfg=cfg, K=K, method=name, seed=seed, time=round(dt, 4), test=round(heldout(H, *diag(est)), 4), train=round(heldout(X, *diag(est)), 4),
                    B=getattr(est, "batch_size_", None), a0=getattr(est, "step_size_", None))
elif mode == "summary":
    import collections
    def rows(p): return [json.loads(l) for l in open(p)] if os.path.exists(p) else []
    def ms(v, nd=3): v = np.array(v, dtype=float); return f"{v.mean():.{nd}f}" + (f" ± {v.std():.{nd}f}" if len(v) > 1 else "")
    big, mid, small, cm, ini = (rows(os.path.join(xp, f"gmm_cpu_{n}.log")) for n in ("1M", "100k", "small", "textbook", "init"))
    for title, rs in (("1M rows", big), ("100k rows", mid)):
        for cfg in sorted({r["cfg"] for r in rs}):
            g = collections.defaultdict(list)
            for r in rs:
                if r["cfg"] != cfg or "threads" in r: continue
                key = (0, r["init"], r["passes"] if "same" in r["init"] else 0) if r["m"] == "sklearn" else (1, r["rule"], r["T"]); g[key].append(r)
            ref = np.mean([r["test"] for r in g.get((0, "defaults (k-means start)", 0), [])] or [np.nan])
            print(f"\n### {cfg.replace('_', ', ').replace('K', 'K = ')}\n\n| Method | Passes | Seeds | Time (s) | Held-out log-lik | vs scikit-learn defaults |\n|---|---|---|---|---|---|")
            for key in sorted(g):
                v = g[key]; te = [r["test"] for r in v]; ti = [r["time"] for r in v]
                if key[0] == 0: name = "`GaussianMixture`, " + ("defaults (k-means start, to convergence)" if key[2] == 0 else "same random start as online"); passes = ms([r["passes"] for r in v], 0) if key[2] == 0 else str(key[2])
                else: name = "`OnlineGaussianMixture`, random start" + ("" if key[1] == "published" else ", batch / 4 and step / 4"); passes = str(key[2])
                print(f"| {name} | {passes} | {len(v)} | {ms(ti, 2)} | {ms(te)} | {'–' if np.isnan(ref) else format(np.mean(te) - ref, '+.3f')} |")
    th = collections.defaultdict(list)
    for r in big:
        if "threads" in r: th[r["threads"]].append(r["time"])
    if th: print("\nThreads (1M rows, K = 256, 3 passes): " + ", ".join(f"{k} threads {np.mean(v):.2f} s" for k, v in th.items()))
    if cm:
        print("\n### Textbook online EM (step t^-0.6, no revival), K = 256\n\n| Rows | Passes | Method | Batch | Time (s) | Held-out log-lik | vs online recipe |\n|---|---|---|---|---|---|---|")
        for cfg in sorted({r["cfg"] for r in cm}):
            for T in (3, 10):
                g = collections.defaultdict(list)
                for r in cm:
                    if r["cfg"] == cfg and r["T"] == T: g[("recipe", r["B"]) if r["m"] == "online" else ("averaged over 2nd half" if r["averaging"] else "last iterate", r["B"])].append(r)
                ref = np.mean([r["test"] for k, v in g.items() if k[0] == "recipe" for r in v])
                for key in sorted(g, key=lambda k: (k[0] != "recipe", k[0], -k[1])):
                    v = g[key]; print(f"| {cfg.split('_')[0]} | {T} | {'`OnlineGaussianMixture`' if key[0] == 'recipe' else 'textbook, ' + key[0]} | {key[1]} | {ms([r['time'] for r in v], 2)} | {ms([r['test'] for r in v])} | {np.mean([r['test'] for r in v]) - ref:+.3f} |")
    if ini:
        print("\n### Starting means (time includes the seeding)\n\n| Data | Passes | Random rows | Time (s) | k-means++ seeds | Time (s) | vs random | e-means clustering | Time (s) | vs random |\n|---|---|---|---|---|---|---|---|---|---|")
        for cfg in sorted({r["cfg"] for r in ini}):
            for T in sorted({r["T"] for r in ini}):
                by = {ip: {r["seed"]: r for r in ini if r["cfg"] == cfg and r["T"] == T and r["init"] == ip} for ip in ("random_from_data", "k-means++", "emeans")}
                a = by["random_from_data"]; cells = [ms([r["test"] for r in a.values()]), f"{np.mean([r['time'] for r in a.values()]):.2f}"]
                for ip in ("k-means++", "emeans"):
                    b = by[ip]; d = [b[k]["test"] - a[k]["test"] for k in a if k in b]
                    cells += [ms([r["test"] for r in b.values()]), f"{np.mean([r['time'] for r in b.values()]):.2f}", f"{np.mean(d):+.3f} ± {np.std(d):.3f}"] if b else ["", "", ""]
                print(f"| {cfg.replace('_', ', ').replace('K', 'K = ')} | {T} | " + " | ".join(cells) + " |")
    more = rows(os.path.join(xp, "gmm_cpu_passes.log"))
    if more:
        print("\n### More passes\n\n| Data | Passes | Random start | Time (s) | k-means++ start | Time (s) |\n|---|---|---|---|---|---|")
        for cfg in sorted({r["cfg"] for r in more}):
            for T in sorted({r["T"] for r in more if r["cfg"] == cfg}):
                c = []
                for ip in ("random_from_data", "k-means++"):
                    v = [r for r in more if r["cfg"] == cfg and r["T"] == T and r["init"] == ip]; c += [ms([r["test"] for r in v]), f"{np.mean([r['time'] for r in v]):.2f}"] if v else ["", ""]
                print(f"| {cfg.replace('_', ', ').replace('K', 'K = ')} | {T} | " + " | ".join(c) + " |")
    if small:
        print("\n### Small and low-dimensional data (10 random 80/20 splits; held-out log-likelihood)\n\n| Data | Method | Held-out log-lik | vs GaussianMixture defaults | Worst split vs defaults | Time (ms) |\n|---|---|---|---|---|---|")
        order = []
        for r in small:
            if r["cfg"] not in order: order.append(r["cfg"])
        for cfg in order:
            g = collections.defaultdict(dict)
            for r in small:
                if r["cfg"] == cfg: g[r["method"]][r["seed"]] = r
            base = g["GaussianMixture defaults"]
            for m, v in g.items():
                d = [v[s]["test"] - base[s]["test"] for s in v]
                print(f"| {cfg} | {m} | {np.mean([r['test'] for r in v.values()]):.3f} | {np.mean(d):+.3f} | {np.min(d):+.3f} | {1000 * np.mean([r['time'] for r in v.values()]):.0f} |")
