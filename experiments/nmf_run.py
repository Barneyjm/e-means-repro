"""Usage: python nmf_run.py <nmf_tfidf.npz> <log file> <K> <seeds> <group> [passes for batch | name filter for sklearn]

Held-out relative squared error on TF-IDF rows (codes solved the same way for every method). Groups, resumable:
  batch    full-batch HALS, error after every pass
  main     online with the e-means recipe at 1/2/3/5/10 passes
  ablate   3 passes, one ingredient changed at a time; plus smaller batches and steps
  textbook running average of all batches (Mairal-style, B = 512) and power-law steps, with and without the repair / clean-up
  sklearn  scikit-learn NMF (default and random start) and MiniBatchNMF (defaults) on the same sparse matrix
"""
import sys, os, json, time, warnings, numpy as np, scipy.sparse as sp, torch
from common import dev, laws
from nmf_torch import batch, online, error, health
path, outp, K = sys.argv[1], sys.argv[2], int(sys.argv[3]); seeds = [int(s) for s in sys.argv[4].split(",")]; group = sys.argv[5]
z = np.load(path)
def get(nm): return sp.csr_matrix((z[nm + "_data"], z[nm + "_indices"], z[nm + "_indptr"]), shape=tuple(z[nm + "_shape"]))
As, Ts = get("A"), get("T"); X = torch.from_numpy(As.toarray()).to(dev); Te = torch.from_numpy(Ts.toarray()).to(dev); n = X.shape[0]
have = {json.dumps([r[k] for k in ("K", "group", "name", "T", "seed")]) for r in map(json.loads, open(outp))} if os.path.exists(outp) else set()
f = open(outp, "a")
def log(**kw): s = json.dumps(dict(ds="nyt8k-tfidf", n=n, K=K, group=group, **kw)); print(s, flush=True); f.write(s + "\n"); f.flush()
def run(name, T, seed, **kw):
    if json.dumps([K, group, name, T, seed]) in have: return
    Ht, dt, info = online(X, K, T, seed, **kw); log(name=name, T=T, seed=seed, time=round(dt, 2), test=round(error(Te, Ht), 6), **info, **health(Ht, X))
for seed in seeds:
    if group == "batch":
        if json.dumps([K, group, "hals", 1, seed]) not in have:
            batch(X, K, int(sys.argv[6]), seed, lambda it, spent, Ht: log(name="hals", T=it, seed=seed, time=round(spent, 2), test=round(error(Te, Ht), 6), **(health(Ht, X) if it % 10 == 0 else {})))
    elif group == "main":
        for T in (1, 2, 3, 5, 10): run("recipe", T, seed)
    elif group == "ablate":
        B, a0 = laws(n, K, 3)
        for name, kw in (("constant step (peak value)", dict(schedule="constant")), ("no repair", dict(revive=False)), ("no last-pass clean-up", dict(final="none")),
                         ("B/4 and step/4", dict(B=B // 4, a0=a0 / 4)), ("B/4, same step", dict(B=B // 4)), ("step/4, same B", dict(a0=a0 / 4)), ("B x4, step 1", dict(B=B * 4, a0=1.0))):
            run(name, 3, seed, **kw)
    elif group == "textbook":
        for name, kw in (("running average (B=512)", dict(B=512, schedule="power", kappa=1.0, tau0=1.0)), ("power step (1+t)^-0.5 (B=512)", dict(B=512, schedule="power", kappa=0.5, tau0=1.0)),
                         ("power step (64+t)^-0.5 (B=4096)", dict(B=4096, schedule="power", kappa=0.5, tau0=64.0)), ("running average (formula batch)", dict(schedule="power", kappa=1.0, tau0=1.0))):
            for T in (1, 3):
                run(name, T, seed, revive=False, final="none", **kw); run(name + " + repair + clean-up", T, seed, **kw)
    elif group == "sklearn":
        from sklearn.decomposition import NMF, MiniBatchNMF
        warnings.filterwarnings("ignore")
        for name, mk in (("NMF default (nndsvda start, coordinate descent)", lambda: NMF(K, random_state=seed)), ("NMF random start", lambda: NMF(K, init="random", random_state=seed)),
                         ("MiniBatchNMF default", lambda: MiniBatchNMF(K, random_state=seed)), ("NMF random start, 25 iterations", lambda: NMF(K, init="random", random_state=seed, max_iter=25))):
            if json.dumps([K, group, name, 0, seed]) in have or (len(sys.argv) > 6 and sys.argv[6] not in name): continue
            t0 = time.time(); m = mk().fit(As); dt = time.time() - t0; Ht = torch.from_numpy(m.components_.T.astype(np.float32).copy()).to(dev)
            log(name=name, T=0, seed=seed, time=round(dt, 2), test=round(error(Te, Ht), 6), passes=int(m.n_iter_), **health(Ht, X))
