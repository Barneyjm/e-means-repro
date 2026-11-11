"""Usage: python run_lab.py <base> <queries> <neighbours.npy> <log file> <k> <seeds> <variants file> [n rows]
Runs emeans_lab variants and scores each one: train/test MSE and search-index cost (see ivf_recall.py).
Variants file: one per line, `name | key=value, key=value`. Keys are emeans_x options plus T (passes, default 3),
lloyd=<iterations> for the baseline, ladder=<levels> and phi for codebook doubling, Bx / ax to scale the
formula's batch size / step. Lines starting with # are skipped.
Set LAB_SAVE=<dir> to also save each set of centroids (for faiss_ivf_bench.py)."""
import sys, os, json, numpy as np, torch
from common import dev, laws
from emeans_torch import lloyd
from emeans_lab import emeans_x, ladder, lloyd_x
from ivf_sweep import load, score
from ivf_recall import evaluate

def parse(line):
    name, _, spec = line.partition("|"); kw = {}
    for item in filter(None, (s.strip() for s in spec.split(","))):
        key, val = item.split("=")
        try: val = int(val)
        except ValueError:
            try: val = float(val)
            except ValueError: val = {"True": True, "False": False}.get(val, val)
        kw[key.strip()] = val
    return name.strip(), kw

def fit(Xd, X, k, seed, kw):
    kw = dict(kw); T = kw.pop("T", 3)
    if "lloyd" in kw: return lloyd_x(Xd, X, k, kw["lloyd"], seed, kw.get("consc", 0.0), kw.get("init", "random")) if kw.get("consc") or kw.get("init") else lloyd(Xd, X, k, kw["lloyd"], seed) + ({},)
    n = len(X); Bl, al = laws(n, k, T)
    if "Bx" in kw: kw["B"] = max(256, int(Bl * kw.pop("Bx")))
    if "ax" in kw: kw["a0"] = min(al * kw.pop("ax"), 1.0)
    if "ladder" in kw: return ladder(Xd, X, k, T, seed, levels=kw.pop("ladder"), **kw)
    return emeans_x(Xd, X, k, T, seed, **kw)

if __name__ == "__main__":
    bp, qp, gp, outp, k = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
    seeds = [int(s) for s in sys.argv[6].split(",")]; variants = [parse(l) for l in open(sys.argv[7]) if l.strip() and not l.startswith("#")]
    X = load(bp); X = X[:int(sys.argv[8])] if len(sys.argv) > 8 else X
    gt = torch.from_numpy(np.load(gp)).long(); Q = load(qp)[:len(gt)]
    Xd = torch.from_numpy(X).to(dev); Qd = torch.from_numpy(Q).to(dev); Xte = Q[:10000]; full = len(sys.argv) <= 8
    done = {(r["k"], r["v"], r["seed"], r["n"]) for r in map(json.loads, open(outp))} if os.path.exists(outp) else set()   # resume
    f = open(outp, "a"); emeans_x(Xd, X, k, 1, 99, util_tau=0.01, donor="error")          # warm up GPU kernels
    for seed in seeds:
        for name, kw in variants:
            if (k, name, seed, len(X)) in done: continue
            C, dt, info = fit(Xd, X, k, seed, kw)
            if os.environ.get("LAB_SAVE"): np.save(f"{os.environ['LAB_SAVE']}/{bp.split('/')[-1].split('.')[0]}_k{k}_{name.replace('/', '_')}_s{seed}.npy", C)
            Cd = torch.from_numpy(C).to(dev); q = Qd[:10000]                     # distance to the nearest centroid, and the gap to the second
            v = torch.topk((q * q).sum(1)[:, None] + (Cd * Cd).sum(1)[None] - 2 * q @ Cd.T, 2, dim=1, largest=False).values
            info = dict(info, q_d1=round(float(v[:, 0].mean()), 4), q_gap=round(float((v[:, 1] - v[:, 0]).mean()), 4))
            s = json.dumps(dict(data=bp.split("/")[-1], n=len(X), d=X.shape[1], k=k, v=name, seed=seed, time=round(dt, 2), **info,
                                **score(Xd, X, Xte, C), **(evaluate(Xd, Qd, gt, C) if full else {})))
            print(s, flush=True); f.write(s + "\n"); f.flush()
