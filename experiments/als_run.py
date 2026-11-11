"""Usage: python als_run.py <ratings.npy> <log file> <mode> <K> <lam> <seeds> [extra]

mode  lam      full ALS, 25 sweeps, for each ridge strength in `extra` (comma separated); run it on a split seed that is not
               used afterwards (e.g. 100) and pick the best
      main     full ALS for 40 sweeps with the held-out curve, then online at 1/2/3/5/10 passes with batch size and step
               from common.laws(users, items, passes)
      fixed    the same with one batch size and peak step for every pass count, `extra` = "<B>:<a0>"
      grid     online, batch size x peak step, at `extra` = "<passes>[:<B>,<B>,...]" (default 3 passes, B around the law's)
      ablate   online at `extra` = "<passes>:<B>:<a0>" with one ingredient changed at a time
Held-out RMSE on a random 10 % of ratings (split fixed by the seed). Finished configurations are skipped on a rerun.
"""
import sys, json, os, torch
from common import laws
from als_torch import Data, als, online, rmse, objective

path, outp, mode, K, lam = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), float(sys.argv[5])
seeds = [int(s) for s in sys.argv[6].split(",")]; extra = sys.argv[7] if len(sys.argv) > 7 else ""
name = os.path.basename(path).split(".")[0]
done = {json.loads(l)["key"] for l in open(outp)} if os.path.exists(outp) else set(); f = open(outp, "a")
def log(key, **kw):
    s = json.dumps(dict(key=key, data=name, K=K, **kw)); print(s, flush=True); f.write(s + "\n"); f.flush()
def run_als(d, seed, lam, sweeps, tag):
    key = f"{name}|{K}|{tag}|als|{lam}|{seed}|{sweeps}"
    if key in done: return None
    curve, times, obj = [], [], []
    als(d, K, sweeps, seed, lam, lambda it, s, P, Q: (curve.append(round(rmse(d, P, Q), 5)), times.append(round(s, 2)), obj.append(round(objective(d, P, Q, lam)[1], 5))))
    log(key, m="als", mode=tag, lam=lam, seed=seed, sweeps=sweeps, test=curve[-1], best=min(curve), best_at=curve.index(min(curve)) + 1, time=times[-1], curve=curve, times=times, obj=obj)
    return curve
def run_online(d, seed, T, tag, curve=None, label="default", **kw):
    key = f"{name}|{K}|{tag}|online|{lam}|{seed}|{T}|{label}"
    if key in done: return
    P, Q, dt, info = online(d, K, T, seed, lam, **kw); e = rmse(d, P, Q); tr, ob = objective(d, P, Q, lam)
    log(key, m="online", mode=tag, lam=lam, seed=seed, T=T, variant=label, time=round(dt, 2), test=round(e, 5), train=round(tr, 5), obj=round(ob, 5), **info,
        **({"sweeps_to_match": next((i + 1 for i, c in enumerate(curve) if c <= e), None)} if curve else {}))
for seed in seeds:
    d = Data(path, seed)
    if seed == seeds[0]: als(d, K, 1, 99, lam); online(d, K, 1, 99, lam, B=d.nu // 4)             # warm up kernels
    if mode == "lam":
        for l in extra.split(","): run_als(d, seed, float(l), 25, "lam")
    elif mode in ("main", "fixed"):
        sweeps = 40; curve = run_als(d, seed, lam, sweeps, "main")
        if curve is None: curve = next(json.loads(l) for l in open(outp) if json.loads(l)["key"] == f"{name}|{K}|main|als|{lam}|{seed}|{sweeps}")["curve"]
        kw = dict(B=int(extra.split(":")[0]), a0=float(extra.split(":")[1])) if mode == "fixed" else {}
        for T in (1, 2, 3, 5, 10): run_online(d, seed, T, mode + extra, curve, **kw)
    elif mode == "grid":
        T, _, Bs = (extra or "3").partition(":"); T = int(T); B0 = laws(d.nu, d.ni, T)[0]
        for B in [int(x) for x in Bs.split(",")] if Bs else (B0 // 8, B0 // 4, B0 // 2, B0, B0 * 2):
            for a in (0.03, 0.1, 0.25, 0.5, 1.0): run_online(d, seed, T, "grid", label=f"B={B},a0={a}", B=B, a0=a)
    else:
        T, B, a = extra.split(":"); T, B, a = int(T), int(B), float(a); tag = f"ablate{T}:{B}:{a}"
        for label, kw in (("default", {}), ("constant step", dict(schedule="constant")), ("fade only touched items", dict(decay="touched")),
                          ("no last-pass clean-up", dict(final="none")), ("re-solve items every 5 batches", dict(resolve=5)),
                          ("textbook step (t+2)^-0.5", dict(schedule="stepwise", kappa=0.5, final="none")),
                          ("textbook step (t+2)^-0.75", dict(schedule="stepwise", kappa=0.75, final="none")),
                          ("textbook step (t+2)^-1.0", dict(schedule="stepwise", kappa=1.0, final="none"))):
            run_online(d, seed, T, tag, label=label, B=B, a0=a, **kw)
