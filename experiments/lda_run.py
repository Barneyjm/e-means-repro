"""Usage: python lda_run.py <nyt8k.npz> <log file> <n docs> <K> <seeds> <group> [passes for batch | library filter for classic]

Groups (every finished run is appended to the log as one JSON line and skipped when the command is repeated):
  batch    full-batch variational EM, held-out perplexity after every pass (checkpointed every 10 passes)
  main     online with the e-means recipe, batch size and step from the paper's formulas, 1/2/3/5/10 passes
  tuned    the same with the step divided by 4 (the formula's step is too large for topics), 1/2/3/5/10 passes
  hoffman  Hoffman's rule at his best published setting, no repair or clean-up, 1/2/3/5/10 passes
  ablate   3 passes at the step/4 setting, one ingredient changed at a time; also a start from documents
  classic  Hoffman's step rule (tau0 + t)^-kappa at library and published settings, with and without the repair, 1 and 3 passes
  grid     3 passes, batch size x peak step around the formula's choice (grid0.25 = only the cells with step 0.25)
  ridge    3 passes, more cells at small steps, chosen by memory length (batch size / step)
"""
import sys, os, json, time, torch
from common import dev, sync, laws
from lda_torch import load, dexp, totals, perplexity, init, online, health
path, outp, n, K = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]); seeds = [int(s) for s in sys.argv[5].split(",")]; group = sys.argv[6]
Xu, Ho, Ht = load(path, n); n = Xu.shape[0]; a = e = 1.0 / K
have = {json.dumps([r[k] for k in ("n", "K", "group", "name", "T", "seed")]) for r in map(json.loads, open(outp))} if os.path.exists(outp) else set()
f = open(outp, "a"); only = [int(t) for t in os.environ.get("LDA_PASSES", "").split(",") if t]      # optional: restrict the pass counts
def log(**kw): s = json.dumps(dict(ds="nyt8k", n=n, K=K, group=group, **kw)); print(s, flush=True); f.write(s + "\n"); f.flush()
def run(name, T, seed, **kw):
    if json.dumps([n, K, group, name, T, seed]) in have or (only and T not in only): return
    lam, dt, info = online(Xu, K, T, seed, a, e, **kw)
    log(name=name, T=T, seed=seed, time=round(dt, 2), test=round(perplexity(lam, Ho, Ht, a), 3), **info, **health(lam, e))
estep_warm = totals(Xu, torch.arange(512, device=dev), dexp(init(K, Xu.shape[1], 99)), a, max_iter=3)
for seed in seeds:
    if group == "batch":
        iters = int(sys.argv[7]); ck = f"{outp}.n{n}.K{K}.s{seed}.ckpt"
        lam, it0, spent = torch.load(ck) if os.path.exists(ck) else (init(K, Xu.shape[1], seed), 0, 0.0); lam = lam.to(dev)
        for it in range(it0 + 1, iters + 1):
            sync(); t0 = time.time(); ss, its = totals(Xu, slice(None), dexp(lam), a); lam = e + ss; sync(); spent += time.time() - t0
            log(name="em", T=it, seed=seed, time=round(spent, 2), test=round(perplexity(lam, Ho, Ht, a), 3), estep_iters=its, **health(lam, e))
            if it % 10 == 0 or it == iters: torch.save((lam.cpu(), it, spent), ck)
    elif group == "main":
        for T in (1, 2, 3, 5, 10): run("recipe", T, seed)
    elif group == "tuned":
        for T in (1, 2, 3, 5, 10): run("recipe, step/4", T, seed, a0=laws(n, K, T)[1] / 4)
    elif group == "hoffman":
        for T in (1, 2, 3, 5, 10): run("Hoffman 2010 best (B=4096, kappa=0.5, tau0=64)", T, seed, B=4096, schedule="hoffman", kappa=0.5, tau0=64.0, revive=False, final="none")
    elif group == "ablate":
        a4 = laws(n, K, 3)[1] / 4
        for name, kw in (("constant step (mean of the cosine curve)", dict(schedule="constant", a0=0.55 * a4)), ("no repair", dict(revive=False)), ("no last-pass clean-up", dict(final="none")),
                         ("no duplicate merging", dict(dup=1.0)), ("tiny split (2^-7), no duplicate merging", dict(delta=2 ** -7, dup=1.0)),
                         ("start from documents (half document, half corpus)", dict(mix=0.5)), ("start from documents (10% document, 90% corpus)", dict(mix=0.9))):
            run(name, 3, seed, **{"a0": a4, **kw})
    elif group == "classic":
        for lib, B, kp, t0 in (("Hoffman 2010 best", 4096, 0.5, 64.0), ("gensim defaults", 2048, 0.5, 1.0), ("scikit-learn defaults", 128, 0.7, 10.0), ("Hoffman 2013 SVI", 512, 0.9, 1.0)):
            if len(sys.argv) > 7 and sys.argv[7] not in lib: continue
            for T in (1, 3):
                for tag, kw in (("", dict(revive=False, final="none")), (" + repair", dict(revive=True, final="none")), (" + repair + clean-up", dict(revive=True))):
                    if tag == " + repair + clean-up" and lib not in ("Hoffman 2010 best", "scikit-learn defaults"): continue
                    run(f"{lib} (B={B}, kappa={kp}, tau0={t0:g}){tag}", T, seed, B=B, schedule="hoffman", kappa=kp, tau0=t0, **kw)
    elif group == "ridge":
        for B, a0 in ((256, 1/128), (1024, 1/32), (1024, 1/64), (1024, 1/16), (4096, 1/16), (16384, 1.0)):
            run(f"grid B={B}, step {a0:.4g} (memory {B / a0 / 1000:.0f}k documents)", 3, seed, B=B, a0=a0)
    elif group.startswith("grid"):
        B0, _ = laws(n, K, 3)
        for bm in (1/16, 1/4, 1, 4):
            for a0 in (0.06, 0.125, 0.25, 0.5, 1.0):
                if group in ("grid", "grid" + str(a0)): run(f"grid B x{bm:g}, step {a0}", 3, seed, B=int(B0 * bm), a0=a0)
