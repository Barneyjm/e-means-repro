"""Usage: python summarize_als.py <log from als_run.py> [sgd log]. Markdown tables per dataset and rank."""
import sys, json, collections, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]; sgd = [json.loads(l) for l in open(sys.argv[2])] if len(sys.argv) > 2 else []
ms = lambda v, p=4: f"{np.mean(v):.{p}f}" + (f" ± {np.std(v):.{p}f}" if len(v) > 1 else "")
for data, K in sorted({(r["data"], r["K"]) for r in rows}):
    rs = [r for r in rows if r["data"] == data and r["K"] == K]; A = {r["seed"]: r for r in rs if r["m"] == "als" and r["mode"] == "main"}
    if A:
        seeds = sorted(A); c = np.array([A[s]["curve"] for s in seeds]); t = np.array([A[s]["times"] for s in seeds]).mean(0); ob = np.array([A[s]["obj"] for s in seeds]).mean(0)
        print(f"\n### {data}, rank {K}, ridge {A[seeds[0]]['lam']}, seeds {seeds}\n\n| method | passes | settings | seconds | held-out RMSE | vs ALS 40 | ALS sweeps to match (per seed) | training objective |\n|---|---|---|---|---|---|---|---|")
        for sw in (1, 2, 3, 5, 10, 15, 20, 30, 40): print(f"| ALS | {sw} | | {t[sw-1]:.2f} | {ms(c[:, sw-1])} | {np.mean(c[:, sw-1] - c[:, -1]):+.4f} | | {ob[sw-1]:.4f} |")
        print(f"| ALS, best sweep of 40 | {[int(np.argmin(x)) + 1 for x in c]} | | | {ms(c.min(1))} | {np.mean(c.min(1) - c[:, -1]):+.4f} | | |")
        for mode in sorted({r["mode"] for r in rs if r["m"] == "online" and r["mode"].startswith(("main", "fixed"))}, key=lambda m: (not m.startswith("main"), m)):
            for T in (1, 2, 3, 5, 10):
                d = [r for r in rs if r["m"] == "online" and r["mode"] == mode and r["T"] == T and r["seed"] in A]
                if d: print(f"| online ({'formula' if mode == 'main' else 'fixed'}) | {T} | B={d[0]['B']}, a0={d[0]['a0']} | {np.mean([r['time'] for r in d]):.2f} | {ms([r['test'] for r in d])} | {np.mean([r['test'] - A[r['seed']]['curve'][-1] for r in d]):+.4f} | {[r['sweeps_to_match'] for r in d]} | {np.mean([r['obj'] for r in d]):.4f} |")
        s = [r for r in sgd if r["data"] == data and r["K"] == K and np.isfinite(r["best"])]
        if s: b = min(s, key=lambda r: r["best"]); print(f"| plain SGD, best of {len(s)} settings (lr {b['lr']}, reg {b['reg']}), seed {b['seed']} | {b['epochs']} | batch {b['batch']} ratings | {b['time']:.1f} | {b['best']:.4f} | | | |")
    for T in sorted({r["T"] for r in rs if r["mode"] == "grid"}):
        g = {(r["B"], r["a0"]): r for r in rs if r["mode"] == "grid" and r["T"] == T}; Bs = sorted({k[0] for k in g}); As = sorted({k[1] for k in g})
        print(f"\n{data}, rank {K}: grid at {T} pass(es), seed {next(iter(g.values()))['seed']}, held-out RMSE (seconds)\n\n| batch size | " + " | ".join(f"step {a}" for a in As) + " |\n|---|" + "---|" * len(As))
        for B in Bs: print(f"| {B} | " + " | ".join(f"{g[(B, a)]['test']:.4f} ({g[(B, a)]['time']:.1f})" if (B, a) in g else "" for a in As) + " |")
    for mode in sorted({r["mode"] for r in rs if r["mode"].startswith("ablate")}):
        d = [r for r in rs if r["mode"] == mode]; base = {r["seed"]: r["test"] for r in d if r["variant"] == "default"}; T, B, a = mode[6:].split(":")
        print(f"\n{data}, rank {K}: {T} pass(es), B={B}, a0={a}, seeds {sorted(base)}\n\n| variant | held-out RMSE | vs default | training objective | seconds |\n|---|---|---|---|---|")
        for v in dict.fromkeys(r["variant"] for r in d):
            x = [r for r in d if r["variant"] == v and r["seed"] in base]; print(f"| {v} | {ms([r['test'] for r in x])} | {np.mean([r['test'] - base[r['seed']] for r in x]):+.4f} ± {np.std([r['test'] - base[r['seed']] for r in x]):.4f} | {np.mean([r['obj'] for r in x]):.4f} | {np.mean([r['time'] for r in x]):.2f} |")
