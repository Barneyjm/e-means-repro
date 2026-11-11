"""Usage: python summarize_gmm.py <gmm log>. Held-out log-likelihood per row (higher is better), mean over seeds."""
import sys, json, collections, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
cfgs = sorted({(r["ds"], r["K"]) for r in rows})
for ds, K in cfgs:
    rs = [r for r in rows if (r["ds"], r["K"]) == (ds, K)]
    seeds = sorted({r["seed"] for r in rs if "seed" in r})
    curves = {r["seed"]: r["test"] for r in rs if r["m"] == "em_curve"}
    print(f"\n### {ds}, K = {K}, seeds {seeds}\n")
    for r in rs:
        if r["m"] == "truth": print(f"True model: held-out {r['test']:.4f}\n")
    if curves:
        best = {s: max(c) for s, c in curves.items()}
        print("| method | passes | train s | held-out log-lik | vs EM 100 | train log-lik | EM passes to match (per seed) | small comps |"); print("|---|---|---|---|---|---|---|---|")
        em100 = {r["seed"]: r["test"] for r in rs if r["m"] == "em" and r["iters"] == 100}
        def line(name, passes, sel, extra=""):
            d = {r["seed"]: r for r in sel}
            if not d: return
            m = lambda key: np.mean([d[s][key] for s in d])
            gap = np.array([d[s]["test"] - em100[s] for s in d])
            print(f"| {name} | {passes} | {m('time'):.2f} | {m('test'):.4f} | {gap.mean():+.4f} ± {gap.std(ddof=1) if len(gap) > 1 else 0:.4f} | {m('train'):.4f} | {extra if extra else [d[s].get('em_passes_to_match') for s in sorted(d)]} | {m('n_small'):.1f} |")
        for it in (10, 25, 50, 100, 200): line("EM", it, [r for r in rs if r["m"] == "em" and r["iters"] == it], "–")
        for T in (1, 2, 3, 5, 10): line("online", T, [r for r in rs if r["m"] == "online" and r["T"] == T])
    ab = collections.defaultdict(dict)
    for r in rs:
        if r["m"] == "ablate": ab[r["variant"]][r["seed"]] = r
    if ab:
        base = ab["default"]
        print("\n| 3-pass variant | held-out log-lik | vs default | revived | small comps |"); print("|---|---|---|---|---|")
        for name, d in ab.items():
            gap = np.array([d[s]["test"] - base[s]["test"] for s in d])
            print(f"| {name} | {np.mean([d[s]['test'] for s in d]):.4f} | {gap.mean():+.4f} ± {gap.std(ddof=1) if len(gap) > 1 else 0:.4f} | {np.mean([d[s]['revived'] for s in d]):.0f} | {np.mean([d[s]['n_small'] for s in d]):.1f} |")
