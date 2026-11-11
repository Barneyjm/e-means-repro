"""Usage: python lda_summarize.py <log file> [n docs] -- markdown tables: batch curve, then every online run with the
number of full-batch passes needed to reach the same held-out perplexity (same seed; '>N' = not reached in N passes)."""
import sys, json, collections, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]; nsel = int(sys.argv[2]) if len(sys.argv) > 2 else None
for n, K in sorted({(r["n"], r["K"]) for r in rows}):
    if nsel and n != nsel: continue
    rs = [r for r in rows if r["n"] == n and r["K"] == K]; curve = collections.defaultdict(dict)
    for r in rs:
        if r["group"] == "batch": curve[r["seed"]][r["T"]] = r
    print(f"\n### NYTimes, {n:,} documents, K = {K}\n")
    if curve:
        top = min(max(c) for c in curve.values()); print(f"Full-batch variational EM, seeds {sorted(curve)} (held-out perplexity, lower is better)\n\n| passes | perplexity | train s | dead topics |\n|---|---|---|---|")
        for it in [i for i in (1, 2, 3, 5, 10, 15, 20, 30, 40, 60, 80, 100, 150, 200) if i <= top]:
            v = [c[it] for c in curve.values()]; print(f"| {it} | {np.mean([x['test'] for x in v]):.1f} ± {np.std([x['test'] for x in v]):.1f} | {np.mean([x['time'] for x in v]):.1f} | {np.mean([x['dead'] for x in v]):.1f} |")
    def match(r):
        c = curve.get(r["seed"])
        if not c: return "–"
        return next((str(it) for it in sorted(c) if c[it]["test"] <= r["test"]), f">{max(c)}")
    for group in ("main", "tuned", "hoffman", "ablate", "classic", "grid", "ridge"):
        g = collections.defaultdict(list)
        for r in rs:
            if r["group"].startswith(group) and r["group"] != "batch": g[(r["name"], r["T"])].append(r)
        if not g: continue
        print(f"\n{group}\n\n| run | passes | seeds | perplexity | train s | batch size | peak step | updates | repaired | merged | dead at end | batch passes to match |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (name, T), v in g.items():
            v.sort(key=lambda r: r["seed"]); t = [x["test"] for x in v]
            print(f"| {name} | {T} | {len(v)} | {np.mean(t):.1f} ± {np.std(t):.1f} | {np.mean([x['time'] for x in v]):.1f} | {v[0]['B']} | {v[0]['a0'] if 'tau0' not in name else '–'} | {v[0]['updates']} | "
                  f"{np.mean([x['revived'] for x in v]):.1f} | {np.mean([x['merged'] for x in v]):.1f} | {np.mean([x['dead'] for x in v]):.1f} | {', '.join(match(x) for x in v)} |")
