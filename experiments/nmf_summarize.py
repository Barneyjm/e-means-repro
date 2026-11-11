"""Usage: python nmf_summarize.py <log file> -- markdown tables: full-batch HALS curve, then every other run with the number
of full-batch passes needed to reach the same held-out error (same seed; '>N' = not reached in N passes)."""
import sys, json, collections, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
for K in sorted({r["K"] for r in rows}):
    rs = [r for r in rows if r["K"] == K]; curve = collections.defaultdict(dict)
    for r in rs:
        if r["group"] == "batch": curve[r["seed"]][r["T"]] = r
    print(f"\n### NYTimes TF-IDF, {rs[0]['n']:,} documents, K = {K} (held-out relative squared error, lower is better)\n")
    if curve:
        top = min(max(c) for c in curve.values()); print(f"Full-batch HALS, seeds {sorted(curve)}\n\n| passes | error | train s |\n|---|---|---|")
        for it in [i for i in (1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200) if i <= top]:
            v = [c[it] for c in curve.values()]; print(f"| {it} | {np.mean([x['test'] for x in v]):.5f} ± {np.std([x['test'] for x in v]):.5f} | {np.mean([x['time'] for x in v]):.1f} |")
    def match(r, key="test", ckey="test"):
        c = curve.get(r["seed"])
        if not c: return "–"
        return next((str(it) for it in sorted(c) if c[it][ckey] <= r[key]), f">{max(c)}")
    def tmatch(r):
        c = curve.get(r["seed"]); hit = next((c[it]["time"] for it in sorted(c) if c[it]["test"] <= r["test"]), None) if c else None
        return "–" if hit is None else f"{hit:.1f}"
    for group in ("main", "ablate", "textbook", "sklearn"):
        g = collections.defaultdict(list)
        for r in rs:
            if r["group"] == group: g[(r["name"], r["T"])].append(r)
        if not g: continue
        print(f"\n{group}\n\n| run | passes | seeds | error | train s | batch size | peak step | updates | repaired | dead at end | batch passes to match | batch seconds to match |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (name, T), v in g.items():
            v.sort(key=lambda r: r["seed"]); t = [x["test"] for x in v]; sk = group == "sklearn"
            print(f"| {name} | {np.mean([x['passes'] for x in v]):.0f} | " if sk else f"| {name} | {T} | ", end="")
            print(f"{len(v)} | {np.mean(t):.5f} ± {np.std(t):.5f} | {np.mean([x['time'] for x in v]):.1f} | {'–' if sk else v[0]['B']} | {'–' if sk or 'power' in name or 'running' in name else v[0]['a0']} | {'–' if sk else v[0]['updates']} | "
                  f"{'–' if sk else round(float(np.mean([x['revived'] for x in v])), 1)} | {np.mean([x['dead'] for x in v]):.1f} | {', '.join(match(x) for x in v)} | {', '.join(tmatch(x) for x in v)} |")
