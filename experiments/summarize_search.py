"""Usage: python summarize_search.py [--compact] <lab logs ...>. One block per dataset and k (or with --compact one row each): the standard index (Lloyd 25) against
e-means and e-means with the size penalty. Changes are against Lloyd 25, paired by seed (mean ± SD over seeds)."""
import sys, json, collections, numpy as np
compact = "--compact" in sys.argv; rows = [json.loads(l) for f in sys.argv[1:] if f != "--compact" for l in open(f)]
show = [("lloyd25", "Lloyd, 25 passes"), ("e1", "e-means, 1 pass"), ("e2", "e-means, 2 passes"), ("e3", "e-means, 3 passes"), ("e5", "e-means, 5 passes"),
        ("ca0.3", "3 passes + size penalty 0.3, faded out"), ("cc0.1", "3 passes + size penalty 0.1, held"), ("cc0.3", "3 passes + size penalty 0.3, held"),
        ("ca0.3,T5", "5 passes + size penalty 0.3, faded out"), ("cc0.3,T5", "5 passes + size penalty 0.3, held")]
if compact:
    print("| data | dim | vectors | k | seeds | recall target | Lloyd 25: vectors scanned | e-means 3 | e-means 5 | 3 + penalty 0.3 faded | 3 + penalty 0.1 held | 3 + penalty 0.3 held | list imbalance: Lloyd / e-means 3 / faded |"); print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for key in sorted({(r["data"], r["n"], r["k"]) for r in rows if r["n"] >= 50_000}):
    by = collections.defaultdict(dict)
    for r in rows:
        if (r["data"], r["n"], r["k"]) == key: by[r["v"]][r["seed"]] = r      # later logs win
    if "lloyd25" not in by: continue
    b = by["lloyd25"]; mean = lambda d, k_: np.mean([x[k_] for x in d.values() if x.get(k_) is not None]) if any(x.get(k_) is not None for x in d.values()) else float("nan")
    if compact:
        tgt = next((t for t in ("cost_r1>=0.95", "cost_r1>=0.9") if all(x.get(t) is not None for v in ("lloyd25", "e3") for x in by.get(v, {}).values())), None)
        def rel(v):
            d = by.get(v, {}); s_ = [x for x in d if x in b and d[x].get(tgt) is not None]
            if not s_ or tgt is None: return ""
            c = [100 * (d[x][tgt] / b[x][tgt] - 1) for x in s_]; return f"{np.mean(c):+.1f}% ± {np.std(c):.1f}" if len(s_) > 1 else f"{c[0]:+.1f}%"
        imb = " / ".join(f"{mean(by[v], 'imbalance'):.2f}" for v in ("lloyd25", "e3", "ca0.3") if v in by)
        print(f"| {key[0].split('_')[0]} | {next(iter(b.values()))['d']} | {key[1]:,} | {key[2]:,} | {min(len(by.get(v, {})) for v in ('lloyd25', 'e3', 'ca0.3'))} | {'95 % R@1' if tgt == 'cost_r1>=0.95' else '90 % R@1'} | {mean(b, tgt):,.0f} | " + " | ".join(rel(v) for v in ("e3", "e5", "ca0.3", "cc0.1", "cc0.3")) + f" | {imb} |"); continue
    print(f"\n### {key[0].split('_')[0]}, n = {key[1]:,}, d = {next(iter(b.values()))['d']}, k = {key[2]:,}\n")
    print("| centroids from | seeds | train s | vectors scanned, 90 % R@1 | 95 % R@1 | 90 % R@10 | 95 % R@10 | test MSE | list imbalance |"); print("|---|---|---|---|---|---|---|---|---|")
    for v, label in show:
        d = by.get(v)
        if not d: continue
        def rel(k_):
            s = [x for x in d if x in b and d[x].get(k_) is not None and b[x].get(k_) is not None]
            if not s: return "not reached"
            c = [100 * (d[x][k_] / b[x][k_] - 1) for x in s]; return f"{np.mean(c):+.1f}% ± {np.std(c):.1f}" if len(s) > 1 else f"{c[0]:+.1f}%"
        if v == "lloyd25": print(f"| {label} | {len(d)} | {mean(d, 'time'):.2f} | {mean(d, 'cost_r1>=0.9'):,.0f} | {mean(d, 'cost_r1>=0.95'):,.0f} | {mean(d, 'cost_r10>=0.9'):,.0f} | {mean(d, 'cost_r10>=0.95'):,.0f} | {mean(d, 'test'):.4g} | {mean(d, 'imbalance'):.3f} |")
        else: print(f"| {label} | {len(d)} | {mean(d, 'time'):.2f} | {rel('cost_r1>=0.9')} | {rel('cost_r1>=0.95')} | {rel('cost_r10>=0.9')} | {rel('cost_r10>=0.95')} | {rel('test')} | {mean(d, 'imbalance'):.3f} |")
