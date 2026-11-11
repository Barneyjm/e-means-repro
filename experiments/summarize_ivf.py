"""Usage: python summarize_ivf.py <ivf log>. Mean over seeds; cost columns are % change vs Lloyd 25 (paired by seed, ± SD)."""
import sys, json, collections, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
by = collections.defaultdict(dict)
for r in rows: by[(r["k"], r["m"])][r["seed"]] = r
cols = ["cost_r1>=0.9", "cost_r1>=0.95", "cost_r1>=0.99", "cost_r10>=0.95"]
print("| k | method | seeds | train s | test MSE | imbalance | R@1 nprobe 8 | cost@8 | " + " | ".join("scan for " + c[5:] for c in cols) + " |")
print("|---|---|---|---|---|---|---|---|" + "---|" * len(cols))
for (k, m), d in sorted(by.items(), key=lambda kv: (kv[0][0], kv[0][1] != "lloyd10", kv[0][1] != "lloyd25", kv[0][1])):
    base = by[(k, "lloyd25")]; seeds = sorted(d)
    mean = lambda key: np.mean([d[s][key] for s in seeds])
    cells = []
    mse = f"{mean('test'):,.1f}" if m == 'lloyd25' else f"{np.mean([d[s]['test'] / base[s]['test'] - 1 for s in seeds]) * 100:+.2f}%"
    for c in cols:
        if m == "lloyd25": cells.append(f"{mean(c):,.0f}")
        else:
            rel = np.array([d[s][c] / base[s][c] - 1 for s in seeds]) * 100
            cells.append(f"{rel.mean():+.2f}% ± {rel.std(ddof=1):.2f}")
    print(f"| {k} | {m} | {len(seeds)} | {mean('time'):.2f} | {mse} | {mean('imbalance'):.3f} | {mean('r1@8'):.4f} | {mean('cost@8'):,.0f} | " + " | ".join(cells) + " |")
