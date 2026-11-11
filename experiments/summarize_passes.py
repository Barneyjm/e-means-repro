"""Usage: python summarize_passes.py <ivf_sweep log>. Quality against the number of passes, relative to Lloyd 25 (paired by seed)."""
import sys, json, collections, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
for key in sorted({(r["data"], r["k"]) for r in rows}):
    rs = [r for r in rows if (r["data"], r["k"]) == key]; by = collections.defaultdict(dict)
    for r in rs: by[r["m"]][r["seed"]] = r
    b = by["lloyd25"]; print(f"\n### {key[0]}, k = {key[1]:,} ({len(b)} seeds; change vs Lloyd 25, mean ± SD over seeds)\n")
    print("| method | seeds | train s | B | step | test MSE | train MSE | scan for R@1 >= 0.95 | scan for R@10 >= 0.95 | imbalance |"); print("|---|---|---|---|---|---|---|---|---|---|")
    for m, d in by.items():
        def rel(k_): v = [100 * (d[s][k_] / b[s][k_] - 1) for s in d if s in b and d[s][k_] is not None and b[s][k_] is not None]; return f"{np.mean(v):+.2f}% ± {np.std(v):.2f}" if v else "not reached"
        mean = lambda k_: np.mean([x[k_] for x in d.values() if x[k_] is not None] or [float('nan')]); x0 = next(iter(d.values()))
        if m == "lloyd25": print(f"| lloyd25 | {len(d)} | {mean('time'):.2f} | | | {mean('test'):,.0f} | {mean('train'):,.0f} | {mean('cost_r1>=0.95'):,.0f} | {mean('cost_r10>=0.95'):,.0f} | {mean('imbalance'):.3f} |")
        else: print(f"| {m} | {len(d)} | {mean('time'):.2f} | {x0.get('B', '')} | {x0.get('a0', '')} | {rel('test')} | {rel('train')} | {rel('cost_r1>=0.95')} | {rel('cost_r10>=0.95')} | {mean('imbalance'):.3f} |")
