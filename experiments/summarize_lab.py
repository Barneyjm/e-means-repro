"""Usage: python summarize_lab.py <lab log> [reference variant, default e3]
Per k: every variant against the reference, paired by seed (mean change +- SD over seeds)."""
import sys, json, collections, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]; ref = sys.argv[2] if len(sys.argv) > 2 else "e3"
def rel(d, b, key): return [100 * (d[s][key] / b[s][key] - 1) for s in d if s in b and d[s].get(key) and b[s].get(key)]
def fmt(v): return f"{np.mean(v):+.2f}% ± {np.std(v):.2f}" if len(v) else ""
for key in sorted({(r["data"], r["n"], r["k"]) for r in rows}):
    rs = [r for r in rows if (r["data"], r["n"], r["k"]) == key]; by = collections.defaultdict(dict)
    for r in rs: by[r["v"]][r["seed"]] = r
    if ref not in by: continue
    b = by[ref]; print(f"\n### {key[0]}, n = {key[1]:,}, k = {key[2]:,}; reference = {ref} ({len(b)} seeds)\n")
    print("| variant | seeds | train s | test MSE | train MSE | scan for R@1 >= 0.95 | scan for R@10 >= 0.95 | imbalance | revived |"); print("|---|---|---|---|---|---|---|---|---|")
    for v, d in by.items():
        m = lambda k_: np.mean([x[k_] for x in d.values() if x.get(k_) is not None]) if any(x.get(k_) is not None for x in d.values()) else float("nan")
        if v == ref: print(f"| **{v}** | {len(d)} | {m('time'):.2f} | {m('test'):,.1f} | {m('train'):,.1f} | {m('cost_r1>=0.95'):,.0f} | {m('cost_r10>=0.95'):,.0f} | {m('imbalance'):.3f} | {m('revived'):.0f} |")
        else: print(f"| {v} | {len(d)} | {m('time'):.2f} | {fmt(rel(d, b, 'test'))} | {fmt(rel(d, b, 'train'))} | {fmt(rel(d, b, 'cost_r1>=0.95'))} | {fmt(rel(d, b, 'cost_r10>=0.95'))} | {m('imbalance'):.3f} | {m('revived'):.0f} |")
