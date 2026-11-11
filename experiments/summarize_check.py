"""Usage: python summarize_check.py <dir with one folder per dataset: train_k<k>.jsonl (centroid_sets.py) + index_k<k>.jsonl (faiss_ivf_bench.py)>
One table per dataset and k: every variant against Faiss k-means with 25 iterations, paired by seed (mean ± sd over seeds)."""
import sys, os, re, glob, json, collections, numpy as np
NAMES = {"faiss25": "Faiss k-means, 25 iterations", "auth": "authors' e-means", "mine": "my estimator"}
def label(v):
    m = re.match(r"(auth|mine)(\d+)(?:\+([\d.]+)(h?))?$", v)
    if not m: return NAMES.get(v, v)
    return f"{NAMES[m.group(1)]}, {m.group(2)} passes" + (f" + penalty {m.group(3)}, {'held' if m.group(4) else 'faded'}" if m.group(3) else "")
def pm(c, unit="%"): return (f"{np.mean(c):+.1f}{unit} ± {np.std(c):.1f}" if len(c) > 1 else f"{c[0]:+.1f}{unit}") if len(c) else ""
for tp in sorted(glob.glob(os.path.join(sys.argv[1], "*", "train_k*.jsonl"))):
    k = int(re.search(r"train_k(\d+)", tp).group(1)); ip = tp.replace("train_k", "index_k"); T = collections.defaultdict(dict); I = collections.defaultdict(dict)
    for r in map(json.loads, open(tp)): T[r["m"]][r["seed"]] = r
    if os.path.exists(ip):
        for r in map(json.loads, open(ip)): m = re.match(r".*_k\d+_(.*)_s(\d+)\.npy", r["centroids"]); I[m.group(1)][int(m.group(2))] = r
    r0 = next(iter(T["faiss25"].values())); ref_t, ref_i = T["faiss25"], I.get("faiss25", {})
    print(f"\n### {r0['data'].replace('_base', '')}: {r0['n']:,} vectors, {r0['d']} dimensions, k = {k:,}\n")
    print("| Centroids from | Seeds | Training time (s) | Held-out error | List imbalance | Distance computations, 95% R@1 | Change | Query time (µs) | Change |"); print("|---|---|---|---|---|---|---|---|---|")
    order = ["faiss25"] + sorted((v for v in T if v != "faiss25"), key=lambda v: (v[:4], "+" in v, v))
    for v in order:
        t, i = T[v], I.get(v, {}); key, us = "ndis_r1>=0.95", "us_r1>=0.95"
        rel = lambda d, b, f: [100 * (f(d[s]) / f(b[s]) - 1) for s in d if s in b and f(d[s]) and f(b[s])]
        nd = [i[s][key] for s in i if i[s][key]]; ut = [i[s][us] for s in i if i[s][us]]
        print(f"| {label(v)} | {len(t)} | {np.mean([r['time'] for r in t.values()]):.1f} | {'–' if v == 'faiss25' else pm(rel(t, ref_t, lambda r: r['test']))} | {np.mean([r['imbalance'] for r in t.values()]):.2f} | "
              f"{np.mean(nd):,.0f} | {'–' if v == 'faiss25' else pm(rel(i, ref_i, lambda r: r[key]))} | {np.mean(ut):.0f} | {'–' if v == 'faiss25' else pm(rel(i, ref_i, lambda r: r[us]))} |" if nd else
              f"| {label(v)} | {len(t)} | {np.mean([r['time'] for r in t.values()]):.1f} | {'–' if v == 'faiss25' else pm(rel(t, ref_t, lambda r: r['test']))} | {np.mean([r['imbalance'] for r in t.values()]):.2f} | | | | |")
