"""Usage: python summarize_faiss.py <faiss_ivf_bench log> [reference, default faiss25]. Real-index cost per set of centroids
(mean over seeds; change against the reference paired by seed). File names: <data>_k<k>_<variant>_s<seed>.npy"""
import sys, json, re, collections, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]; ref = sys.argv[2] if len(sys.argv) > 2 else "faiss25"; g = collections.defaultdict(dict)
for r in rows:
    m = re.match(r".*_k\d+_(.*)_s(\d+)\.npy", r["centroids"]); g[m.group(1)][m.group(2)] = r
base = g[ref]; i16 = rows[0]["probes"].index(16)
print(f"| centroids | seeds | distance computations for 95 % R@1 | vs {ref} | for 95 % R@10 | vs {ref} | query time at 95 % R@1 (µs) | vs {ref} | R@1 at nprobe 16 |"); print("|---|---|---|---|---|---|---|---|---|")
for v, d in sorted(g.items()):
    def mean(k): x = [r[k] for r in d.values() if r[k] is not None]; return np.mean(x) if x else float("nan")
    def rel(k):
        c = [100 * (d[s][k] / base[s][k] - 1) for s in d if s in base and d[s][k] and base[s][k]]
        return (f"{np.mean(c):+.1f}% ± {np.std(c):.1f}" if len(c) > 1 else f"{c[0]:+.1f}%") if c else ""
    print(f"| {v} | {len(d)} | {mean('ndis_r1>=0.95'):,.0f} | {rel('ndis_r1>=0.95')} | {mean('ndis_r10>=0.95'):,.0f} | {rel('ndis_r10>=0.95')} | {mean('us_r1>=0.95'):.0f} | {rel('us_r1>=0.95')} | {np.mean([r['r1'][i16] for r in d.values()]):.4f} |")
