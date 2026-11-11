"""Usage: python summarize_check_targets.py <dataset dir with train_k<k>.jsonl + index_k<k>.jsonl> <k> <recall targets, e.g. 0.7,0.8,0.85>
Distance computations per query needed for each R@1 target (interpolated over the probe sweep), every variant against
Faiss k-means with 25 iterations, paired by seed. For datasets where 95% is out of reach within the sweep. Seeds that never
reach a target are left out of that cell; the count of seeds that did is in brackets."""
import sys, re, json, collections, numpy as np
from summarize_check import label
d, k, targets = sys.argv[1], int(sys.argv[2]), [float(t) for t in sys.argv[3].split(",")]
I = collections.defaultdict(dict)
for r in map(json.loads, open(f"{d}/index_k{k}.jsonl")):
    m = re.match(r".*_k\d+_(.*)_s(\d+)\.npy", r["centroids"]); I[m.group(1)][int(m.group(2))] = r
need = lambda r, t: float(np.interp(t, r["r1"], r["ndis"])) if r["r1"][-1] >= t else np.nan
print("| Centroids from | " + " | ".join(f"R@1 {t:.0%}: distance computations | Change" for t in targets) + " |"); print("|---" * (1 + 2 * len(targets)) + "|")
for v in ["faiss25"] + sorted((v for v in I if v != "faiss25"), key=lambda v: (v[:4], "+" in v, v)):
    cells = []
    for t in targets:
        a = np.array([need(I[v][s], t) for s in sorted(I[v])]); b = np.array([need(I["faiss25"][s], t) for s in sorted(I[v])]); c = 100 * (a / b - 1); ok = ~np.isnan(c)
        cells += [f"{np.nanmean(a):,.0f}" if ok.any() else "", "–" if v == "faiss25" else (f"{c[ok].mean():+.0f}% ± {c[ok].std():.0f} [{ok.sum()}]" if ok.any() else "not reached")]
    print(f"| {label(v)} | " + " | ".join(cells) + " |")
