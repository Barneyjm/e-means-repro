"""Usage: python summarize_lambda.py <lab log> [more logs]. The held size penalty at each strength, for every dataset and k:
change against plain 3-pass e-means (paired by seed) in scan cost at 95 % recall, test MSE and list imbalance."""
import sys, json, collections, numpy as np
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
names = ["cc0.03", "cc0.1", "cc0.2", "cc0.3", "cc0.5", "cc1", "cc2"]
print("| data | k | seeds | gap / distortion | metric | " + " | ".join("λ=" + v[2:] for v in names) + " | Lloyd 25 |"); print("|---|---|---|---|---|" + "---|" * (len(names) + 1))
for key in sorted({(r["data"], r["k"]) for r in rows if r["n"] >= 50_000}):
    by = collections.defaultdict(dict)
    for r in rows:
        if (r["data"], r["k"]) == key and r["n"] >= 50_000: by[r["v"]][r["seed"]] = r
    b = by["e3"]; ratio = np.mean([x["q_gap"] / x["q_d1"] for x in b.values() if "q_gap" in x]) if any("q_gap" in x for x in b.values()) else float("nan")
    def cell(v, met):
        d = by.get(v, {}); s = [s for s in d if s in b]
        if not s: return ""
        if met == "imbalance": return f"{np.mean([d[x]['imbalance'] for x in s]):.3f}"
        if any(d[x][met] is None or b[x][met] is None for x in s): return "n/a"
        v_ = [100 * (d[x][met] / b[x][met] - 1) for x in s]; return f"{np.mean(v_):+.1f} ± {np.std(v_):.1f}" if len(s) > 1 else f"{v_[0]:+.1f}"
    for met, label in (("cost_r1>=0.95", "scan, R@1 95 %"), ("cost_r10>=0.95", "scan, R@10 95 %"), ("test", "test MSE"), ("imbalance", "imbalance")):
        print(f"| {key[0].split('_')[0]} | {key[1]:,} | {len(b)} | {ratio:.3f} | {label} | " + " | ".join(cell(v, met) for v in names) + f" | {cell('lloyd25', met)} |")
