"""Usage: python fit_lambda_rule.py <lab log> [more logs]
Where should the size penalty sit? Model: log(scan cost at fixed recall) = a*log(imbalance) + b*log(train MSE) + const,
fitted per dataset and k over the logged runs that use no size penalty (seed as a fixed effect), so the prediction
is made before any penalty is tried. On the trade-off curve traced by the penalty,
d(MSE)/d(imbalance) = -strength * MSE / 2, so cost is lowest where strength = 2a / (b * imbalance).
Prints the fitted a and b, the strength the rule predicts, and the best strength actually measured on the grid."""
import sys, json, collections, numpy as np
rows = [json.loads(l) for f in sys.argv[1:] for l in open(f)]
grid = {"e3": 0.0, "cc0.03": 0.03, "cc0.1": 0.1, "cc0.2": 0.2, "cc0.3": 0.3, "cc0.5": 0.5, "cc1": 1.0, "cc2": 2.0}
print("| data | k | runs | recall target | a (imbalance) | b (train MSE) | R² | predicted strength 2a/(b·I) | best on grid (mean scan change vs none) | at the predicted grid point |"); print("|---|---|---|---|---|---|---|---|---|---|")
for key in sorted({(r["data"], r["k"]) for r in rows if r["n"] >= 50_000}):
    for tgt, label in (("cost_r1>=0.95", "R@1 95 %"), ("cost_r10>=0.95", "R@10 95 %"), ("cost_r10>=0.9", "R@10 90 %")):
        allr = [r for r in rows if (r["data"], r["k"]) == key and r["n"] >= 50_000 and r.get(tgt) and np.isfinite(r["train"]) and r["imbalance"] < 1.6 and not r["v"].startswith("lloyd")]
        rs = [r for r in allr if not r["v"].startswith(("cc", "ca", "consc", "c1end", "c0.3end", "clog"))]      # fit only on runs without a size penalty
        seeds = sorted({r["seed"] for r in rs})
        if len(rs) < 6 + len(seeds): continue
        A = np.array([[np.log(r["imbalance"]), np.log(r["train"])] + [float(r["seed"] == s) for s in seeds] for r in rs]); y = np.log([r[tgt] for r in rs])
        coef = np.linalg.lstsq(A, y, rcond=None)[0]; pred = A @ coef; r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        by = collections.defaultdict(dict)
        for r in allr:
            if r["v"] in grid: by[r["v"]][r["seed"]] = r
        b0 = by["e3"]; chg = {v: np.mean([100 * (d[s][tgt] / b0[s][tgt] - 1) for s in d if s in b0]) for v, d in by.items() if v != "e3"}
        best = min(chg, key=chg.get) if chg else None; Ibal = np.mean([x["imbalance"] for x in by.get("cc0.2", by["e3"]).values()])
        lam = 2 * coef[0] / (coef[1] * Ibal) if coef[1] > 0 else float("nan"); near = min((v for v in chg), key=lambda v: abs(np.log(grid[v] / lam)) if lam > 0 else 0) if chg and lam > 0 else None
        print(f"| {key[0].split('_')[0]} | {key[1]:,} | {len(rs)} | {label} | {coef[0]:.2f} | {coef[1]:.1f} | {r2:.2f} | {lam:.2f} | {grid[best]} ({chg[best]:+.1f}%) | " + (f"{grid[near]} ({chg[near]:+.1f}%) |" if near else "|"))
