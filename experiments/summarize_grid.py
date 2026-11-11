"""Usage: python summarize_grid.py <grid log>. Held-out log-likelihood relative to the published formula's setting
(rows: batch size as a multiple of the formula's; columns: peak step), mean over seeds."""
import sys, json, collections, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
for key in sorted({(r["ds"], r["K"], r["n"]) for r in rows}):
    rs = [r for r in rows if (r["ds"], r["K"], r["n"]) == key]
    for T in sorted({r["T"] for r in rs if r["m"] == "grid"}):
        law = next(r for r in rs if r["m"] == "law" and r["T"] == T)
        g = collections.defaultdict(list)
        for r in rs:
            if r["m"] == "grid" and r["T"] == T: g[(r["bm"], r["a0"])].append(r["test"])
        bms = sorted({k[0] for k in g}); als = sorted({k[1] for k in g})
        ref_a = min(als, key=lambda a: abs(a - law["a0"])); ref = np.mean(g[(1, ref_a)])
        print(f"\n{key[0]} K={key[1]} n={key[2]:,}, {T} pass(es); formula: B={law['B']}, step={law['a0']} (reference cell: B x1, step {ref_a} = {ref:.4f})\n")
        print("| B multiple | " + " | ".join(f"step {a}" for a in als) + " |"); print("|---|" + "---|" * len(als))
        for bm in bms: print(f"| x{bm:g} | " + " | ".join(f"{np.mean(g[(bm, a)]) - ref:+.3f}" if (bm, a) in g else "" for a in als) + " |")
