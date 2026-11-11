"""Usage: python summarize.py results/sklearn_cpu.log  -> mean time and MSE per (config, method) as a markdown table."""
import sys, json, collections
rows = collections.defaultdict(list)
for line in open(sys.argv[1]):
    r = json.loads(line); rows[(r["cfg"], r["method"])].append(r)
print("| config | method | runs | time (s) | train MSE | test MSE |\n|---|---|---|---|---|---|")
for (cfg, m), rs in rows.items():
    mean = lambda key: sum(r[key] for r in rs) / len(rs)
    test = f"{mean('test_mse'):.2f}" if "test_mse" in rs[0] else ""
    print(f"| {cfg} | {m} | {len(rs)} | {mean('time'):.2f} | {mean('train_mse'):.2f} | {test} |")
