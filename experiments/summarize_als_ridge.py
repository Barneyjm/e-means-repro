"""Usage: python summarize_als_ridge.py <K> <B:a0> <log> [more logs]. Full ALS (40 sweeps) and online at fixed settings, by ridge
strength: is the online version's held-out edge still there when ALS gets its own best ridge?"""
import sys, json, numpy as np
K, mode = int(sys.argv[1]), "fixed" + sys.argv[2]; rows = [json.loads(l) for f in sys.argv[3:] for l in open(f)]; rows = [r for r in rows if r["K"] == K]
print("| ridge | seeds | ALS 10 sweeps | ALS 25 | ALS 40 | online 1 pass | 2 | 3 | 5 | 10 | training objective: ALS 40 | online 3 |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
for lam in sorted({r["lam"] for r in rows}):
    on = lambda T: {r["seed"]: r for r in rows if r["lam"] == lam and r["m"] == "online" and r["mode"] == mode and r["T"] == T}
    a = {r["seed"]: r for r in rows if r["lam"] == lam and r["m"] == "als" and r["mode"] == "main" and r["seed"] in on(3)}
    if a: print(f"| {lam} | {sorted(a)} | " + " | ".join(f"{np.mean([r['curve'][s - 1] for r in a.values()]):.4f}" for s in (10, 25, 40)) + " | " +
                " | ".join(f"{np.mean([r['test'] for r in on(T).values()]):.4f}" if on(T) else "" for T in (1, 2, 3, 5, 10)) + f" | {np.mean([r['obj'][-1] for r in a.values()]):.4f} | {np.mean([r['obj'] for r in on(3).values()]):.4f} |")
