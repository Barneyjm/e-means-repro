"""Usage: python als_pick.py lam <lam log> <K>  |  python als_pick.py grid <log> <K> <passes>
Prints the ridge strength with the lowest held-out RMSE after 25 ALS sweeps, or the best "<B>:<a0>" of a grid."""
import sys, json
rows = [json.loads(l) for l in open(sys.argv[2])]; K = int(sys.argv[3])
if sys.argv[1] == "lam": print(min((r for r in rows if r["K"] == K and r["mode"] == "lam"), key=lambda r: r["test"])["lam"])
else:
    r = min((r for r in rows if r["K"] == K and r["mode"] == "grid" and r["T"] == int(sys.argv[4])), key=lambda r: r["test"]); print(f"{r['B']}:{r['a0']}")
