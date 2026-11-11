"""Usage: python als_buckets.py <ratings.npy> <K> <lam> <B> <a0> <passes> [seed]
Where does online ALS differ from full ALS? Held-out RMSE by item popularity (training ratings per item), factor norms of
rare items, and how many items the online version never re-solved. Looks for a failure mode worth a repair step."""
import sys, numpy as np, torch
from als_torch import Data, als, online, rmse
path, K, lam, B, a0, T = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6]); seed = int(sys.argv[7]) if len(sys.argv) > 7 else 0
d = Data(path, seed); cnt = d.I.cnt; edges = [0, 1, 5, 20, 100, 500, 10**9]; tc = cnt[d.ti]
models = {"ALS 40 sweeps": als(d, K, 40, seed, lam)[:2], f"online {T} passes B={B} a0={a0}": online(d, K, T, seed, lam, B=B, a0=a0)[:2],
          "online, no clean-up": online(d, K, T, seed, lam, B=B, a0=a0, final="none")[:2]}
print(f"{path.split('/')[-1]} rank {K} ridge {lam} seed {seed}\n\n| training ratings of the item | held-out ratings | " + " | ".join(models) + " |\n|---|---|" + "---|" * len(models))
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (tc >= lo) & (tc < hi); n = int(m.sum())
    if n: print(f"| {lo}-{hi - 1 if hi < 10**9 else '...'} | {n:,} | " + " | ".join(f"{float(((P[d.tu[m]] * Q[d.ti[m]]).sum(1) - d.tr[m]).pow(2).mean().sqrt()):.4f}" for P, Q in models.values()) + " |")
print("| all | " + f"{len(tc):,} | " + " | ".join(f"{rmse(d, P, Q):.4f}" for P, Q in models.values()) + " |")
print("\n| | " + " | ".join(models) + " |\n|---|" + "---|" * len(models))
for name, m in (("items with 1-4 training ratings", (cnt >= 1) & (cnt < 5)), ("items with 5-99", (cnt >= 5) & (cnt < 100)), ("items with 100+", cnt >= 100)):
    print(f"| mean factor norm, {name} ({int(m.sum())}) | " + " | ".join(f"{float(Q[m].norm(dim=1).mean()):.3f}" for P, Q in models.values()) + " |")
print("| largest factor norm | " + " | ".join(f"{float(Q.norm(dim=1).max()):.2f}" for P, Q in models.values()) + " |")
print("| non-finite entries | " + " | ".join(str(int((~torch.isfinite(Q)).sum() + (~torch.isfinite(P)).sum())) for P, Q in models.values()) + " |")
