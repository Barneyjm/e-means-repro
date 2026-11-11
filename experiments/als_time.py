"""Usage: python als_time.py <ratings.npy> <log file> <K> <lam> [repeats=3]
Back-to-back timing of full ALS sweeps and online passes in one process (other jobs on the machine shift absolute times,
so only times taken side by side are compared)."""
import sys, json, torch
from als_torch import Data, als, online, rmse
path, outp, K, lam = sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]); reps = int(sys.argv[5]) if len(sys.argv) > 5 else 3
d = Data(path, 0); f = open(outp, "a"); als(d, K, 1, 99, lam); online(d, K, 1, 99, lam, B=d.nu // 4)
for rep in range(reps):
    P, Q, dt = als(d, K, 10, 0, lam); out = dict(K=K, rep=rep, als_s_per_sweep=round(dt / 10, 3), als10_rmse=round(rmse(d, P, Q), 5))
    for T, B in ((1, 4096), (1, 8192), (1, 16384), (3, 4096)):
        P, Q, dt, _ = online(d, K, T, 0, lam, B=B, a0=1.0); out[f"online_T{T}_B{B}_s"] = round(dt, 2); out[f"online_T{T}_B{B}_rmse"] = round(rmse(d, P, Q), 5)
    s = json.dumps(out); print(s, flush=True); f.write(s + "\n"); f.flush()
