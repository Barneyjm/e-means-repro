"""Usage: python run_v1.py <data dir> <log file> <k> <seeds, comma separated>. Times include scoring."""
import sys, json, time, numpy as np, torch
from common import fvecs, dev, laws, sync
from emeans_v1 import lloyd, emeans
D = sys.argv[1]; out = sys.argv[2]; k = int(sys.argv[3]); seeds = [int(s) for s in sys.argv[4].split(",")]
X = fvecs(D + "/sift_base.fvecs"); Xte = fvecs(D + "/sift_query.fvecs")
Xd = torch.from_numpy(X).to(dev)
f = open(out, "a")
def log(s): print(s, flush=True); f.write(s + "\n"); f.flush()
log(f"n={len(X)} d={X.shape[1]} k={k} test={len(Xte)} laws T=1..5: {[laws(len(X),k,T) for T in (1,2,3,5)]}")
for seed in seeds:
    t0 = time.time(); sync()
    r = lloyd(Xd, X, Xte, k, 25, seed, eval_at=(10, 25), log=log)
    log(f"LLOYD seed{seed} {json.dumps(r)} time={time.time()-t0:.1f}s")
    for T in (1, 2, 3, 5):
        t0 = time.time(); r = emeans(Xd, X, Xte, k, T, seed); sync()
        log(f"EMEANS seed{seed} T={T} {json.dumps(r)} time={time.time()-t0:.1f}s")
