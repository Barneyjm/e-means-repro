"""Usage: python run.py <data dir> <log file> <k> <seeds, comma separated>. Times are training only."""
import sys, json, time, numpy as np, torch
from common import fvecs, dev
from emeans_torch import lloyd, emeans, score
D = sys.argv[1]; out = sys.argv[2]; k = int(sys.argv[3]); seeds = [int(s) for s in sys.argv[4].split(",")]
X = fvecs(D + "/sift_base.fvecs"); Xte = fvecs(D + "/sift_query.fvecs"); Xd = torch.from_numpy(X).to(dev)
f = open(out, "a")
def log(s): print(s, flush=True); f.write(s + "\n"); f.flush()
emeans(Xd, X, k, 1, 99); emeans(Xd, X, k, 1, 99, half=True)   # warm up GPU kernels
for seed in seeds:
    C, dt = lloyd(Xd, X, k, 25, seed); log(json.dumps(dict(m="lloyd25", seed=seed, time=round(dt, 2), **score(Xd, X, Xte, C))))
    for T in (1, 2, 3, 5):
        for half in (False, True):
            C, dt, info = emeans(Xd, X, k, T, seed, half=half)
            log(json.dumps(dict(m="emeans" + ("_fp16" if half else ""), T=T, seed=seed, time=round(dt, 2), **info, **score(Xd, X, Xte, C))))
