"""Usage: python lab_check.py <sift dir>. emeans_lab with default options must reproduce emeans_torch.emeans."""
import sys, numpy as np, torch
from common import fvecs, dev
from emeans_torch import emeans
from emeans_lab import emeans_x
X = fvecs(sys.argv[1] + "/sift_learn.fvecs"); Xd = torch.from_numpy(X).to(dev)
for k, T in ((1024, 1), (1024, 3), (4096, 2)):
    a, ta, _ = emeans(Xd, X, k, T, 0); b, tb, info = emeans_x(Xd, X, k, T, 0)
    print(k, T, "max abs difference", float(np.abs(a - b).max()), "identical" if np.array_equal(a, b) else "DIFFERENT", "times", round(ta, 3), round(tb, 3), info)
for kw in (dict(donor="error"), dict(revive="point"), dict(revive="d2"), dict(util_tau=0.1), dict(consc=0.1), dict(soft_m=4), dict(bgrow=4), dict(nest=0.5), dict(init="pp"), dict(polish=1), dict(sched="exp", floor=0.01)):
    b, tb, info = emeans_x(Xd, X, 1024, 3, 0, **kw); print(kw, round(tb, 3), info, "finite" if np.isfinite(b).all() else "NAN")
from emeans_lab import ladder
b, tb, info = ladder(Xd, X, 1024, 2, 0); print("ladder", round(tb, 3), info, np.isfinite(b).all())
