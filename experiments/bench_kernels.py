"""Usage: python bench_kernels.py <data dir>. Times one full pass of nearest-centroid assignment per kernel variant."""
import sys, time, numpy as np, torch
from common import fvecs, dev
X = fvecs(sys.argv[1] + "/sift_base.fvecs"); n, d = X.shape; k = 16384
rng = np.random.default_rng(0); Cn = X[rng.choice(n, k, replace=False)]
Xd = torch.from_numpy(X).to(dev); C = torch.from_numpy(Cn).to(dev)
def run(name, fn):
    fn(Xd[:20000]).cpu(); t = time.time(); a = fn(Xd).cpu(); dt = time.time() - t   # full epoch, forced to CPU
    print(f"{name:36s} {dt:6.2f} s/epoch", flush=True); return a
def base(Xs, chunk=8192):
    cn = (C * C).sum(1)
    return torch.cat([torch.argmin(cn[None, :] - 2 * Xs[i:i+chunk] @ C.T, dim=1) for i in range(0, len(Xs), chunk)])
def mk_addmm(chunk, dtype=torch.float32, scale=1.0):
    Cc = (C * scale); cn = (Cc ** 2).sum(1).to(dtype); Ct = Cc.T.contiguous().to(dtype)
    def f(Xs):
        out = []
        for i in range(0, Xs.shape[0], chunk):
            xc = (Xs[i:i+chunk] * scale).to(dtype) if (scale != 1 or dtype != torch.float32) else Xs[i:i+chunk]
            out.append(torch.argmin(torch.addmm(cn[None, :].expand(xc.shape[0], -1), xc, Ct, alpha=-2), dim=1))
        return torch.cat(out)
    return f
def mk_aug(chunk):
    Ca = torch.cat([C, (-(C * C).sum(1) / 2)[:, None]], 1).T.contiguous()   # (d+1, k)
    def f(Xs):
        out = []
        for i in range(0, Xs.shape[0], chunk):
            xc = Xs[i:i+chunk]; xa = torch.cat([xc, torch.ones(len(xc), 1, device=dev)], 1)
            out.append(torch.argmax(xa @ Ca, dim=1))
        return torch.cat(out)
    return f
ref = run("baseline (broadcast, chunk 8192)", base)
for ch in (4096, 8192, 16384, 32768, 65536):
    a = run(f"addmm fp32 chunk {ch}", mk_addmm(ch)); print("    agree", round(float((a == ref).float().mean()), 5))
for ch in (8192, 16384):
    a = run(f"augmented matmul chunk {ch}", mk_aug(ch)); print("    agree", round(float((a == ref).float().mean()), 5))
for ch in (8192, 16384):
    a = run(f"addmm fp16 chunk {ch}", mk_addmm(ch, torch.float16, 1/256)); print("    agree", round(float((a == ref).float().mean()), 5))
