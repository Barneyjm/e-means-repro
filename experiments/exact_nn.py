"""Usage: python exact_nn.py <base .npy|.fvecs> <queries .npy|.fvecs> <out.npy> [n queries]
Exact 10 nearest base vectors (squared L2) for each query, on the GPU. Input to ivf_sweep.py."""
import sys, numpy as np, torch
from common import dev
from ivf_sweep import load
X = torch.from_numpy(load(sys.argv[1])).to(dev); Q = load(sys.argv[2]); Q = Q[:int(sys.argv[4])] if len(sys.argv) > 4 else Q
xn = (X * X).sum(1); Xt = X.T.contiguous(); ch = max(256, int(2e9 / len(X)) // 256 * 256); out = []
for i in range(0, len(Q), ch):
    q = torch.from_numpy(Q[i:i+ch]).to(dev)
    out.append(torch.topk(torch.addmm(xn[None].expand(len(q), -1), q, Xt, alpha=-2), 10, dim=1, largest=False).indices.cpu())
nn = torch.cat(out).numpy().astype(np.int32); np.save(sys.argv[3], nn); print("saved", nn.shape, "base", tuple(X.shape))
