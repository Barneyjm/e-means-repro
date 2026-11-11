"""Usage: python exact_neighbours.py <data dir> <out.npy> [n queries]. Exact 10 nearest base vectors for the SIFT 'learn'
vectors, to use them as a 10x larger query set. Checks itself against the official ground truth of the 10k queries."""
import sys, numpy as np, torch
from common import fvecs, dev
from ivf_recall import ivecs
D = sys.argv[1]; nq = int(sys.argv[3]) if len(sys.argv) > 3 else 100_000
X = torch.from_numpy(fvecs(D + "/sift_base.fvecs")).to(dev); xn = (X * X).sum(1)
def knn(Q, k=10, ch=2000):
    Q = torch.from_numpy(Q).to(dev)
    return torch.cat([torch.topk(torch.addmm(xn[None].expand(min(ch, len(Q) - i), -1), Q[i:i+ch], X.T, alpha=-2), k, dim=1, largest=False).indices.cpu()
                      for i in range(0, len(Q), ch)]).numpy()
Xc = X.cpu().numpy().astype(np.float64)
def dist(Q, idx): return ((Q.astype(np.float64)[:, None, :] - Xc[idx]) ** 2).sum(2)
Qo = fvecs(D + "/sift_query.fvecs"); mine = knn(Qo); gt = ivecs(D + "/sift_groundtruth.ivecs")[:, :10]
print("official queries: same nearest id %.4f, same distances for all 10 %.4f" % ((mine[:, 0] == gt[:, 0]).mean(), np.isclose(dist(Qo, mine), dist(Qo, gt)).all(1).mean()))
L = fvecs(D + "/sift_learn.fvecs")[:nq]; nn = knn(L); np.save(sys.argv[2], nn.astype(np.int32)); print("saved", nn.shape)
