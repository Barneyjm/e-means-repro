"""Usage: python score_centroids.py <data dir> <centroids.npy>. Exact train/test MSE for saved centroids."""
import sys, numpy as np, torch
from common import fvecs, dev
from emeans_torch import score
X = fvecs(sys.argv[1] + "/sift_base.fvecs"); Xte = fvecs(sys.argv[1] + "/sift_query.fvecs")
print(score(torch.from_numpy(X).to(dev), X, Xte, np.load(sys.argv[2])))
