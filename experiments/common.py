"""Shared helpers: data loading, device pick, exact scoring, and the paper's scaling laws."""
import math, numpy as np, torch

dev = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def sync():
    if dev.type == "mps": torch.mps.synchronize()
    elif dev.type == "cuda": torch.cuda.synchronize()

def fvecs(p):
    a = np.fromfile(p, dtype=np.int32); d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].view(np.float32).copy()

def mse(Xnp, Cnp, idx=None):
    """exact float64 MSE (sum of squared diffs per vector, averaged). If idx None, exact float64 assignment."""
    X = torch.from_numpy(Xnp).double(); C = torch.from_numpy(Cnp).double()
    if idx is None:
        cn = (C*C).sum(1); idx = torch.cat([torch.argmin(cn[None]-2*X[i:i+2048]@C.T,1) for i in range(0,len(X),2048)])
    else: idx = torch.from_numpy(idx)
    return float(((X - C[idx])**2).sum(1).mean())

def split_pattern(d, delta):
    s = np.where(np.arange(d) % 2 == 0, 1.0, -1.0)
    return 1 + delta * s, 1 - delta * s

def laws(n, k, T):
    """Batch size and peak step from the paper's Table 2 (Eq. 6 and 7), used as published."""
    B = 0.035 * k**0.13 * n**0.88 * T**0.81
    if n > 2_000_000: B /= 8
    B = int(min(2 ** round(math.log2(B)), n))
    a0 = min(0.824 * k**-0.46 * B**0.90 * n**-0.45 * T**-0.11, 1.0)
    return B, a0
