"""Usage: python prep_h5.py <benchmark .hdf5> <out dir> <name> <max base rows> <queries held out> <queries with exact neighbours>
ann-benchmarks / VIBE hdf5 -> <name>_base.npy, <name>_q.npy (unit-length float32; a fixed random split of the `train` array)
and <name>_gt.npy (exact 10 nearest base rows for the first queries, by Faiss on the CPU)."""
import sys, numpy as np, h5py, faiss
src, out, name, nmax, nq, ngt = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
with h5py.File(src, "r") as f:
    print({k: f[k].shape for k in f}, dict(f.attrs), flush=True); A = f["train"][:].astype(np.float32)
p = np.random.default_rng(0).permutation(len(A))[:nmax + nq]; X, Q = [a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12) for a in (A[np.sort(p[nq:])], A[p[:nq]])]
_, I = faiss.knn(Q[:ngt], X, 10); np.save(f"{out}/{name}_base.npy", X); np.save(f"{out}/{name}_q.npy", Q); np.save(f"{out}/{name}_gt.npy", I.astype(np.int32))
print(name, "base", X.shape, "queries", len(Q), "with neighbours", len(I), flush=True)
