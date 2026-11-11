"""Usage: python lda_data.py <nyt_full.npz> <out.npz> [vocab size, default 8000] [held-out docs, default 10000]

Cuts the UCI NYTimes bag of words to a fixed vocabulary (drop the 100 most document-frequent words, keep the next V),
drops documents left with fewer than 20 tokens, shuffles, and holds out documents. Each held-out document's tokens are
split at random into an observed half and a scored half (document completion). Saves CSR pieces in one .npz."""
import sys, numpy as np, scipy.sparse as sp
src, out = sys.argv[1], sys.argv[2]; V = int(sys.argv[3]) if len(sys.argv) > 3 else 8000; nh = int(sys.argv[4]) if len(sys.argv) > 4 else 10000
M = sp.load_npz(src).tocsc(); df = np.diff(M.indptr); order = np.argsort(-df, kind="stable"); keep = np.sort(order[100:100 + V])
M = M[:, keep].tocsr(); M = M[np.asarray(M.sum(1)).ravel() >= 20]; rng = np.random.default_rng(0); M = M[rng.permutation(M.shape[0])]
H, X = M[:nh].tocoo(), M[nh:].tocsr(); c = H.data.astype(np.int64); o = rng.binomial(c, 0.5)
Ho = sp.csr_matrix((o.astype(np.float32), (H.row, H.col)), shape=H.shape); Ht = sp.csr_matrix(((c - o).astype(np.float32), (H.row, H.col)), shape=H.shape)
Ho.eliminate_zeros(); Ht.eliminate_zeros()
def pack(n, A): return {n + "_data": A.data, n + "_indices": A.indices, n + "_indptr": A.indptr, n + "_shape": np.array(A.shape)}
np.savez(out, vocab_ids=keep, **pack("X", X), **pack("Ho", Ho), **pack("Ht", Ht))
u = np.diff(X.indptr); print(f"train {X.shape}, tokens {X.sum():.0f}, nnz {X.nnz}, unique words/doc mean {u.mean():.0f} p99 {np.percentile(u, 99):.0f} max {u.max()}, tokens/doc mean {X.sum() / X.shape[0]:.0f}; held-out {nh}: observed {Ho.sum():.0f}, scored {Ht.sum():.0f}")
