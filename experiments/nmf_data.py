"""Usage: python nmf_data.py <nyt8k.npz> <out.npz> [n train docs, default 100000] -- TF-IDF (scikit-learn defaults, rows
scaled to unit length) of the first n training documents and of the 10,000 held-out documents (both halves together)."""
import sys, numpy as np, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer
z = np.load(sys.argv[1]); n = int(sys.argv[3]) if len(sys.argv) > 3 else 100000
def get(nm): return sp.csr_matrix((z[nm + "_data"], z[nm + "_indices"], z[nm + "_indptr"]), shape=tuple(z[nm + "_shape"]))
X = get("X")[:n]; H = get("Ho") + get("Ht"); tf = TfidfTransformer().fit(X); A = tf.transform(X).astype(np.float32).tocsr(); T = tf.transform(H).astype(np.float32).tocsr()
np.savez(sys.argv[2], A_data=A.data, A_indices=A.indices, A_indptr=A.indptr, A_shape=np.array(A.shape), T_data=T.data, T_indices=T.indices, T_indptr=T.indptr, T_shape=np.array(T.shape))
print("train", A.shape, A.nnz, "held-out", T.shape, T.nnz)
