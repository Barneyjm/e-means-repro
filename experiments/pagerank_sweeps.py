"""Usage: python pagerank_sweeps.py <edges.npy> <log file> [max nodes, for a quick test]

PageRank (damping 0.85) as the linear system x = d M x + v, M = column-normalised adjacency (dangling columns zero),
v uniform; its solution, normalised, is the PageRank vector with uniform dangling redistribution.
Compares, by sweeps and seconds to reach a residual ||d M x + v - x||_1 / ||x||_1 of 1e-6 and 1e-10:
  jacobi          every node updated from the previous sweep's values (the linear-system form of power iteration)
  gauss-seidel    nodes updated one at a time, each using the freshest values (one sparse triangular solve per sweep)
  blocks          the nodes cut into nb blocks; a block is updated at once from the freshest values of the others
                  (nb = 1 is jacobi; nb = number of nodes is gauss-seidel), block order fixed or reshuffled every sweep
  relaxed jacobi  x <- (1 - w) x + w (d M x + v): w < 1 is the analogue of a fading running total, w = cosine 1 -> 0.1 of
                  the e-means step curve
Orderings of the nodes: as in the file, random relabelling, and reverse Cuthill-McKee.
One sweep = every node updated once = one pass over the edges. Timing covers the sweeps only, not the residual checks.
"""
import sys, json, time, math, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve_triangular
from scipy.sparse.csgraph import reverse_cuthill_mckee

D, TOL, MAXS = 0.85, (1e-6, 1e-10), 400
e = np.load(sys.argv[1]); outp = sys.argv[2]
ids, inv = np.unique(e, return_inverse=True); src, dst = inv.reshape(e.shape).T; n = len(ids)
if len(sys.argv) > 3: keep = (src < int(sys.argv[3])) & (dst < int(sys.argv[3])); src, dst, n = src[keep], dst[keep], int(sys.argv[3])
f = open(outp, "a")

def matrix(perm=None):
    s, t = (src, dst) if perm is None else (perm[src], perm[dst]); out = np.bincount(s, minlength=n)
    return sp.csr_matrix((1.0 / out[s], (t, s)), shape=(n, n))

def run(name, order, M, sweep):
    """sweep(x, it) -> x after one more sweep"""
    v = np.full(n, 1.0 / n); x = v.copy(); spent = 0.0; hit = {}; res = []
    for it in range(1, MAXS + 1):
        t = time.perf_counter(); x = sweep(x, it); spent += time.perf_counter() - t
        r = np.abs(D * (M @ x) + v - x).sum() / np.abs(x).sum(); res.append(r)
        for tol in TOL:
            if r <= tol and tol not in hit: hit[tol] = (it, round(spent, 3))
        if len(hit) == len(TOL) or not np.isfinite(r): break
    s = json.dumps(dict(nodes=n, edges=len(src), method=name, order=order, **{f"sweeps_{tol:g}": hit.get(tol, (None, None))[0] for tol in TOL},
                        **{f"sec_{tol:g}": hit.get(tol, (None, None))[1] for tol in TOL}, ms_per_sweep=round(1000 * spent / it, 2), last=float(f"{res[-1]:.3g}")))
    print(s, flush=True); f.write(s + "\n"); f.flush(); return x / x.sum()

rng = np.random.default_rng(0); v = np.full(n, 1.0 / n); ref = None
orders = {"file": None, "random": rng.permutation(n)}
orders["rcm"] = np.argsort(reverse_cuthill_mckee((matrix() + matrix().T).tocsr(), symmetric_mode=True))      # new label of each node
for oname, perm in orders.items():
    M = matrix(perm)
    x = run("jacobi", oname, M, lambda x, it: D * (M @ x) + v)
    if ref is None: ref = x
    else: print("  same answer as file-order jacobi: max abs diff", float(np.abs((x if perm is None else x[perm]) - ref).max()))
    Lo = (sp.identity(n, format="csr") - D * sp.tril(M, 0, format="csr")).tocsr(); Up = D * sp.triu(M, 1, format="csr")
    x = run("gauss-seidel", oname, M, lambda x, it: spsolve_triangular(Lo, Up @ x + v, lower=True))
    print("  same answer as file-order jacobi: max abs diff", float(np.abs((x if perm is None else x[perm]) - ref).max()))
    for nb in (4, 16, 64, 256, 1024, 4096):
        cuts = np.linspace(0, n, nb + 1).astype(int); blocks = [(a, b, M[a:b]) for a, b in zip(cuts[:-1], cuts[1:])]
        def sweep(x, it, shuffle=False):
            x = x.copy()
            for j in (rng.permutation(nb) if shuffle else range(nb)):
                a, b, Mb = blocks[j]; x[a:b] = D * (Mb @ x) + v[a:b]
            return x
        run(f"blocks nb={nb}", oname, M, sweep)
        if oname == "file": run(f"blocks nb={nb}, order reshuffled each sweep", oname, M, lambda x, it: sweep(x, it, True))
    if oname == "file":
        for w in (0.5, 0.8, 1.1, 1.2): run(f"relaxed jacobi w={w}", oname, M, lambda x, it: (1 - w) * x + w * (D * (M @ x) + v))
        cosw = lambda it, total: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(it - 1, total) / total))
        for total in (50, 150): run(f"relaxed jacobi w=cosine 1->0.1 over {total} sweeps", oname, M, lambda x, it: (lambda w: (1 - w) * x + w * (D * (M @ x) + v))(cosw(it, total)))
