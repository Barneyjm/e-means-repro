"""Usage: python lda_check.py <nyt8k.npz> [docs, default 3000] [K, default 20]

Checks the E-step and both M-steps against scikit-learn's LatentDirichletAllocation, drawing the topic start and each
pass's per-document gamma start from the same random stream as scikit-learn does (one job). Remaining differences:
float32 here against float64 there. Also reports how much the usual start (gamma = 1) changes the result."""
import sys, numpy as np, scipy.sparse as sp, torch
from sklearn.decomposition import LatentDirichletAllocation as SK
from lda_torch import load, em, perplexity, dexp, estep, dev
path = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 3000; K = int(sys.argv[3]) if len(sys.argv) > 3 else 20
Xu, Ho, Ht = load(path, n); Ho, Ht = Ho[:2000], Ht[:2000]; V = Xu.shape[1]; a = e = 1.0 / K; Xs = sp.csr_matrix(Xu.cpu().numpy().astype(np.float64))
def draw(rs, shape): return torch.from_numpy(rs.gamma(100.0, 0.01, shape).astype(np.float32)).to(dev)
rs = np.random.RandomState(0); lam = draw(rs, (K, V)); same = {}
for it in range(1, 11):
    _, ss, _ = estep(Xu, dexp(lam), a, strict=True, gamma0=draw(rs, (n, K))); lam = e + ss; same[it] = lam.clone()
ones = {}; em(Xu, K, 10, 0, a, e, lambda it, spent, lam: ones.__setitem__(it, lam.clone()), strict=True)
print("| batch passes | perplexity, scikit-learn | this code, same gamma start | gap | median rel. diff. of lambda | this code, gamma start 1 | gap |\n|---|---|---|---|---|---|---|")
for it in (1, 2, 5, 10):
    sk = SK(n_components=K, learning_method="batch", max_iter=it, random_state=0, evaluate_every=-1).fit(Xs); ls = torch.from_numpy(sk.components_.astype(np.float32)).to(dev)
    ps, pm, po = (perplexity(l, Ho, Ht, a) for l in (ls, same[it], ones[it]))
    print(f"| {it} | {ps:.2f} | {pm:.2f} | {pm / ps - 1:+.4%} | {float(((same[it] - ls).abs() / ls).median()):.1e} | {po:.2f} | {po / ps - 1:+.3%} |", flush=True)
# one online step: scikit-learn partial_fit on the first 500 documents against the same formula here
sk = SK(n_components=K, learning_method="online", total_samples=n, batch_size=500, random_state=0, learning_offset=10.0, learning_decay=0.7).partial_fit(Xs[:500])
rs = np.random.RandomState(0); lam0 = draw(rs, (K, V)); _, ss, _ = estep(Xu[:500], dexp(lam0), a, strict=True, gamma0=draw(rs, (500, K))); rho = (10.0 + 1) ** -0.7
lam1 = (1 - rho) * lam0 + rho * (e + n / 500 * ss); ls = torch.from_numpy(sk.components_.astype(np.float32)).to(dev)
print(f"\none online update (rho = {rho:.4f}): perplexity {perplexity(ls, Ho, Ht, a):.2f} scikit-learn, {perplexity(lam1, Ho, Ht, a):.2f} here; median relative difference of lambda {float(((lam1 - ls).abs() / ls).median()):.1e}")
