"""Usage: python hmm_validate.py <text8.npy>
Checks hmm_torch's Baum-Welch against hmmlearn.CategoricalHMM from the same start (log-likelihood before each of
10 updates), in float64 on the CPU and in float32 on the GPU, and the float32 held-out score against float64."""
import sys, numpy as np, torch, warnings
from hmmlearn.hmm import CategoricalHMM
from common import dev
from hmm_torch import init, stats, to_params, bits
warnings.filterwarnings("ignore")
x = np.load(sys.argv[1], mmap_mode="r"); K, V, n, L, iters = 8, 27, 300, 64, 10
X = np.ascontiguousarray(x[:n * L]).reshape(n, L).astype(np.int64)
pi, A, Bm, _ = init(K, V, 0, "cpu", torch.float64)
h = CategoricalHMM(n_components=K, n_iter=iters, tol=0, init_params="", params="ste", n_features=V, implementation="scaling")
h.startprob_, h.transmat_, h.emissionprob_ = pi.numpy().copy(), A.numpy().copy(), Bm.numpy().copy()
h.fit(X.reshape(-1, 1), [L] * n); ref = np.array(h.monitor_.history)
for device, dtype in (("cpu", torch.float64), (dev, torch.float32)):
    p = [v.to(device=device, dtype=dtype) for v in (pi, A, Bm)]; Xd = torch.from_numpy(X).to(device); mine = []
    for _ in range(iters):
        I, Tr, E, ll = stats(Xd, *p, 128); mine.append(ll); p = to_params(I, Tr, E)
    mine = np.array(mine); print(f"{device} {dtype}: max relative difference in log-likelihood over {iters} iterations {np.abs(mine / ref - 1).max():.2e}")
    print("   mine", mine[[0, 1, iters - 1]], "hmmlearn", ref[[0, 1, iters - 1]])
    for name, a, b in (("transitions", p[1], h.transmat_), ("emissions", p[2], h.emissionprob_)): print(f"   max abs difference in final {name}: {np.abs(a.cpu().numpy() - b).max():.2e}")
xe = torch.from_numpy(np.ascontiguousarray(x[-200_000:]).reshape(-1, 1000).astype(np.int64))
p64 = [torch.from_numpy(v) for v in (h.startprob_, h.transmat_, h.emissionprob_)]
b64 = bits(xe, 100, *p64); b32 = bits(xe.to(dev), 100, *[v.float().to(dev) for v in p64])
ref = -h.score(xe[:, :].numpy().reshape(-1, 1), [1000] * len(xe)) + h.score(xe[:, :100].numpy().reshape(-1, 1), [100] * len(xe))
print(f"held-out bits/char: float64 {b64:.6f}, float32 GPU {b32:.6f}, hmmlearn {ref / (900 * len(xe)) / np.log(2):.6f}")
