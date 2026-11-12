// Full-pass EM and the online recipe for diagonal Gaussian mixtures on 2-D points, one update at a time, for the
// follow-ups page. Follows experiments/gmm_torch.py and online_gmm_sklearn.py: start from K random points with
// weight 1/K and the overall variance; online = decayed totals of weight N, sum S and sum of squares Q, cosine
// step from a0 to a0 / 10, revival every 5 updates (halves placed at mean +- 0.5 sd, alternating sign per
// feature), and at the end the plain totals of the last pass. Needs demo-core.js (EMDemo) for data and laws.
(function (root) {
  "use strict";
  const REG = 1e-6, LOG2PI = Math.log(2 * Math.PI);

  // log w_k N(x | mu_k, var_k) for all k; returns the log-sum and fills r with responsibilities
  function estep(x, y, P, r) {
    const K = P.w.length;
    let m = -Infinity;
    for (let k = 0; k < K; k++) {
      const dx = x - P.mu[2 * k], dy = y - P.mu[2 * k + 1];
      const l = Math.log(P.w[k]) - 0.5 * (Math.log(P.v[2 * k]) + Math.log(P.v[2 * k + 1]) + dx * dx / P.v[2 * k] + dy * dy / P.v[2 * k + 1] + 2 * LOG2PI);
      r[k] = l; if (l > m) m = l;
    }
    let s = 0;
    for (let k = 0; k < K; k++) { r[k] = Math.exp(r[k] - m); s += r[k]; }
    for (let k = 0; k < K; k++) r[k] /= s;
    return m + Math.log(s);
  }

  function params(N, S, Q) {
    const K = N.length, tot = N.reduce((a, b) => a + b, 0);
    const w = new Float64Array(K), mu = new Float64Array(2 * K), v = new Float64Array(2 * K);
    for (let k = 0; k < K; k++) {
      const Ns = Math.max(N[k], 1e-30);
      w[k] = N[k] / tot;
      for (let d = 0; d < 2; d++) {
        mu[2 * k + d] = S[2 * k + d] / Ns;
        v[2 * k + d] = Math.max(Q[2 * k + d] / Ns - mu[2 * k + d] * mu[2 * k + d], 0) + REG;
      }
    }
    return { w, mu, v };
  }

  function start(X, n, K, seed) {
    const C = root.EMDemo.initCentres(X, n, K, seed);
    let sx = 0, sy = 0, qx = 0, qy = 0;
    for (let i = 0; i < n; i++) { sx += X[2 * i]; sy += X[2 * i + 1]; qx += X[2 * i] ** 2; qy += X[2 * i + 1] ** 2; }
    const gv = [qx / n - (sx / n) ** 2, qy / n - (sy / n) ** 2];
    const w = new Float64Array(K).fill(1 / K), v = new Float64Array(2 * K);
    for (let k = 0; k < K; k++) { v[2 * k] = gv[0] + REG; v[2 * k + 1] = gv[1] + REG; }
    return { w, mu: Float64Array.from(C), v };
  }

  function loglik(X, n, P) {
    const r = new Float64Array(P.w.length);
    let s = 0;
    for (let i = 0; i < n; i++) s += estep(X[2 * i], X[2 * i + 1], P, r);
    return s / n;
  }

  // Ordinary EM: statistics over every point with the current model, then the model from those totals. One update per pass.
  class EM {
    constructor(X, n, K, P0, passes) {
      const order = Int32Array.from({ length: n }, (_, i) => i).sort((a, b) => X[2 * a] - X[2 * b]);
      Object.assign(this, { X, n, K, passes, order, P: { w: Float64Array.from(P0.w), mu: Float64Array.from(P0.mu), v: Float64Array.from(P0.v) },
        N: new Float64Array(K), S: new Float64Array(2 * K), Q: new Float64Array(2 * K), r: new Float64Array(K), pos: 0, pass: 0, updates: 0, events: [] });
    }
    get done() { return this.pass >= this.passes; }
    advance(upto) {
      const { X, n, K, N, S, Q, r } = this;
      while (!this.done) {
        const target = Math.min(Math.floor(upto) - this.pass * n, n);
        for (; this.pos < target; this.pos++) {
          const i = this.order[this.pos], x = X[2 * i], y = X[2 * i + 1];
          estep(x, y, this.P, r);
          for (let k = 0; k < K; k++) { N[k] += r[k]; S[2 * k] += r[k] * x; S[2 * k + 1] += r[k] * y; Q[2 * k] += r[k] * x * x; Q[2 * k + 1] += r[k] * y * y; }
        }
        if (this.pos < n) break;
        const before = this.P;
        this.P = params(N, S, Q);
        N.fill(0); S.fill(0); Q.fill(0); this.pos = 0; this.pass++; this.updates++;
        this.events.push({ type: "update", before });
      }
    }
  }

  class OnlineEM {
    constructor(X, n, K, P0, opt, seed) {
      const B = Math.min(opt.B, n), nb = Math.floor(n / B), D = root.EMDemo;
      Object.assign(this, { X, n, K, B, nb, T: opt.T, a0: opt.a0, revive: opt.revive, M: opt.T * nb,
        P: { w: Float64Array.from(P0.w), mu: Float64Array.from(P0.mu), v: Float64Array.from(P0.v) },
        N: new Float64Array(K).fill(1e-10), S: new Float64Array(2 * K), Q: new Float64Array(2 * K),
        fN: new Float64Array(K), fS: new Float64Array(2 * K), fQ: new Float64Array(2 * K), r: new Float64Array(K),
        rand: D.rng(seed * 15485863 + 11), t: 0, epoch: 0, b: 0, perm: null, batch: null, alpha: opt.a0, revivals: 0, done: false, events: [] });
      for (let k = 0; k < K; k++) for (let d = 0; d < 2; d++) {
        const m = P0.mu[2 * k + d];
        this.S[2 * k + d] = 1e-10 * m; this.Q[2 * k + d] = 1e-10 * (P0.v[2 * k + d] - REG + m * m);
      }
    }
    batchStats(ids) {
      const { X, K, r } = this, bN = new Float64Array(K), bS = new Float64Array(2 * K), bQ = new Float64Array(2 * K);
      for (const i of ids) {
        const x = X[2 * i], y = X[2 * i + 1];
        estep(x, y, this.P, r);
        for (let k = 0; k < K; k++) { bN[k] += r[k]; bS[2 * k] += r[k] * x; bS[2 * k + 1] += r[k] * y; bQ[2 * k] += r[k] * x * x; bQ[2 * k + 1] += r[k] * y * y; }
      }
      return [bN, bS, bQ];
    }
    step() {
      const { K, B, N, S, Q } = this;
      if (this.b === 0) this.perm = permutation(this.n, this.rand);
      const ids = this.perm.subarray(this.b * B, (this.b + 1) * B);
      const alpha = this.a0 * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * this.t / this.M)));
      this.alpha = alpha; this.t++;
      const [bN, bS, bQ] = this.batchStats(ids);
      if (this.epoch === this.T - 1) for (let k = 0; k < K; k++) { this.fN[k] += bN[k]; for (let d = 0; d < 2; d++) { this.fS[2 * k + d] += bS[2 * k + d]; this.fQ[2 * k + d] += bQ[2 * k + d]; } }
      if (alpha >= 1) {
        const tot = bN.reduce((a, b) => a + b, 0);
        for (let k = 0; k < K; k++) if (bN[k] > 1e-8 * tot) { N[k] = bN[k]; for (let d = 0; d < 2; d++) { S[2 * k + d] = bS[2 * k + d]; Q[2 * k + d] = bQ[2 * k + d]; } }
      } else {
        for (let k = 0; k < K; k++) {
          N[k] = (1 - alpha) * N[k] + alpha * bN[k];
          for (let d = 0; d < 2; d++) { S[2 * k + d] = (1 - alpha) * S[2 * k + d] + alpha * bS[2 * k + d]; Q[2 * k + d] = (1 - alpha) * Q[2 * k + d] + alpha * bQ[2 * k + d]; }
        }
      }
      if (this.revive && this.t % 5 === 0) this.doRevive();
      this.P = params(N, S, Q);
      this.batch = ids;
      this.b++;
      if (this.b === this.nb) { this.b = 0; this.epoch++; if (this.epoch === this.T) this.finish(); }
    }
    doRevive() {
      const { N, S, Q, K, rand } = this, mean = N.reduce((a, b) => a + b, 0) / K, sign = [1, -1];
      const dead = [], alive = [];
      for (let k = 0; k < K; k++) (N[k] < 0.1 * mean ? dead : alive).push(k);
      if (!dead.length) return;
      const pool = alive.slice(), wts = pool.map(k => N[k]);
      for (const j of dead.slice(0, alive.length)) {
        let u = rand() * wts.reduce((a, b) => a + b, 0), q = 0;
        while (u > wts[q] && q < wts.length - 1) u -= wts[q++];
        const d = pool[q]; pool.splice(q, 1); wts.splice(q, 1);
        const Ns = Math.max(N[d], 1e-30), half = N[d] / 2, from = [this.P.mu[2 * j], this.P.mu[2 * j + 1]], at = [];
        const m = [S[2 * d] / Ns, S[2 * d + 1] / Ns], v = [0, 1].map(e => Math.max(Q[2 * d + e] / Ns - m[e] * m[e], 0));
        for (const [idx, sg] of [[j, 1], [d, -1]]) {
          N[idx] = half;
          for (let e = 0; e < 2; e++) {
            const m2 = m[e] + sg * 0.5 * Math.sqrt(v[e]) * sign[e];
            S[2 * idx + e] = half * m2; Q[2 * idx + e] = half * (v[e] + m2 * m2);
          }
        }
        this.revivals++;
        this.events.push({ type: "revive", from, to: m });
      }
    }
    finish() {
      const rest = this.perm.subarray(this.nb * this.B), [bN, bS, bQ] = this.batchStats(rest), K = this.K;
      for (let k = 0; k < K; k++) { this.fN[k] += bN[k]; for (let d = 0; d < 2; d++) { this.fS[2 * k + d] += bS[2 * k + d]; this.fQ[2 * k + d] += bQ[2 * k + d]; } }
      const tot = this.fN.reduce((a, b) => a + b, 0), N = new Float64Array(K), S = new Float64Array(2 * K), Q = new Float64Array(2 * K);
      for (let k = 0; k < K; k++) if (this.fN[k] > 1e-8 * tot) { N[k] = this.fN[k]; for (let d = 0; d < 2; d++) { S[2 * k + d] = this.fS[2 * k + d]; Q[2 * k + d] = this.fQ[2 * k + d]; } }
      const before = this.P;
      this.P = params(N, S, Q);
      this.batch = null; this.done = true;
      this.events.push({ type: "cleanup", before });
    }
  }

  function permutation(n, r) {
    const p = new Int32Array(n);
    for (let i = 0; i < n; i++) p[i] = i;
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = p[i]; p[i] = p[j]; p[j] = t; }
    return p;
  }

  const api = { estep, params, start, loglik, EM, OnlineEM };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.GMMDemo = api;
})(typeof window !== "undefined" ? window : globalThis);
