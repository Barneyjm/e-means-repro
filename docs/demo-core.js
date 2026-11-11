// Lloyd's k-means and e-means on 2-D points, written to step through one update at a time for the demo page.
// The e-means update follows emeans_sklearn.EMeans: decayed counts N and sums S (centre = S / N), cosine step from
// a0 to a0 / 10, revival every 5 updates, split offset 1e-4, last-pass mean at the end, tail of each pass unused.
(function (root) {
  "use strict";

  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  function gauss(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
  const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  function permutation(n, r) {
    const p = new Int32Array(n);
    for (let i = 0; i < n; i++) p[i] = i;
    for (let i = n - 1; i > 0; i--) { const j = Math.floor(r() * (i + 1)); const t = p[i]; p[i] = p[j]; p[j] = t; }
    return p;
  }

  // Points in the unit square. Coordinates stay positive, as the split offset at revival is relative to the centre.
  function makeData(kind, n, seed) {
    const r = rng(seed * 7919 + 17), X = new Float64Array(2 * n);
    const put = (i, x, y) => { X[2 * i] = clamp(x, 0.02, 0.98); X[2 * i + 1] = clamp(y, 0.02, 0.98); };
    if (kind === "uniform") {
      for (let i = 0; i < n; i++) put(i, 0.04 + 0.92 * r(), 0.04 + 0.92 * r());
      return X;
    }
    const m = kind === "clumps" ? 7 : 9, minGap = kind === "clumps" ? 0.2 : 0.17, cs = [];
    for (let tries = 0; cs.length < m && tries < 5000; tries++) {
      const c = [0.12 + 0.76 * r(), 0.12 + 0.76 * r()];
      if (cs.every(d => Math.hypot(d[0] - c[0], d[1] - c[1]) > minGap)) cs.push(c);
    }
    const w = cs.map(() => 0.3 + 6 * r() * r()), sd = cs.map(() => kind === "clumps" ? 0.004 : 0.02 + 0.05 * r());
    const wsum = w.reduce((a, b) => a + b, 0);
    for (let i = 0; i < n; i++) {
      if (kind === "clumps" && r() < 0.3) { put(i, 0.04 + 0.92 * r(), 0.04 + 0.92 * r()); continue; }
      let u = r() * wsum, j = 0;
      while (u > w[j] && j < w.length - 1) u -= w[j++];
      put(i, cs[j][0] + sd[j] * gauss(r), cs[j][1] + sd[j] * gauss(r));
    }
    return X;
  }

  // Random start: k distinct data points.
  function initCentres(X, n, k, seed) {
    const p = permutation(n, rng(seed * 104729 + 3)), C = new Float64Array(2 * k);
    for (let j = 0; j < k; j++) { C[2 * j] = X[2 * p[j]]; C[2 * j + 1] = X[2 * p[j] + 1]; }
    return C;
  }

  function nearest(x, y, C, k) {
    let best = 0, bd = Infinity;
    for (let j = 0; j < k; j++) {
      const dx = x - C[2 * j], dy = y - C[2 * j + 1], d = dx * dx + dy * dy;
      if (d < bd) { bd = d; best = j; }
    }
    return best;
  }

  // Mean squared distance to the nearest centre over all points, and how many clusters hold under 10% of n / k points.
  function measure(X, n, C, k) {
    const cnt = new Float64Array(k);
    let s = 0;
    for (let i = 0; i < n; i++) {
      const x = X[2 * i], y = X[2 * i + 1], j = nearest(x, y, C, k), dx = x - C[2 * j], dy = y - C[2 * j + 1];
      s += dx * dx + dy * dy; cnt[j]++;
    }
    let starved = 0;
    for (let j = 0; j < k; j++) if (cnt[j] < 0.1 * n / k) starved++;
    return { mse: s / n, starved };
  }

  // The paper's Table 2 laws (fitted on high-dimensional data).
  function lawBatch(n, k, T) {
    const b = 0.035134 * Math.pow(k, 0.134749) * Math.pow(n, 0.876473) * Math.pow(T, 0.813344);
    return clamp(Math.pow(2, Math.round(Math.log2(b))), 4, n);
  }
  function lawStep(n, k, B, T) { return Math.min(1, 0.824 * Math.pow(k, -0.46) * Math.pow(B, 0.9) * Math.pow(n, -0.45) * Math.pow(T, -0.11)); }

  // Lloyd: assign every point with the current centres, then move each centre to the mean of its points. One update per pass.
  class Lloyd {
    constructor(X, n, k, C0, passes) {
      Object.assign(this, { X, n, k, passes, C: Float64Array.from(C0), lab: new Int32Array(n).fill(-1),
        sum: new Float64Array(2 * k), cnt: new Float64Array(k), pos: 0, pass: 0, updates: 0, events: [] });
    }
    get done() { return this.pass >= this.passes; }
    get clock() { return this.pass * this.n + this.pos; }   // vectors processed so far
    advance(upto) {
      const { X, n, k, C, lab, sum, cnt } = this;
      while (!this.done) {
        const target = Math.min(Math.floor(upto) - this.pass * n, n);
        for (; this.pos < target; this.pos++) {
          const i = this.pos, x = X[2 * i], y = X[2 * i + 1], j = nearest(x, y, C, k);
          lab[i] = j; cnt[j]++; sum[2 * j] += x; sum[2 * j + 1] += y;
        }
        if (this.pos < n) break;
        const before = Float64Array.from(C);
        for (let j = 0; j < k; j++) if (cnt[j] > 0) { C[2 * j] = sum[2 * j] / cnt[j]; C[2 * j + 1] = sum[2 * j + 1] / cnt[j]; }
        sum.fill(0); cnt.fill(0); this.pos = 0; this.pass++; this.updates++;
        this.events.push({ type: "update", before, at: this.clock });
      }
    }
  }

  class EMeans {
    constructor(X, n, k, C0, opt, seed) {
      const B = Math.min(opt.B, n), nb = Math.floor(n / B);
      Object.assign(this, { X, n, k, B, nb, T: opt.T, a0: opt.a0, revive: opt.revive, cosine: opt.cosine, M: opt.T * nb,
        C: Float64Array.from(C0), N: new Float64Array(k).fill(1e-10), S: new Float64Array(2 * k),
        lab: new Int32Array(n).fill(-1), last: new Int32Array(n).fill(-1), r: rng(seed * 15485863 + 11),
        t: 0, epoch: 0, b: 0, perm: null, batch: null, alpha: opt.a0, revivals: 0, done: false, events: [] });
      for (let j = 0; j < k; j++) { this.S[2 * j] = 1e-10 * C0[2 * j]; this.S[2 * j + 1] = 1e-10 * C0[2 * j + 1]; }
    }
    get clock() { return this.done ? this.T * this.n : this.epoch * this.n + this.b * this.B; }
    stepAlpha(t) { return this.cosine ? this.a0 * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * t / this.M))) : this.a0; }
    advance(upto) {
      while (!this.done && this.epoch * this.n + (this.b + 1) * this.B <= upto) this.step();
    }
    step() {
      const { X, k, B, C, N, S, lab, last } = this;
      if (this.b === 0) this.perm = permutation(this.n, this.r);
      const ids = this.perm.subarray(this.b * B, (this.b + 1) * B);
      const alpha = this.stepAlpha(this.t);
      this.alpha = alpha; this.t++;
      const nj = new Float64Array(k), sj = new Float64Array(2 * k);
      for (const i of ids) {
        const x = X[2 * i], y = X[2 * i + 1], a = nearest(x, y, C, k);
        lab[i] = a; if (this.epoch === this.T - 1) last[i] = a;
        nj[a]++; sj[2 * a] += x; sj[2 * a + 1] += y;
      }
      if (alpha >= 1) {
        for (let j = 0; j < k; j++) if (nj[j] > 0) { N[j] = nj[j]; S[2 * j] = sj[2 * j]; S[2 * j + 1] = sj[2 * j + 1]; }
      } else {
        for (let j = 0; j < k; j++) {
          N[j] = (1 - alpha) * N[j] + alpha * nj[j];
          S[2 * j] = (1 - alpha) * S[2 * j] + alpha * sj[2 * j];
          S[2 * j + 1] = (1 - alpha) * S[2 * j + 1] + alpha * sj[2 * j + 1];
        }
      }
      if (this.revive && this.t % 5 === 0) this.doRevive();
      for (let j = 0; j < k; j++) { C[2 * j] = S[2 * j] / N[j]; C[2 * j + 1] = S[2 * j + 1] / N[j]; }
      this.batch = ids;
      this.b++;
      if (this.b === this.nb) { this.b = 0; this.epoch++; if (this.epoch === this.T) this.finish(); }
    }
    // Each cluster whose count is under 10% of the mean takes half of a donor drawn in proportion to count.
    doRevive() {
      const { N, S, k, r } = this, mean = N.reduce((a, b) => a + b, 0) / k, delta = 1e-4;
      const dead = [], alive = [];
      for (let j = 0; j < k; j++) (N[j] < 0.1 * mean ? dead : alive).push(j);
      if (!dead.length) return;
      const pool = alive.slice(), w = pool.map(j => N[j]);
      for (const j of dead.slice(0, alive.length)) {
        let u = r() * w.reduce((a, b) => a + b, 0), q = 0;
        while (u > w[q] && q < w.length - 1) u -= w[q++];
        const d = pool[q]; pool.splice(q, 1); w.splice(q, 1);
        const cx = S[2 * d] / N[d], cy = S[2 * d + 1] / N[d], half = N[d] / 2;
        N[j] = half; N[d] = half;
        S[2 * j] = cx * (1 + delta) * half; S[2 * j + 1] = cy * (1 - delta) * half;
        S[2 * d] = cx * (1 - delta) * half; S[2 * d + 1] = cy * (1 + delta) * half;
        this.revivals++;
        this.events.push({ type: "revive", from: [this.C[2 * j], this.C[2 * j + 1]], to: [cx, cy], at: this.epoch * this.n + (this.b + 1) * this.B });
      }
    }
    // Points left over after the last full batch are assigned too; then each centre becomes the exact mean of the
    // points it won during the last pass.
    finish() {
      const { X, n, k, C, last } = this;
      for (let p = this.nb * this.B; p < n; p++) { const i = this.perm[p]; last[i] = nearest(X[2 * i], X[2 * i + 1], C, k); }
      const cnt = new Float64Array(k), sum = new Float64Array(2 * k);
      for (let i = 0; i < n; i++) { const j = last[i]; cnt[j]++; sum[2 * j] += X[2 * i]; sum[2 * j + 1] += X[2 * i + 1]; }
      const before = Float64Array.from(C);
      for (let j = 0; j < k; j++) if (cnt[j] > 0) { C[2 * j] = sum[2 * j] / cnt[j]; C[2 * j + 1] = sum[2 * j + 1] / cnt[j]; }
      this.batch = null; this.done = true;
      this.events.push({ type: "cleanup", before, at: this.T * n });
    }
  }

  const api = { rng, makeData, initCentres, measure, lawBatch, lawStep, Lloyd, EMeans, nearest };
  if (typeof module !== "undefined" && module.exports) module.exports = api; else root.EMDemo = api;
})(typeof window !== "undefined" ? window : globalThis);
