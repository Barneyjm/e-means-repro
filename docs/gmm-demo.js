// Drives the mixture demo on the follow-ups page: ordinary EM and the online recipe on one clock in passes.
(function () {
  "use strict";
  const D = window.EMDemo, G = window.GMMDemo, $ = id => document.getElementById(id);
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const ui = {};
  ["gKind", "gN", "gK", "gTe", "gTl", "gSp", "gLaws", "gB", "gA", "gRevive"].forEach(id => ui[id] = $(id));
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let seed = 1, X = null, dataKey = "", A = null, O = null, clock = 0, playing = !reduceMotion, last = 0, cfg = null, histA = [], histO = [], seenAt = null;

  const speed = () => +(0.1 * Math.pow(100, +ui.gSp.value)).toFixed(2);
  function settings() {
    const n = +ui.gN.value, K = +ui.gK.value, T = +ui.gTe.value;
    let B = Math.pow(2, +ui.gB.value), a0 = +ui.gA.value;
    if (ui.gLaws.checked) { B = D.lawBatch(n, K, T); a0 = D.lawStep(n, K, B, T); ui.gB.value = Math.log2(B); ui.gA.value = a0.toFixed(2); }
    ui.gB.disabled = ui.gA.disabled = ui.gLaws.checked;
    return { kind: ui.gKind.value, n, K, T, TL: +ui.gTl.value, B: Math.min(B, n), a0, revive: ui.gRevive.checked };
  }
  function showValues() {
    $("gNOut").textContent = (+ui.gN.value).toLocaleString("en"); $("gKOut").textContent = ui.gK.value;
    $("gTeOut").textContent = ui.gTe.value; $("gTlOut").textContent = ui.gTl.value; $("gSpOut").textContent = speed();
    $("gBOut").textContent = cfg ? cfg.B.toLocaleString("en") : ""; $("gAOut").textContent = cfg ? cfg.a0.toFixed(2) : "";
  }
  function restart() {
    cfg = settings();
    const key = `${cfg.kind}|${cfg.n}|${seed}`;
    if (key !== dataKey) { X = D.makeData(cfg.kind, cfg.n, seed); dataKey = key; }
    const P0 = G.start(X, cfg.n, cfg.K, seed);
    A = new G.EM(X, cfg.n, cfg.K, P0, cfg.TL);
    O = new G.OnlineEM(X, cfg.n, cfg.K, P0, cfg, seed);
    clock = 0; seenAt = new Float64Array(cfg.n).fill(-1e9);
    const l0 = G.loglik(X, cfg.n, P0);
    histA = [[0, l0]]; histO = [[0, l0, null]];
    showValues();
  }
  function advanceTo(c) {
    const n = cfg.n, now = performance.now(), before = A.updates;
    A.advance(c);
    if (A.updates !== before) histA.push([A.pass, G.loglik(X, n, A.P)]);
    while (!O.done && O.epoch * n + (O.b + 1) * O.B <= c) {
      O.step();
      if (O.batch) for (const i of O.batch) seenAt[i] = now;
      histO.push([O.done ? O.T : O.epoch + O.b * O.B / n, G.loglik(X, n, O.P), O.done ? null : O.alpha]);
    }
    for (const ev of A.events.concat(O.events)) if (ev.wall === undefined) ev.wall = now;
  }
  const finishedAt = () => Math.max(cfg.TL, cfg.T) * cfg.n;

  function sizeCanvas(cv, ratio) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = ratio ? w * ratio : cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    return { ctx: cv.getContext("2d"), w: cv.width, h: cv.height, dpr };
  }
  const hue = j => (j * 137.508) % 360, ease = t => t < 0 ? 0 : t > 1 ? 1 : 1 - Math.pow(1 - t, 3);
  function shown(alg, now) {
    for (let i = alg.events.length - 1; i >= 0; i--) {
      const ev = alg.events[i];
      if (!ev.before) continue;
      const t = ease((now - ev.wall) / 350);
      if (t >= 1) break;
      const mix = (a, b) => a.map((v, q) => v + (b[q] - v) * t);
      return { w: mix(ev.before.w, alg.P.w), mu: mix(ev.before.mu, alg.P.mu), v: mix(ev.before.v, alg.P.v) };
    }
    return alg.P;
  }

  function drawPanel(cv, alg, online, now) {
    const { ctx, w: s, dpr } = sizeCanvas(cv, 1), n = cfg.n, K = cfg.K;
    const fg = css("--fg"), bg = css("--surface"), light = css("--pt-l");
    ctx.fillStyle = bg; ctx.fillRect(0, 0, s, s);
    const r = 1.3 * dpr;
    let sweepX = null;
    if (!online && !alg.done && (alg.pass > 0 || alg.pos > 0)) sweepX = alg.pos < n ? X[2 * alg.order[alg.pos]] : 1;
    ctx.fillStyle = fg;
    for (let i = 0; i < n; i++) {
      const x = X[2 * i];
      ctx.globalAlpha = sweepX !== null && x > sweepX ? 0.16 : 0.42;
      ctx.fillRect(x * s - r, (1 - X[2 * i + 1]) * s - r, 2 * r, 2 * r);
    }
    if (online) {
      ctx.fillStyle = css("--plain");
      const rb = 2 * dpr;
      for (let i = 0; i < n; i++) {
        const age = (now - seenAt[i]) / 400;
        if (age >= 1) continue;
        ctx.globalAlpha = 0.9 * (1 - age);
        ctx.fillRect(X[2 * i] * s - rb, (1 - X[2 * i + 1]) * s - rb, 2 * rb, 2 * rb);
      }
    }
    ctx.globalAlpha = 1;
    if (sweepX !== null) { ctx.strokeStyle = css("--lloyd"); ctx.lineWidth = 2 * dpr; ctx.beginPath(); ctx.moveTo(sweepX * s, 0); ctx.lineTo(sweepX * s, s); ctx.stroke(); }
    // components: ellipses at 1 and 2 standard deviations; fill strength follows the weight
    const P = shown(alg, now);
    ctx.save(); ctx.beginPath(); ctx.rect(0, 0, s, s); ctx.clip();
    for (let k = 0; k < K; k++) {
      const cx = P.mu[2 * k] * s, cy = (1 - P.mu[2 * k + 1]) * s, sx = Math.sqrt(P.v[2 * k]) * s, sy = Math.sqrt(P.v[2 * k + 1]) * s;
      const col = `hsl(${hue(k)} 70% ${light})`, wk = P.w[k];
      if (!(wk > 0)) continue;
      // outlines only, so overlapping components do not pile up into a haze; heavier components get a thicker, stronger line
      const rel = Math.min(2, wk * K);
      ctx.strokeStyle = col;
      ctx.beginPath(); ctx.ellipse(cx, cy, 2 * sx, 2 * sy, 0, 0, 2 * Math.PI);
      ctx.globalAlpha = 0.45 + 0.27 * rel; ctx.lineWidth = (0.8 + 1.2 * rel) * dpr; ctx.stroke();
      ctx.beginPath(); ctx.ellipse(cx, cy, sx, sy, 0, 0, 2 * Math.PI);
      ctx.globalAlpha = 0.35 + 0.2 * rel; ctx.lineWidth = 1 * dpr; ctx.setLineDash([3 * dpr, 3 * dpr]); ctx.stroke(); ctx.setLineDash([]);
      ctx.globalAlpha = 1; ctx.beginPath(); ctx.arc(cx, cy, 3.5 * dpr, 0, 2 * Math.PI); ctx.fillStyle = col; ctx.fill();
      ctx.lineWidth = 1.2 * dpr; ctx.strokeStyle = fg; ctx.stroke();
    }
    ctx.restore();
    for (const ev of alg.events) {
      if (ev.type !== "revive") continue;
      const age = (now - ev.wall) / 1100;
      if (age >= 1) continue;
      ctx.globalAlpha = 1 - age; ctx.strokeStyle = css("--penalty"); ctx.lineWidth = 1.5 * dpr; ctx.setLineDash([4 * dpr, 4 * dpr]);
      ctx.beginPath(); ctx.moveTo(ev.from[0] * s, (1 - ev.from[1]) * s); ctx.lineTo(ev.to[0] * s, (1 - ev.to[1]) * s); ctx.stroke();
      ctx.setLineDash([]); ctx.lineWidth = 2.5 * dpr;
      ctx.beginPath(); ctx.arc(ev.to[0] * s, (1 - ev.to[1]) * s, (7 + 18 * age) * dpr, 0, 2 * Math.PI); ctx.stroke();
      ctx.globalAlpha = 1;
    }
    const lastEv = online && alg.events.length ? alg.events[alg.events.length - 1] : null;
    if (lastEv && lastEv.type === "cleanup" && now - lastEv.wall < 2500) {
      const msg = "Clean-up: model rebuilt from the plain totals of the last pass";
      ctx.font = `${12 * dpr}px ${css("--sans")}`; ctx.textBaseline = "top";
      const tw = ctx.measureText(msg).width;
      ctx.globalAlpha = 1 - Math.max(0, (now - lastEv.wall - 1500) / 1000);
      ctx.fillStyle = bg; ctx.fillRect(6 * dpr, 6 * dpr, tw + 10 * dpr, 20 * dpr);
      ctx.fillStyle = fg; ctx.fillText(msg, 11 * dpr, 10 * dpr); ctx.globalAlpha = 1;
    }
  }

  function drawChart() {
    const cv = $("gC"), { ctx, w, h, dpr } = sizeCanvas(cv);
    const muted = css("--muted"), rule = css("--rule"), fg = css("--fg"), colA = css("--lloyd"), colO = css("--plain");
    ctx.fillStyle = css("--surface"); ctx.fillRect(0, 0, w, h);
    const padL = 48 * dpr, padR = 12 * dpr, top = 10 * dpr, strip = 22 * dpr, gap = 26 * dpr, plotB = h - 2 * strip - gap - 8 * dpr;
    const X0 = padL, X1 = w - padR, P = Math.max(cfg.TL, cfg.T), xs = p => X0 + (X1 - X0) * p / P;
    const vals = histA.concat(histO).map(r => r[1]), hi0 = Math.max(...vals), lo0 = Math.min(...vals);
    // the starting model is far below everything after the first update; leave it off the scale
    const later = histA.slice(1).concat(histO.slice(1)).map(r => r[1]);
    const lo = later.length ? Math.min(...later) : lo0, span = Math.max(hi0 - lo, 1e-3), y0 = lo - 0.05 * span, y1 = hi0 + 0.05 * span;
    const ys = v => top + (plotB - top) * (y1 - Math.max(v, y0)) / (y1 - y0);
    ctx.font = `${11 * dpr}px ${css("--sans")}`; ctx.strokeStyle = rule; ctx.lineWidth = 1 * dpr; ctx.fillStyle = muted;
    const stepv = Math.pow(10, Math.floor(Math.log10((y1 - y0) / 4))), mult = [1, 2, 5].find(m => (y1 - y0) / (m * stepv) <= 6) || 10;
    ctx.textAlign = "right"; ctx.textBaseline = "middle";
    for (let v = Math.ceil(y0 / (mult * stepv)) * mult * stepv; v <= y1; v += mult * stepv) {
      const y = ys(v); ctx.beginPath(); ctx.moveTo(X0, y); ctx.lineTo(X1, y); ctx.stroke();
      ctx.fillText(v.toFixed(Math.max(0, -Math.floor(Math.log10(mult * stepv)))).replace("-", "−"), X0 - 6 * dpr, y);
    }
    const every = P <= 12 ? 1 : P <= 30 ? 5 : P <= 60 ? 10 : 20;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let p = 0; p <= P; p += every) ctx.fillText(p === P ? `${p} passes` : String(p), p === P ? xs(p) - 18 * dpr : xs(p), plotB + 4 * dpr);
    ctx.lineWidth = 2 * dpr; ctx.strokeStyle = colA; ctx.beginPath();
    histA.forEach(([p, v], i) => { if (i === 0) ctx.moveTo(xs(p), ys(v)); else { ctx.lineTo(xs(p), ys(histA[i - 1][1])); ctx.lineTo(xs(p), ys(v)); } });
    const cp = clock / cfg.n;
    if (!A.done) ctx.lineTo(xs(Math.min(cp, P)), ys(histA[histA.length - 1][1]));
    ctx.stroke();
    ctx.strokeStyle = colO; ctx.beginPath(); histO.forEach(([p, v], i) => i ? ctx.lineTo(xs(p), ys(v)) : ctx.moveTo(xs(p), ys(v))); ctx.stroke();
    if (O.done) { const [p, v] = histO[histO.length - 1]; ctx.fillStyle = colO; ctx.beginPath(); ctx.arc(xs(p), ys(v), 3.5 * dpr, 0, 2 * Math.PI); ctx.fill(); }
    const yL = plotB + gap, yE = yL + strip;
    ctx.fillStyle = muted; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText("Classic", X0 - 6 * dpr, yL + strip / 2); ctx.fillText("e-means", X0 - 6 * dpr, yE + strip / 2);
    ctx.strokeStyle = rule; ctx.lineWidth = 1 * dpr; ctx.beginPath();
    ctx.moveTo(X0, yL + strip - 3 * dpr); ctx.lineTo(X1, yL + strip - 3 * dpr); ctx.moveTo(X0, yE + strip - 3 * dpr); ctx.lineTo(X1, yE + strip - 3 * dpr); ctx.stroke();
    ctx.strokeStyle = colA; ctx.lineWidth = 2 * dpr; ctx.beginPath();
    for (let i = 1; i < histA.length; i++) { const x = xs(histA[i][0]); ctx.moveTo(x, yL + strip - 3 * dpr); ctx.lineTo(x, yL + 2 * dpr); }
    ctx.stroke();
    ctx.strokeStyle = colO; ctx.lineWidth = Math.max(1, dpr); ctx.beginPath();
    for (let i = 1; i < histO.length; i++) { const [p, , a] = histO[i]; if (a === null) continue; const x = xs(p); ctx.moveTo(x, yE + strip - 3 * dpr); ctx.lineTo(x, yE + strip - 3 * dpr - (strip - 5 * dpr) * a / (cfg.a0 || 1)); }
    ctx.stroke();
    if (cp > 0 && cp < P) { ctx.strokeStyle = fg; ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.moveTo(xs(cp), top); ctx.lineTo(xs(cp), yE + strip); ctx.stroke(); ctx.globalAlpha = 1; }
  }

  function stats() {
    const n = cfg.n, a = histA[histA.length - 1][1], o = histO[histO.length - 1][1];
    const starved = P => P.w.filter(v => v < 0.1 / cfg.K).length;
    $("gStA").textContent = `Pass ${Math.min(clock / n, cfg.TL).toFixed(2)} of ${cfg.TL} · ${A.updates} update${A.updates === 1 ? "" : "s"} · log-likelihood ${a.toFixed(3)} · ${starved(A.P)} starved`;
    $("gStO").textContent = `Pass ${Math.min(clock / n, cfg.T).toFixed(2)} of ${cfg.T} · ${O.t} updates of ${O.B.toLocaleString("en")} points · α ${O.alpha.toFixed(3)} · ${O.revivals} revival${O.revivals === 1 ? "" : "s"} · log-likelihood ${o.toFixed(3)} · ${starved(O.P)} starved${O.done ? " · done" : ""}`;
    let txt = "Starved: components with under 10% of the average weight.";
    if (O.done) {
      const match = histA.find(r => r[0] > 0 && r[1] >= o);
      txt = `e-means-style after ${cfg.T} pass${cfg.T === 1 ? "" : "es"}: ${o.toFixed(3)}. ` + (match ? `Classic EM first reaches it at pass ${match[0]}.` : A.done ? `Classic EM did not reach it in ${cfg.TL} passes.` : "Classic EM has not reached it yet.");
    }
    $("gStC").textContent = txt;
  }

  function tick(ts) {
    const dt = last ? Math.min(0.1, (ts - last) / 1000) : 0; last = ts;
    if (playing && clock < finishedAt()) {
      clock = Math.min(finishedAt(), clock + dt * speed() * cfg.n); advanceTo(clock);
      if (clock >= finishedAt()) setPlaying(false);
    }
    const now = performance.now();
    drawPanel($("gA_cv"), A, false, now); drawPanel($("gO_cv"), O, true, now); drawChart(); stats();
    requestAnimationFrame(tick);
  }
  function setPlaying(p) { playing = p; $("gPlay").textContent = p ? "Pause" : clock >= finishedAt() ? "Replay" : "Play"; }
  $("gPlay").addEventListener("click", () => { if (!playing && clock >= finishedAt()) restart(); setPlaying(!playing); });
  $("gStep").addEventListener("click", () => {
    const target = !O.done ? O.epoch * cfg.n + (O.b + 1) * O.B : Math.min(finishedAt(), (A.pass + 1) * cfg.n);
    clock = Math.max(clock, target); advanceTo(clock); setPlaying(false);
  });
  $("gRestart").addEventListener("click", () => { restart(); setPlaying(true); });
  $("gReseed").addEventListener("click", () => { seed++; restart(); setPlaying(true); });
  for (const id of ["gKind", "gN", "gK", "gTe", "gTl", "gLaws", "gB", "gA", "gRevive"])
    ui[id].addEventListener("input", () => { restart(); setPlaying(playing || clock === 0); });
  ui.gSp.addEventListener("input", showValues);
  restart(); setPlaying(playing);
  requestAnimationFrame(tick);
})();
