// Drives the side-by-side demo: one clock in passes over the data, both algorithms advanced to it every frame.
(function () {
  "use strict";
  const D = window.EMDemo, $ = id => document.getElementById(id);
  const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const ui = {};
  ["kind", "n", "k", "te", "tl", "sp", "laws", "b", "a", "cosine", "revive"].forEach(id => ui[id] = $(id));
  const reduceMotion = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  let seed = 1, X = null, dataKey = "", C0 = null, L = null, E = null, clock = 0, playing = !reduceMotion, last = 0;
  let histL = [], histE = [], cfg = null;

  const speed = () => +(0.1 * Math.pow(50, +ui.sp.value)).toFixed(2);   // 0.1 to 5 passes per second
  function settings() {
    const n = +ui.n.value, k = +ui.k.value, T = +ui.te.value;
    let B = Math.pow(2, +ui.b.value), a0 = +ui.a.value;
    if (ui.laws.checked) {
      B = D.lawBatch(n, k, T); a0 = D.lawStep(n, k, B, T);
      ui.b.value = Math.log2(B); ui.a.value = a0.toFixed(2);
    }
    ui.b.disabled = ui.a.disabled = ui.laws.checked;
    return { kind: ui.kind.value, n, k, T, TL: +ui.tl.value, B: Math.min(B, n), a0, cosine: ui.cosine.checked, revive: ui.revive.checked };
  }
  function showValues() {
    $("nOut").textContent = (+ui.n.value).toLocaleString("en");
    $("kOut").textContent = ui.k.value; $("teOut").textContent = ui.te.value; $("tlOut").textContent = ui.tl.value;
    $("spOut").textContent = speed();
    $("bOut").textContent = cfg ? cfg.B.toLocaleString("en") : ""; $("aOut").textContent = cfg ? cfg.a0.toFixed(2) : "";
  }

  function restart() {
    cfg = settings();
    const key = `${cfg.kind}|${cfg.n}|${seed}`;
    if (key !== dataKey) { X = D.makeData(cfg.kind, cfg.n, seed); dataKey = key; }
    C0 = D.initCentres(X, cfg.n, cfg.k, seed);
    L = new D.Lloyd(X, cfg.n, cfg.k, C0, cfg.TL);
    E = new D.EMeans(X, cfg.n, cfg.k, C0, cfg, seed);
    clock = 0;
    const m0 = D.measure(X, cfg.n, C0, cfg.k).mse;
    histL = [[0, m0]]; histE = [[0, m0, null]];
    showValues(); frame(true);
  }

  // Advance both to the clock; record the error after every update.
  function advanceTo(c) {
    const n = cfg.n, now = performance.now();
    const before = L.updates;
    L.advance(c);
    if (L.updates !== before) histL.push([L.pass, D.measure(X, n, L.C, cfg.k).mse]);
    while (!E.done && E.epoch * n + (E.b + 1) * E.B <= c) {
      E.step();
      histE.push([E.done ? E.T : E.epoch + E.b * E.B / n, D.measure(X, n, E.C, cfg.k).mse, E.done ? null : E.alpha]);
    }
    for (const ev of L.events.concat(E.events)) if (ev.wall === undefined) ev.wall = now;
  }
  const finishedAt = () => Math.max(cfg.TL, cfg.T) * cfg.n;

  // ---------- drawing ----------
  function sizeCanvas(cv, ratio) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = ratio ? w * ratio : cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    return { ctx: cv.getContext("2d"), w: cv.width, h: cv.height, dpr };
  }
  const hue = j => (j * 137.508) % 360;
  const ease = t => t < 0 ? 0 : t > 1 ? 1 : 1 - Math.pow(1 - t, 3);

  // Centres as drawn: a jump (Lloyd update, e-means clean-up) is shown as a short glide from the old positions.
  function shownCentres(alg, now) {
    const C = alg.C, evs = alg.events;
    for (let i = evs.length - 1; i >= 0; i--) {
      const ev = evs[i];
      if (!ev.before) continue;
      const t = ease((now - ev.wall) / 350);
      if (t >= 1) break;
      return C.map((v, q) => ev.before[q] + (v - ev.before[q]) * t);
    }
    return C;
  }

  function drawPoints(cv, alg, isE, now) {
    const { ctx, w, dpr } = sizeCanvas(cv, 1), n = cfg.n, k = cfg.k, s = w;
    const light = css("--pt-l"), muted = css("--muted");
    ctx.fillStyle = css("--surface"); ctx.fillRect(0, 0, s, s);
    const r = 1.5 * dpr, lab = alg.lab;
    const lloydPos = isE ? -1 : (alg.done ? n : alg.pos);
    for (let i = 0; i < n; i++) {
      const j = lab[i], x = X[2 * i] * s, y = (1 - X[2 * i + 1]) * s;
      if (j < 0) { ctx.globalAlpha = 0.35; ctx.fillStyle = muted; }
      else { ctx.globalAlpha = !isE && i >= lloydPos ? 0.28 : 0.6; ctx.fillStyle = `hsl(${hue(j)} 60% ${light})`; }
      ctx.fillRect(x - r, y - r, 2 * r, 2 * r);
    }
    ctx.globalAlpha = 1;
    if (isE && alg.batch) {
      const rb = 2.6 * dpr;
      for (const i of alg.batch) {
        const j = lab[i];
        ctx.fillStyle = `hsl(${hue(j)} 70% ${light})`;
        ctx.beginPath(); ctx.arc(X[2 * i] * s, (1 - X[2 * i + 1]) * s, rb, 0, 2 * Math.PI); ctx.fill();
      }
    }
    // revival: dashed path from the stranded centre to the busy cluster it splits, and a ring there
    for (const ev of alg.events) {
      if (ev.type !== "revive") continue;
      const age = (now - ev.wall) / 1100;
      if (age >= 1) continue;
      const pen = css("--penalty");
      ctx.globalAlpha = 1 - age; ctx.strokeStyle = pen; ctx.lineWidth = 1.5 * dpr;
      ctx.setLineDash([4 * dpr, 4 * dpr]);
      ctx.beginPath(); ctx.moveTo(ev.from[0] * s, (1 - ev.from[1]) * s); ctx.lineTo(ev.to[0] * s, (1 - ev.to[1]) * s); ctx.stroke();
      ctx.setLineDash([]); ctx.lineWidth = 2.5 * dpr;
      ctx.beginPath(); ctx.arc(ev.to[0] * s, (1 - ev.to[1]) * s, (7 + 18 * age) * dpr, 0, 2 * Math.PI); ctx.stroke();
      ctx.globalAlpha = 1;
    }
    const C = shownCentres(alg, now), fg = css("--fg"), bg = css("--surface");
    for (let j = 0; j < k; j++) {
      const x = C[2 * j] * s, y = (1 - C[2 * j + 1]) * s;
      ctx.beginPath(); ctx.arc(x, y, 6 * dpr, 0, 2 * Math.PI);
      ctx.fillStyle = `hsl(${hue(j)} 70% ${light})`; ctx.fill();
      ctx.lineWidth = 3.5 * dpr; ctx.strokeStyle = bg; ctx.stroke();
      ctx.lineWidth = 1.6 * dpr; ctx.strokeStyle = fg; ctx.stroke();
    }
    const cleanup = isE && alg.events.length && alg.events[alg.events.length - 1].type === "cleanup" ? alg.events[alg.events.length - 1] : null;
    if (cleanup && now - cleanup.wall < 2500) {
      ctx.globalAlpha = 1 - Math.max(0, (now - cleanup.wall - 1500) / 1000);
      ctx.font = `${13 * dpr}px ${css("--sans")}`; ctx.fillStyle = fg; ctx.textBaseline = "top";
      ctx.fillText("Clean-up: each centre moved to the mean of its last-pass points", 10 * dpr, 10 * dpr);
      ctx.globalAlpha = 1;
    }
  }

  function niceLogTicks(lo, hi) {
    const out = [];
    for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++)
      for (const m of [1, 2, 5]) { const v = m * Math.pow(10, e); if (v >= lo && v <= hi) out.push(v); }
    if (out.length < 3)
      for (let e = Math.floor(Math.log10(lo)) - 1; e <= Math.ceil(Math.log10(hi)); e++)
        for (const m of [1.5, 3, 4, 6, 7, 8, 9]) { const v = m * Math.pow(10, e); if (v >= lo && v <= hi) out.push(v); }
    return out.sort((a, b) => a - b);
  }

  function drawChart() {
    const cv = $("cvC"), { ctx, w, h, dpr } = sizeCanvas(cv);
    const font = css("--sans"), fg = css("--fg"), muted = css("--muted"), rule = css("--rule");
    const colL = css("--lloyd"), colE = css("--plain");
    ctx.fillStyle = css("--surface"); ctx.fillRect(0, 0, w, h);
    const padL = 44 * dpr, padR = 12 * dpr, top = 10 * dpr, strip = 22 * dpr, gap = 26 * dpr;
    const plotB = h - 2 * strip - gap - 8 * dpr, X0 = padL, X1 = w - padR, P = Math.max(cfg.TL, cfg.T);
    const xs = p => X0 + (X1 - X0) * p / P;
    const all = histL.concat(histE).map(r => r[1]);
    let lo = Math.min(...all) * 0.97, hi = Math.max(...all) * 1.03;
    if (hi / lo < 1.2) { const m = Math.sqrt(hi * lo); lo = m / 1.1; hi = m * 1.1; }
    const ys = v => top + (plotB - top) * (Math.log(hi) - Math.log(v)) / (Math.log(hi) - Math.log(lo));
    ctx.font = `${11 * dpr}px ${font}`; ctx.textBaseline = "middle";
    ctx.strokeStyle = rule; ctx.lineWidth = 1 * dpr; ctx.fillStyle = muted;
    for (const v of niceLogTicks(lo, hi)) {
      const y = ys(v); ctx.beginPath(); ctx.moveTo(X0, y); ctx.lineTo(X1, y); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(String(+(v * 1e4).toPrecision(2)), X0 - 6 * dpr, y);
    }
    const every = P <= 12 ? 1 : P <= 30 ? 5 : 10;
    ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (let p = 0; p <= P; p += every) ctx.fillText(p === P ? `${p} passes` : String(p), p === P ? xs(p) - 18 * dpr : xs(p), plotB + 4 * dpr);
    // Lloyd: constant between updates, so a step line
    ctx.lineWidth = 2 * dpr; ctx.strokeStyle = colL; ctx.beginPath();
    histL.forEach(([p, v], i) => { if (i === 0) ctx.moveTo(xs(p), ys(v)); else { ctx.lineTo(xs(p), ys(histL[i - 1][1])); ctx.lineTo(xs(p), ys(v)); } });
    const cp = clock / cfg.n;
    if (!L.done) ctx.lineTo(xs(Math.min(cp, P)), ys(histL[histL.length - 1][1]));
    ctx.stroke();
    ctx.strokeStyle = colE; ctx.beginPath();
    histE.forEach(([p, v], i) => i === 0 ? ctx.moveTo(xs(p), ys(v)) : ctx.lineTo(xs(p), ys(v)));
    ctx.stroke();
    if (E.done) { const [p, v] = histE[histE.length - 1]; ctx.fillStyle = colE; ctx.beginPath(); ctx.arc(xs(p), ys(v), 3.5 * dpr, 0, 2 * Math.PI); ctx.fill(); }
    // update strips
    const yL = plotB + gap, yE = yL + strip;
    ctx.fillStyle = muted; ctx.textAlign = "right"; ctx.textBaseline = "middle";
    ctx.fillText("Lloyd", X0 - 6 * dpr, yL + strip / 2); ctx.fillText("e-means", X0 - 6 * dpr, yE + strip / 2);
    ctx.strokeStyle = rule; ctx.lineWidth = 1 * dpr;
    ctx.beginPath(); ctx.moveTo(X0, yL + strip - 3 * dpr); ctx.lineTo(X1, yL + strip - 3 * dpr); ctx.moveTo(X0, yE + strip - 3 * dpr); ctx.lineTo(X1, yE + strip - 3 * dpr); ctx.stroke();
    ctx.strokeStyle = colL; ctx.lineWidth = 2 * dpr; ctx.beginPath();
    for (let i = 1; i < histL.length; i++) { const x = xs(histL[i][0]); ctx.moveTo(x, yL + strip - 3 * dpr); ctx.lineTo(x, yL + 2 * dpr); }
    ctx.stroke();
    ctx.strokeStyle = colE; ctx.lineWidth = Math.max(1, 1 * dpr); ctx.beginPath();
    const aMax = cfg.a0 || 1;
    for (let i = 1; i < histE.length; i++) {
      const [p, , a] = histE[i]; if (a === null) continue;
      const x = xs(p), hgt = (strip - 5 * dpr) * a / aMax; ctx.moveTo(x, yE + strip - 3 * dpr); ctx.lineTo(x, yE + strip - 3 * dpr - hgt);
    }
    ctx.stroke();
    // clock
    if (cp > 0 && cp < P) { ctx.strokeStyle = fg; ctx.globalAlpha = 0.35; ctx.lineWidth = 1 * dpr; ctx.beginPath(); ctx.moveTo(xs(cp), top); ctx.lineTo(xs(cp), yE + strip); ctx.stroke(); ctx.globalAlpha = 1; }
  }

  const fmt = v => (v * 1e4).toFixed(2);
  const pctDiff = (a, b) => { const d = 100 * (a / b - 1); return (d >= 0 ? "+" : "−") + Math.abs(d).toFixed(1) + "%"; };
  function stats() {
    const n = cfg.n, mL = histL[histL.length - 1], mE = histE[histE.length - 1];
    const sL = D.measure(X, n, L.C, cfg.k).starved, sE = D.measure(X, n, E.C, cfg.k).starved;
    $("stL").textContent = `Pass ${Math.min(clock / n, cfg.TL).toFixed(2)} of ${cfg.TL} · ${L.updates} update${L.updates === 1 ? "" : "s"} · error ${fmt(mL[1])} · ${sL} starved`;
    $("stE").textContent = `Pass ${Math.min(clock / n, cfg.T).toFixed(2)} of ${cfg.T} · ${E.t} updates of ${E.B.toLocaleString("en")} points · α ${E.alpha.toFixed(3)} · ${E.revivals} revival${E.revivals === 1 ? "" : "s"} · error ${fmt(mE[1])} · ${sE} starved${E.done ? " · done" : ""}`;
    let txt = "";
    if (E.done) {
      const eFinal = mE[1], atT = histL.find(r => r[0] === cfg.T), lFinal = L.done ? histL[histL.length - 1] : null;
      txt = `e-means after ${cfg.T} pass${cfg.T === 1 ? "" : "es"}: ${fmt(eFinal)}.`;
      if (atT) txt += ` Lloyd after ${cfg.T}: ${fmt(atT[1])} (e-means ${pctDiff(eFinal, atT[1])}).`;
      if (lFinal && cfg.TL !== cfg.T) txt += ` Lloyd after ${cfg.TL}: ${fmt(lFinal[1])} (e-means ${pctDiff(eFinal, lFinal[1])}).`;
    }
    $("stC").textContent = txt || "Starved: clusters holding fewer than 10% of the average number of points.";
  }

  function frame(force) {
    const now = performance.now();
    drawPoints($("cvL"), L, false, now);
    drawPoints($("cvE"), E, true, now);
    drawChart();
    stats();
  }

  function tick(ts) {
    const dt = last ? Math.min(0.1, (ts - last) / 1000) : 0; last = ts;
    if (playing && clock < finishedAt()) {
      clock = Math.min(finishedAt(), clock + dt * speed() * cfg.n);
      advanceTo(clock);
      if (clock >= finishedAt()) setPlaying(false);
    }
    frame();
    requestAnimationFrame(tick);
  }

  function setPlaying(p) { playing = p; $("play").textContent = p ? "Pause" : clock >= finishedAt() ? "Replay" : "Play"; }

  $("play").addEventListener("click", () => {
    if (!playing && clock >= finishedAt()) restart();
    setPlaying(!playing);
  });
  $("step").addEventListener("click", () => {
    setPlaying(false);
    const target = !E.done ? E.epoch * cfg.n + (E.b + 1) * E.B : Math.min(finishedAt(), (L.pass + 1) * cfg.n);
    clock = Math.max(clock, target); advanceTo(clock); setPlaying(false);
  });
  $("restart").addEventListener("click", () => { restart(); setPlaying(true); });
  $("reseed").addEventListener("click", () => { seed++; restart(); setPlaying(true); });
  for (const id of ["kind", "n", "k", "te", "tl", "laws", "b", "a", "cosine", "revive"])
    ui[id].addEventListener("input", () => { if (id === "b" || id === "a") ui.laws.checked = false; restart(); setPlaying(playing || clock === 0); });
  ui.sp.addEventListener("input", showValues);

  restart(); setPlaying(playing);
  requestAnimationFrame(tick);
})();
