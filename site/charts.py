"""Small SVG charts for the follow-ups page. Every number is copied from a table in EXTENSIONS.md or README.md;
the section is named on each chart."""
import html, math

W, H, L, R, T, B = 480, 270, 54, 16, 14, 44
CLS = {"full": "s-full", "recipe": "s-recipe", "alt": "s-alt", "alt2": "s-alt2"}


def _ticks_log(lo, hi):
    out = []
    for e in range(math.floor(math.log10(lo)) - 1, math.ceil(math.log10(hi)) + 1):
        for m in (1, 2, 5):
            v = m * 10 ** e
            if lo * 0.999 <= v <= hi * 1.001:
                out.append(v)
    return out


def _ticks_lin(lo, hi, n=5):
    raw = (hi - lo) / n
    step = 10 ** math.floor(math.log10(raw))
    step *= next(m for m in (1, 2, 2.5, 5, 10) if raw / step <= m)
    v, out = math.ceil(lo / step) * step, []
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def _fmt(v):
    s = f"{v:,.4f}".rstrip("0").rstrip(".") if abs(v) < 10 else f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.1f}".rstrip("0").rstrip(".")
    return s.replace("-", "−")


def line_chart(series, xlabel, ylabel, ylim, xlog=True, xlim=None, refs=(), yfmt=_fmt):
    xs_all = [x for s in series for x, _ in s["points"]]
    x0, x1 = xlim or (min(xs_all), max(xs_all))
    y0, y1 = ylim
    fx = (lambda x: L + (W - L - R) * (math.log(x) - math.log(x0)) / (math.log(x1) - math.log(x0))) if xlog else \
         (lambda x: L + (W - L - R) * (x - x0) / (x1 - x0))
    fy = lambda y: T + (H - T - B) * (y1 - y) / (y1 - y0)
    o = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="{html.escape(ylabel)} against {html.escape(xlabel)}">']
    for v in _ticks_lin(y0, y1):
        o.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{fy(v):.1f}" y2="{fy(v):.1f}"/><text class="tick" x="{L - 6}" y="{fy(v):.1f}" text-anchor="end" dominant-baseline="middle">{yfmt(v)}</text>')
    for v in (_ticks_log(x0, x1) if xlog else _ticks_lin(x0, x1)):
        o.append(f'<text class="tick" x="{fx(v):.1f}" y="{H - B + 16}" text-anchor="middle">{_fmt(v)}</text>')
    o.append(f'<text class="axis" x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle">{html.escape(xlabel)}</text>')
    o.append(f'<line class="frame" x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}"/>')
    for y, label in refs:
        o.append(f'<line class="ref" x1="{L}" x2="{W - R}" y1="{fy(y):.1f}" y2="{fy(y):.1f}"/><text class="reflabel" x="{W - R - 4}" y="{fy(y) - 5:.1f}" text-anchor="end">{html.escape(label)}</text>')
    for s in series:
        c = CLS[s["style"]]
        pts = [(fx(x), fy(min(max(y, y0), y1)), y) for x, y in s["points"]]
        if len(pts) > 1:
            o.append(f'<polyline class="{c}{" dash" if s.get("dash") else ""}" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b, _ in pts)}"/>')
        for a, b, y in pts:
            o.append(f'<circle class="{c} m" cx="{a:.1f}" cy="{b:.1f}" r="4"><title>{html.escape(s["name"])}: {_fmt(y)}</title></circle>')
            if y > y1 or y < y0:
                o.append(f'<text class="clip" x="{a + 7:.1f}" y="{b + (12 if y < y0 else 4):.1f}">{_fmt(y)}</text>')
    o.append("</svg>")
    return "".join(o)


def bar_chart(rows, groups, xlabel, xmax):
    """rows: [(label, style, [value or None per group])]; groups: [(name, shade class)]"""
    lw, rh, gap = 150, 13, 10
    h = T + len(rows) * (len(groups) * rh + gap) + 34
    fx = lambda v: lw + (W - lw - R - 34) * v / xmax
    o = [f'<svg viewBox="0 0 {W} {h}" class="chart" role="img" aria-label="{html.escape(xlabel)}">']
    for v in _ticks_lin(0, xmax, 4):
        o.append(f'<line class="grid" x1="{fx(v):.1f}" x2="{fx(v):.1f}" y1="{T}" y2="{h - 34}"/><text class="tick" x="{fx(v):.1f}" y="{h - 20}" text-anchor="middle">{_fmt(v)}</text>')
    o.append(f'<text class="axis" x="{(lw + W - R) / 2}" y="{h - 4}" text-anchor="middle">{html.escape(xlabel)}</text>')
    y = T
    for label, style, vals in rows:
        o.append(f'<text class="rowlabel" x="{lw - 8}" y="{y + len(groups) * rh / 2:.1f}" text-anchor="end" dominant-baseline="middle">{html.escape(label)}</text>')
        for (gname, shade), v in zip(groups, vals):
            if v is None:
                o.append(f'<text class="clip" x="{lw + 4}" y="{y + rh - 3}">not reached</text>')
            else:
                o.append(f'<rect class="{CLS[style]} bar {shade}" x="{lw}" y="{y + 1}" width="{fx(v) - lw:.1f}" height="{rh - 2}"><title>{html.escape(label)}, {html.escape(gname)}: {v}</title></rect>'
                         f'<text class="tick" x="{fx(v) + 4:.1f}" y="{y + rh - 3}">{v}</text>')
            y += rh
        y += gap
    o.append("</svg>")
    return "".join(o)


ROLE = {"full": "Classic", "recipe": "e-means-style", "alt": "Other", "alt2": "Other"}


def figure(title, finding, chart, legend, note, section):
    """legend: [(name, style, dash)] or [(name, style, dash, role)]; entries are grouped under their role."""
    groups = {}
    for name, st, dash, *role in legend:
        groups.setdefault(role[0] if role else ROLE[st], []).append(
            f'<span><i class="sw {CLS[st]}{" dash" if dash else ""}"></i>{html.escape(name)}</span>')
    leg = "".join(f'<span class="lg"><b>{html.escape(r)}</b><span class="items">{"".join(items)}</span></span>' for r, items in groups.items())
    return (f'<figure class="smallchart"><h3>{html.escape(title)}</h3><p>{finding}</p><div class="legend">{leg}</div>{chart}'
            f'<figcaption>{note} <a href="{section[0]}">{html.escape(section[1])}</a></figcaption></figure>')


def all_charts():
    c = {}
    c["passes"] = figure(
        "k-means: more passes", "Each doubling of the passes buys less: most of the gain comes by three passes. More Lloyd passes do nothing.",
        line_chart([{"name": "e-means", "style": "recipe", "points": [(1, 11.6), (2, -0.4), (3, -3.8), (4, -4.2), (5, -6.0), (6, -5.1), (8, -6.3), (10, -6.3), (15, -7.5), (25, -8.2)]},
                    {"name": "Lloyd", "style": "full", "points": [(25, 0.0), (50, 0.0), (100, 0.2)]}],
                   "passes over the data", "scan change", (-10, 12), xlim=(1, 100), refs=[(0, "Lloyd, 25 passes")]),
        [("Lloyd", "full", False), ("e-means", "recipe", False)],
        "SIFT1M, <var>k</var> = 16,384. Vectors scanned for 95% recall@1, % change against 25 Lloyd passes; lower is better.", ("extensions.html#2-do-more-passes-keep-helping", "Section 2"))
    c["penalty"] = figure(
        "k-means: size-penalty strength", "Helps on image descriptors from about 0.1 to 0.5; on word vectors every strength made the index worse. "
        "The <a href=\"demo.html\">k-means demo</a> has a slider for it.",
        line_chart([{"name": "SIFT, k = 16,384", "style": "recipe", "points": [(0.03, -3.1), (0.1, -6.1), (0.2, -7.6), (0.3, -7.7), (0.5, -7.3), (1, -4.1), (2, 4.0)]},
                    {"name": "DEEP, k = 16,384", "style": "alt2", "points": [(0.03, -2.7), (0.1, -5.7), (0.2, -6.3), (0.3, -6.5), (0.5, -5.6), (1, -2.3)]},
                    {"name": "GloVe, k = 4,096", "style": "alt", "points": [(0.03, 3.3), (0.1, 2.7), (0.2, 4.0), (0.3, 3.5), (0.5, 9.5), (1, 25.6)]}],
                   "penalty strength λ (held)", "scan change", (-10, 12), refs=[(0, "plain e-means")]),
        [("SIFT", "recipe", False, "e-means + penalty on"), ("DEEP", "alt2", False, "e-means + penalty on"), ("GloVe", "alt", False, "e-means + penalty on")],
        "All three lines are e-means with the penalty, one per data set; there is no classic method here. Vectors scanned at 95% recall (SIFT, DEEP: recall@10; GloVe: recall@1), % change against plain 3-pass e-means; lower is better.", ("extensions.html#5-a-size-penalty", "Section 5"))
    c["pq"] = figure(
        "Product quantization", "With the authors' smaller batch for low-dimensional data, three passes beat 100 Lloyd passes.",
        line_chart([{"name": "e-means, published batch", "style": "recipe", "dash": True, "points": [(1, 0.77), (2, -0.34), (3, -0.73), (5, -0.57), (10, -1.00)]},
                    {"name": "e-means, batch ÷ 8", "style": "recipe", "points": [(1, 0.08), (2, -0.84), (3, -1.05), (5, -1.19), (10, -1.35)]},
                    {"name": "Lloyd", "style": "full", "points": [(10, 2.02), (25, 0.0), (100, -1.00)]}],
                   "passes over the data", "error change", (-1.6, 2.2), xlim=(1, 100)),
        [("Lloyd", "full", False), ("e-means, published batch", "recipe", True), ("e-means, batch ÷ 8", "recipe", False)],
        "SIFT1M, 16 codebooks × 256 codes in 8 dimensions. Held-out reconstruction error, % change against 25 Lloyd passes; lower is better.", ("extensions.html#6-product-quantization", "Section 6"))
    c["gmm"] = figure(
        "Gaussian mixtures at full size", "Three e-means-style passes match 40–57 passes of classic EM; with many more passes EM ends slightly ahead.",
        line_chart([{"name": "e-means-style online EM", "style": "recipe", "points": [(1, -81.227), (3, -80.877), (10, -80.838)]},
                    {"name": "Classic EM", "style": "full", "points": [(25, -80.971), (100, -80.838), (200, -80.820)]}],
                   "passes over the data", "log-likelihood", (-81.3, -80.75), xlim=(1, 200), yfmt=lambda v: f"{v:.1f}".replace("-", "−")),
        [("EM, full passes", "full", False), ("online EM", "recipe", False)],
        "SIFT in 64 PCA dimensions, 1M rows, 256 diagonal components. Held-out log-likelihood per row; higher is better.", ("reproduction.html#3-the-same-recipe-for-gaussian-mixtures-em", "Reproduction, section 3"))
    c["nmf"] = figure(
        "NMF", "Three e-means-style passes match 34–79 classic full passes and come within 0.1% of scikit-learn's converged <code>NMF</code>.",
        line_chart([{"name": "e-means-style online HALS", "style": "recipe", "points": [(1, 0.80698), (3, 0.80398)]},
                    {"name": "Classic full-batch HALS", "style": "full", "points": [(30, 0.80443), (100, 0.80369)]},
                    {"name": "Textbook running average", "style": "alt", "points": [(3, 0.80827)]}],
                   "passes over the data", "error", (0.802, 0.809), xlim=(1, 100), refs=[(0.80314, "scikit-learn NMF, converged")]),
        [("full-batch HALS", "full", False), ("online HALS", "recipe", False), ("online, textbook running average", "alt", False)],
        "TF-IDF of 100,000 New York Times articles, 100 components. Held-out relative reconstruction error; lower is better.", ("extensions.html#nmf-it-transfers", "Section 8"))
    c["lda"] = figure(
        "Topic models (LDA)", "With the published step the e-means-style version is too jumpy here: the result gets worse after the first pass. Divided by four, it ties Hoffman's 2010 online rule.",
        line_chart([{"name": "Classic full-batch EM", "style": "full", "points": [(1, 3474), (2, 2503), (3, 2253), (5, 2115), (10, 2071)]},
                    {"name": "e-means-style, published step", "style": "recipe", "dash": True, "points": [(1, 2208), (2, 2289), (3, 2295), (5, 2203), (10, 2233)]},
                    {"name": "e-means-style, step ÷ 4", "style": "recipe", "points": [(1, 2160), (2, 2108), (3, 2101), (5, 2078), (10, 2090)]},
                    {"name": "Hoffman's online rule", "style": "alt2", "points": [(1, 2146), (2, 2104), (3, 2098), (5, 2100), (10, 2108)]}],
                   "passes over the data", "perplexity", (2050, 2350), xlim=(1, 10)),
        [("full-batch EM", "full", False), ("published step", "recipe", True), ("step ÷ 4", "recipe", False), ("Hoffman's 2010 online rule", "alt2", False)],
        "100,000 New York Times articles, 100 topics. Held-out perplexity; lower is better. Points off the scale are labelled.", ("extensions.html#lda-the-recipe-as-published-does-not-transfer", "Section 8"))
    c["hmm"] = figure(
        "Hidden Markov models", "With the formulas fed the number of characters rather than the number of chunks, one e-means-style pass equals 62–80 classic Baum-Welch passes.",
        line_chart([{"name": "Classic Baum-Welch", "style": "full", "points": [(20, 3.196), (60, 3.037), (100, 3.015)]},
                    {"name": "e-means-style, formulas fed chunks", "style": "recipe", "dash": True, "points": [(3, 3.096), (10, 3.048)]},
                    {"name": "e-means-style, formulas fed characters", "style": "recipe", "points": [(1, 3.027), (3, 2.982)]}],
                   "passes over the data", "bits per character", (2.96, 3.22), xlim=(1, 100)),
        [("Baum-Welch", "full", False), ("formulas fed chunks", "recipe", True), ("formulas fed characters", "recipe", False)],
        "text8 (Wikipedia as 27 symbols), 64 states. Held-out bits per character; lower is better.", ("extensions.html#9-hidden-markov-models", "Section 9"))
    c["als"] = figure(
        "Recommender factorization (ALS)", "With the formulas the e-means-style version gains nothing. With a hand-picked batch and a step of 1, one pass reaches what ALS reaches in about 25 sweeps and two passes beat 40 sweeps; part of that gain is regularization.",
        line_chart([{"name": "Classic ALS", "style": "full", "points": [(10, 0.7755), (20, 0.7737), (40, 0.7729)]},
                    {"name": "e-means-style, formulas", "style": "recipe", "dash": True, "points": [(1, 0.7817), (3, 0.7770), (10, 0.7752)]},
                    {"name": "e-means-style, batch 4,096, step 1", "style": "recipe", "points": [(1, 0.7735), (2, 0.7724), (3, 0.7723), (10, 0.7718)]}],
                   "passes (ALS: sweeps)", "RMSE", (0.771, 0.783), xlim=(1, 40)),
        [("ALS", "full", False), ("formulas", "recipe", True), ("batch 4,096, step 1", "recipe", False)],
        "MovieLens-20M, rank 64. Held-out RMSE; lower is better.", ("extensions.html#10-recommender-factorization-als", "Section 10"))
    c["pagerank"] = figure(
        "PageRank", "Only one ingredient applies: update in pieces and use fresh values, which is Gauss-Seidel. Forgetting and a step curve slow it down.",
        bar_chart([("Power iteration", "full", [65, 119]), ("Gauss-Seidel", "full", [34, 62]), ("256 blocks, fresh values", "recipe", [34, 80]),
                   ("Fading running total", "recipe", [135, 249]), ("Cosine step", "recipe", [80, None])],
                  [("to residual 1e-6", ""), ("to residual 1e-10", "pale")], "sweeps to reach the residual", 250),
        [("power iteration, Gauss-Seidel", "full", False), ("one recipe ingredient per bar", "recipe", False)],
        "Upper bar of each pair: residual 1e-6; lower, paler bar: 1e-10. web-Google graph, 875,713 pages, damping 0.85. Fewer is better.", ("extensions.html#11-pagerank", "Section 11"))
    return c
