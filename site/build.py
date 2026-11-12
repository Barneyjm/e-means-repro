"""Build the GitHub Pages site into docs/ from README.md, EXTENSIONS.md and results/check/.

Usage: python site/build.py      (needs the `markdown` package: pip install -e ".[site]")
The Markdown files stay the source of truth; docs/ is generated and committed so Pages can serve it as is."""
import collections, glob, html, json, os, re, shutil, sys
import numpy as np
import markdown

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import charts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs")
REPO = "https://github.com/Barneyjm/e-means-repro"
PAGES = {"README.md": "reproduction.html", "EXTENSIONS.md": "extensions.html"}


def slug(text, sep="-"):
    """GitHub's heading anchors: lower case, punctuation dropped, spaces to hyphens."""
    text = re.sub(r"<[^>]+>", "", html.unescape(text)).strip().lower()
    return re.sub(r"\s", sep, re.sub(r"[^\w\- ]", "", text))


def rewrite_links(md):
    def fix(m):
        text, target = m.group(1), m.group(2)
        if re.match(r"[a-z]+:", target) or target.startswith("#"):
            return m.group(0)
        path, _, anchor = target.partition("#")
        if path in PAGES:
            return f"[{text}]({PAGES[path]}{'#' + anchor if anchor else ''})"
        return f"[{text}]({REPO}/blob/main/{target})"
    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", fix, md)


def render(md_name):
    md = open(os.path.join(ROOT, md_name), encoding="utf-8").read()
    body = markdown.markdown(rewrite_links(md), extensions=["tables", "fenced_code", "toc", "sane_lists"],
                             extension_configs={"toc": {"slugify": slug, "permalink": False}})
    body = re.sub(r"<table>", '<div class="table-wrap"><table>', body).replace("</table>", "</table></div>")
    return body


DATASETS = [  # (folder, k, label) for the chart; the copy-free NYT file never reaches 95% and is left out
    ("nyt256", 4096, "New York Times"), ("nyt256", 1024, "New York Times"),
    ("lastfm", 4096, "Last.fm"), ("lastfm", 1024, "Last.fm"), ("lastfmdedup", 4096, "Last.fm, copies removed"),
    ("sift", 16384, "SIFT 1M"), ("ccnews", 4096, "CC-News"), ("ccnews", 1024, "CC-News"),
    ("landmark", 4096, "Landmark"), ("landmark", 1024, "Landmark"), ("yahoo", 4096, "Yahoo Answers"), ("yahoo", 1024, "Yahoo Answers"),
]
SERIES = [("auth3", "plain", "Authors' e-means, 3 passes"), ("auth3+0.3", "penalty", "Same, plus the size penalty (0.3, faded)")]


def chart_rows():
    """Change in distance computations at 95% recall@1 against Faiss k-means (25 iterations), paired by seed."""
    rows = []
    for d, k, name in DATASETS:
        I = collections.defaultdict(dict)
        for r in map(json.loads, open(os.path.join(ROOT, "results", "check", d, f"index_k{k}.jsonl"))):
            m = re.match(r".*_k\d+_(.*)_s(\d+)\.npy", r["centroids"]); I[m.group(1)][int(m.group(2))] = r["ndis_r1>=0.95"]
        cells = {}
        for v, cls, _ in SERIES:
            c = [100 * (I[v][s] / I["faiss25"][s] - 1) for s in I[v] if I[v][s] and I["faiss25"].get(s)]
            cells[cls] = (float(np.mean(c)), float(np.std(c)), len(c))
        rows.append((name, k, cells))
    return rows


def pct(v, sign=True):
    t = f"{round(v):+d}%" if sign else f"{round(v):d}"
    return "0%" if t in ("+0%", "-0%") else t.replace("-", "\u2212")


def chart_html():
    lo, hi = -90, 25
    x = lambda v: f"{100 * (min(max(v, lo), hi) - lo) / (hi - lo):.2f}%"
    out = ['<div class="dotplot" role="img" aria-label="Change in distance computations per query against Faiss k-means, by dataset">']
    for name, k, cells in chart_rows():
        marks = []
        for cls, (m, sd, n) in cells.items():
            marks.append(f'<span class="bar {cls}" style="left:{x(m - sd)};width:calc({x(m + sd)} - {x(m - sd)})"></span>'
                         f'<span class="dot {cls}" style="left:{x(m)}" title="{pct(m)} ± {sd:.0f} ({n} seeds)"></span>')
        vals = " / ".join(f'<span class="{c}">{pct(m)}</span>' for c, (m, _, _) in cells.items())
        out.append(f'<div class="row"><div class="lab">{html.escape(name)} <span class="k">k = {k:,}</span><span class="vals">{vals}</span></div>'
                   f'<div class="track"><span class="zero" style="left:{x(0)}"></span>{"".join(marks)}</div></div>')
    ticks = "".join(f'<span style="left:{x(t)}">{pct(t)}</span>' if t else f'<span style="left:{x(0)}">0</span>' for t in range(-80, 21, 20))
    out.append(f'<div class="row axis"><div class="lab"></div><div class="track ticks">{ticks}</div></div></div>')
    legend = "".join(f'<span><i class="dot {cls}"></i>{html.escape(lab)}</span>' for _, cls, lab in SERIES)
    return f'<div class="legend">{legend}</div>' + "\n".join(out)


FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600'
         '&family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap">')


def page(title, desc, current, body, wide=False, scripts=()):
    nav = "".join(f'<a href="{href}"{" aria-current=page" if href == current else ""}>{lab}</a>' for href, lab in
                  [("index.html", "Overview"), ("demo.html", "k-means demo"), ("followups.html", "Other algorithms"), ("reproduction.html", "Reproduction"), ("extensions.html", "Full write-up"), (REPO, "Code")])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(desc)}">
{FONTS}
<link rel="stylesheet" href="style.css">
</head>
<body>
<header class="site"><div class="wrap"><a class="name" href="index.html">e-means-repro</a><nav>{nav}</nav></div></header>
<main class="wrap{' doc' if wide else ''}">
{body}
</main>
<footer class="site"><div class="wrap">An independent reproduction. The method is the work of the paper's authors
(<a href="https://arxiv.org/abs/2611.09501">arXiv 2611.09501</a>); this site is not affiliated with them. Code MIT,
<a href="{REPO}">{REPO.removeprefix("https://")}</a>. Built from the repository's Markdown by <code>site/build.py</code>.</div></footer>
{"".join(f'<script src="{s}"></script>' for s in scripts)}
</body>
</html>
"""


INDEX = """
<header class="intro">
  <p class="kicker">Independent reproduction · arXiv 2611.09501</p>
  <h1>An independent check of e-means</h1>
  <p class="lede">The paper <em>e-means: outperforming k-means (in one-tenth the epochs)</em> says a mini-batch k-means with a
  decaying running average, a cosine step schedule and a rule that re-seeds near-empty clusters gets a better result in
  2–3 passes over the data than standard k-means gets in 25. I re-implemented it from the paper, measured it, wrote a
  scikit-learn estimator, and later compared it with the authors' released code.</p>
  <p><a href="demo.html">See the two algorithms side by side</a> in an interactive demo.</p>
</header>

<section>
  <h2>Result on SIFT1M</h2>
  <dl class="figures">
    <div><dt>Test error at 3 passes, against 25 Lloyd passes</dt><dd>−0.37%</dd><p>The paper reports −0.35% for the same dataset and <var>k</var>.</p></div>
    <div><dt>Training time, GPU</dt><dd>7.9× less</dd><p>2.3 s against 18.5 s, same distance kernel for both.</p></div>
    <div><dt>Training time, CPU, against scikit-learn</dt><dd>9× less</dd><p><code>EMeans(n_epochs=3)</code> 12.5 s, <code>KMeans(max_iter=25)</code> 112.7 s, slightly lower error.</p></div>
  </dl>
  <p class="caption">1,000,000 vectors, 128 dimensions, <var>k</var> = 16,384, random initialization. Error is the mean squared distance to the nearest centre on the 10,000 SIFT queries, which the clustering never sees. GPU: 3 seeds on an Apple M2 Max.</p>
  <div class="table-wrap"><table>
    <thead><tr><th>Method</th><th>Passes</th><th>Time</th><th>Test error vs Lloyd</th><th>Train error vs Lloyd</th><th>Clusters with &lt; 10 points</th></tr></thead>
    <tbody>
      <tr><td>Lloyd</td><td>25</td><td>18.46 s</td><td>41,711</td><td>38,260</td><td>182</td></tr>
      <tr><td>e-means</td><td>1</td><td>0.83 s</td><td>+0.80%</td><td>+3.51%</td><td>257</td></tr>
      <tr><td>e-means</td><td>2</td><td>1.62 s</td><td>−0.09%</td><td>+0.39%</td><td>53</td></tr>
      <tr><td>e-means</td><td>3</td><td>2.33 s</td><td>−0.37%</td><td>−0.26%</td><td>40</td></tr>
      <tr><td>e-means</td><td>5</td><td>3.86 s</td><td>−0.44%</td><td>−1.03%</td><td>21</td></tr>
    </tbody>
  </table></div>
  <p>Details, the CPU table and small-data results: <a href="reproduction.html#what-i-measured">Reproduction</a>.</p>
</section>

<section>
  <h2>In a real Faiss search index</h2>
  <p>The usual reason to run k-means at this scale is an inverted-file (IVF) search index. Each set of centroids below
  went into a Faiss <code>IndexIVFFlat</code>, and the measure is how many distance computations a query needs to find its
  true nearest neighbour 95% of the time. Zero is Faiss's own k-means with 25 iterations; left is cheaper.</p>
  {chart}
  <p class="caption">Mean over 3 seeds, bars ± one standard deviation, 10,000 held-out queries per dataset, CPU. The authors' NumPy
  code with its defaults. Full tables, including my estimator (which matches it): <a href="extensions.html#14-recheck-with-the-authors-code-and-what-it-changed">section 14</a>.</p>
  <ul>
    <li>Plain e-means is about as good as Faiss k-means on SIFT, CC-News, landmark and Yahoo Answers, and better on New York Times and Last.fm, with 5–8× less training time.</li>
    <li>The size penalty is an idea I added (DeSieno 1988, known as balanced k-means). It saves a few percent on SIFT, CC-News and landmark at the larger <var>k</var>, nothing on Yahoo Answers, and costs most of the gain on Last.fm.</li>
    <li>The New York Times numbers depend on duplicate rows: 21% of that file's rows are exact copies of other rows. With the copies removed, Faiss k-means is the cheapest or tied and the penalty saves nothing.</li>
  </ul>
</section>

<section>
  <h2>Use it</h2>
  <pre><code>pip install git+https://github.com/Barneyjm/e-means-repro

from emeans_sklearn import EMeans
km = EMeans(n_clusters=4096, n_epochs=3, random_state=0).fit(X)
km.cluster_centers_, km.labels_, km.predict(X_new)</code></pre>
  <p>A scikit-learn estimator in plain numpy and scipy: it passes scikit-learn's <code>check_estimator</code> suite and works in
  <code>Pipeline</code> and <code>GridSearchCV</code>. Batch size and step size default to the paper's formulas. Worth it for large
  <var>n</var> and large <var>k</var>; on small data it gains nothing. The same recipe for diagonal Gaussian mixtures is in
  <code>OnlineGaussianMixture</code>.</p>
</section>

<section>
  <h2>Read more</h2>
  <ul class="links">
    <li><a href="demo.html">k-means demo</a>: Lloyd's k-means and e-means side by side on 2-D points, with sliders for the settings.</li>
    <li><a href="followups.html">Other algorithms</a>: an animated Gaussian mixture (EM against the online recipe) and charts for product quantization, NMF, topic models, hidden Markov models, ALS and PageRank.</li>
    <li><a href="reproduction.html">Reproduction</a>: what the paper claims, what I measured, the algorithm, how to rerun it.</li>
    <li><a href="extensions.html">Follow-up experiments</a>: search indexes on more datasets, a dozen older ideas tried on top of the recipe, the recipe applied to product quantization, mixtures, topic models, NMF, hidden Markov models, ALS and PageRank, and the recheck against the authors' code.</li>
    <li><a href="{repo}/tree/main/results">Raw logs</a> for every number on this site.</li>
  </ul>
</section>
"""

CSS = """:root {
  --bg: #f6f5f1; --surface: #ffffff; --fg: #1c1e22; --muted: #5b616b; --rule: #dad8d1; --code-bg: #eeece6;
  --accent: #1f5f8b; --plain: #1f5f8b; --penalty: #b4532a; --zero: #8a8f98; --lloyd: #8a6a1f; --pt-l: 42%;
  --serif: "Source Serif 4", Georgia, "Times New Roman", serif;
  --sans: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #14161a; --surface: #1b1e23; --fg: #e5e3dd; --muted: #a0a6af; --rule: #30343b; --code-bg: #21252b;
  --accent: #82b6dd; --plain: #82b6dd; --penalty: #e38b5d; --zero: #6d727a; --lloyd: #d6ad5c; --pt-l: 64%; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #14161a; --surface: #1b1e23; --fg: #e5e3dd; --muted: #a0a6af; --rule: #30343b; --code-bg: #21252b;
  --accent: #82b6dd; --plain: #82b6dd; --penalty: #e38b5d; --zero: #6d727a; --lloyd: #d6ad5c; --pt-l: 64%; color-scheme: dark; }
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--bg); color: var(--fg); font: 18px/1.6 var(--serif); }
a { color: var(--accent); text-underline-offset: 2px; }
.wrap { max-width: 860px; margin: 0 auto; padding-inline: 20px; }
.wrap.doc { max-width: 1040px; }
header.site { border-bottom: 1px solid var(--rule); font-family: var(--sans); font-size: 15px; }
header.site .wrap { display: flex; flex-wrap: wrap; align-items: baseline; justify-content: space-between; gap: 6px 24px; padding-block: 14px; }
header.site .name { font-family: var(--mono); font-weight: 500; color: var(--fg); text-decoration: none; }
header.site nav { display: flex; flex-wrap: wrap; gap: 4px 18px; }
header.site nav a { color: var(--muted); text-decoration: none; }
header.site nav a:hover, header.site nav a[aria-current] { color: var(--fg); }
header.site nav a[aria-current] { text-decoration: underline; text-decoration-color: var(--accent); text-decoration-thickness: 2px; text-underline-offset: 6px; }
main { padding-block: 40px 64px; }
main :is(p, ul, ol, h1, h2, h3, h4, blockquote) { max-width: 72ch; }
h1, h2, h3, h4 { font-family: var(--sans); line-height: 1.25; text-wrap: balance; font-weight: 600; }
h1 { font-size: 2.1rem; margin: 0.2em 0 0.4em; letter-spacing: -0.01em; }
h2 { font-size: 1.4rem; margin: 2.2em 0 0.6em; padding-top: 0.8em; border-top: 1px solid var(--rule); }
h3 { font-size: 1.12rem; margin: 1.8em 0 0.5em; }
h4 { font-size: 1rem; }
.kicker { font-family: var(--sans); font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin: 0; }
.lede { font-size: 1.15rem; }
.caption { font-family: var(--sans); font-size: 14px; color: var(--muted); }
code, pre { font-family: var(--mono); font-size: 0.84em; }
code { background: var(--code-bg); padding: 0.1em 0.3em; border-radius: 3px; }
pre { background: var(--code-bg); padding: 14px 16px; border-radius: 4px; overflow-x: auto; line-height: 1.5; }
pre code { background: none; padding: 0; font-size: 0.95em; }
var { font-style: italic; }
.table-wrap { overflow-x: auto; margin: 1em 0 1.4em; border-top: 1px solid var(--rule); }
table { border-collapse: collapse; font-family: var(--sans); font-size: 14px; font-variant-numeric: tabular-nums; min-width: 100%; }
th, td { text-align: left; padding: 7px 12px 7px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: 13px; }
td:not(:first-child) { white-space: nowrap; }
.doc td:not(:first-child) { white-space: normal; min-width: 5.5em; }
.doc td:first-child { min-width: 11em; }
.figures { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 16px; margin: 1em 0; }
.figures > div { background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; padding: 14px 16px; min-width: 0; }
.figures dt { font-family: var(--sans); font-size: 13px; color: var(--muted); }
.figures dd { margin: 4px 0; font-family: var(--sans); font-size: 1.8rem; font-weight: 600; font-variant-numeric: tabular-nums; }
.figures p { margin: 0; font-family: var(--sans); font-size: 14px; color: var(--muted); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 22px; font-family: var(--sans); font-size: 14px; margin: 1.2em 0 0.4em; }
.legend span { display: inline-flex; align-items: center; gap: 8px; }
.legend .dot { position: static; transform: none; display: inline-block; }
.dotplot { font-family: var(--sans); font-size: 14px; font-variant-numeric: tabular-nums; margin-bottom: 0.8em; }
.dotplot .row { display: grid; grid-template-columns: 15rem 1fr; gap: 12px; align-items: center; min-height: 30px; }
.dotplot .lab { min-width: 0; }
.dotplot .k { color: var(--muted); white-space: nowrap; }
.dotplot .vals { display: block; font-size: 12px; color: var(--muted); }
.dotplot .vals .plain { color: var(--plain); } .dotplot .vals .penalty { color: var(--penalty); }
.dotplot .track { position: relative; height: 26px; }
.dotplot .zero { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--zero); }
.dotplot .bar { position: absolute; top: 50%; height: 2px; margin-top: -1px; opacity: 0.55; }
.dotplot .bar.plain { background: var(--plain); } .dotplot .bar.penalty { background: var(--penalty); }
.dot { position: absolute; top: 50%; width: 11px; height: 11px; transform: translate(-50%, -50%); border-radius: 50%; }
.dot.plain { background: var(--plain); }
.dot.penalty { background: var(--bg); border: 2.5px solid var(--penalty); }
.dotplot .axis { min-height: 22px; }
.dotplot .ticks { height: 22px; border-top: 1px solid var(--rule); }
.dotplot .ticks span { position: absolute; top: 3px; transform: translateX(-50%); font-size: 12px; color: var(--muted); }
.links li { margin-bottom: 0.4em; }
footer.site { border-top: 1px solid var(--rule); font-family: var(--sans); font-size: 13px; color: var(--muted); }
footer.site .wrap { padding-block: 20px 32px; }
/* demo */
.compare { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px 32px; margin: 1.2em 0; }
.compare h3 { margin: 0 0 0.3em; display: flex; align-items: center; gap: 8px; }
.compare ul { padding-left: 1.1em; margin: 0; font-size: 16px; }
.key { display: inline-block; width: 12px; height: 12px; border-radius: 50%; flex: none; }
.key.lloyd { background: var(--lloyd); } .key.emeans { background: var(--plain); }
.controls { display: grid; grid-template-columns: repeat(auto-fill, minmax(230px, 1fr)); gap: 12px 24px; padding: 16px 18px;
  background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; font-family: var(--sans); font-size: 14px; margin: 1.5em 0; }
.controls label { display: flex; flex-direction: column; gap: 4px; min-width: 0; }
.controls label.check { flex-direction: row; align-items: center; gap: 8px; }
.controls output { font-variant-numeric: tabular-nums; color: var(--muted); }
.controls label:not(.check) { display: grid; grid-template-columns: 1fr auto; }
.controls label:not(.check) > input, .controls label:not(.check) > select { grid-column: 1 / -1; }
.controls input[type=range] { width: 100%; accent-color: var(--accent); }
.controls input:disabled { opacity: 0.45; }
.controls select { font: inherit; padding: 4px 6px; background: var(--bg); color: var(--fg); border: 1px solid var(--rule); border-radius: 3px; }
.buttons { grid-column: 1 / -1; display: flex; flex-wrap: wrap; gap: 8px; }
.buttons button { font: 500 14px var(--sans); padding: 7px 14px; border-radius: 3px; border: 1px solid var(--rule); background: var(--bg); color: var(--fg); cursor: pointer; }
.buttons button:first-child { background: var(--accent); border-color: var(--accent); color: var(--surface); min-width: 6em; }
.panels { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px; }
.panel, .chartfig { margin: 0; min-width: 0; }
.panel figcaption, .chartfig figcaption { font-family: var(--sans); font-size: 15px; font-weight: 600; display: flex; align-items: center; gap: 8px; margin-bottom: 6px; }
.chartfig figcaption { font-weight: 400; font-size: 14px; color: var(--muted); display: block; max-width: 80ch; }
.panel canvas { width: 100%; aspect-ratio: 1; display: block; border: 1px solid var(--rule); border-radius: 4px; }
.chartfig { margin-top: 24px; }
.chartfig canvas { width: 100%; height: 300px; display: block; border: 1px solid var(--rule); border-radius: 4px; }
.stats { font-family: var(--sans); font-size: 13px; color: var(--muted); font-variant-numeric: tabular-nums; margin: 6px 0 0; min-height: 2.6em; }
.chartgrid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 28px 32px; margin-top: 1em; }
.smallchart { margin: 1.5em 0 0; min-width: 0; }
.chartgrid .smallchart { margin: 0; }
section > .smallchart { max-width: 560px; }
.smallchart h3 { margin: 0 0 0.3em; font-size: 1.05rem; }
.smallchart > p { font-size: 15px; margin: 0 0 0.5em; }
.smallchart figcaption { font-family: var(--sans); font-size: 13px; color: var(--muted); margin-top: 4px; }
.smallchart .legend { margin: 0.4em 0; font-size: 13px; gap: 4px 16px; }
.sw { display: inline-block; width: 18px; height: 0; border-top: 2.5px solid; }
.sw.dash { border-top-style: dashed; }
.smallchart .legend { flex-direction: column; gap: 3px; }
.legend .lg { display: grid; grid-template-columns: 8.5em 1fr; align-items: baseline; }
.legend .lg b { font-weight: 600; }
.legend .lg .items { display: flex; flex-wrap: wrap; gap: 2px 14px; min-width: 0; }
rect.pale { opacity: 0.45; }
.rolekey { display: grid; gap: 6px; font-family: var(--sans); font-size: 14px; margin-top: 1.2em; max-width: 72ch; }
.rolekey span { display: flex; align-items: baseline; gap: 8px; }
.rolekey .sws { flex: none; width: 42px; display: flex; gap: 4px; transform: translateY(-3px); }
.rolekey .sws .sw { width: 18px; }
.rolekey b { flex: none; min-width: 7.5em; }
.panel figcaption .sub { font-weight: 400; color: var(--muted); margin-left: 4px; }
svg.chart { width: 100%; height: auto; display: block; font-family: var(--sans); background: var(--surface); border: 1px solid var(--rule); border-radius: 4px; }
svg.chart .grid { stroke: var(--rule); stroke-width: 1; }
svg.chart .frame { stroke: var(--muted); stroke-width: 1; }
svg.chart .ref { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 3; }
svg.chart text { fill: var(--muted); font-size: 12px; }
svg.chart .axis { font-size: 12px; }
svg.chart .rowlabel { fill: var(--fg); font-size: 12px; }
svg.chart .clip { fill: var(--fg); font-size: 11px; }
svg.chart polyline { fill: none; stroke-width: 2.2; }
svg.chart .dash { stroke-dasharray: 6 4; }
svg.chart .m { stroke: var(--surface); stroke-width: 1.5; }
.s-full { stroke: var(--lloyd); fill: var(--lloyd); color: var(--lloyd); }
.s-recipe { stroke: var(--plain); fill: var(--plain); color: var(--plain); }
.s-alt { stroke: var(--penalty); fill: var(--penalty); color: var(--penalty); }
.s-alt2 { stroke: var(--zero); fill: var(--zero); color: var(--zero); }
polyline.s-full, polyline.s-recipe, polyline.s-alt, polyline.s-alt2 { fill: none; }
svg.chart rect.bar { stroke: none; }
@media (max-width: 640px) {
  .chartgrid { grid-template-columns: 1fr; }
  .compare, .panels { grid-template-columns: 1fr; }
  .chartfig canvas { height: 260px; }
  body { font-size: 17px; }
  .wrap { padding-inline: 16px; }
  h1 { font-size: 1.7rem; }
  .dotplot .row { grid-template-columns: 1fr; gap: 0; margin-bottom: 8px; }
  .figures { grid-template-columns: 1fr; }
  .dotplot .axis .lab { display: none; }
}
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, "style.css"), "w").write(CSS)
    open(os.path.join(OUT, ".nojekyll"), "w").write("")
    open(os.path.join(OUT, "index.html"), "w").write(page(
        "e-means reproduction", "An independent reproduction of e-means (arXiv 2611.09501), a scikit-learn estimator, and follow-up experiments.",
        "index.html", INDEX.replace("{chart}", chart_html()).replace("{repo}", REPO)))
    open(os.path.join(OUT, "reproduction.html"), "w").write(page(
        "e-means: reproduction notes", "What the e-means paper claims and what an independent re-implementation measured.", "reproduction.html", render("README.md"), wide=True))
    open(os.path.join(OUT, "extensions.html"), "w").write(page(
        "e-means: follow-up experiments", "Follow-up experiments on the e-means recipe, and a recheck against the authors' code.", "extensions.html", render("EXTENSIONS.md"), wide=True))
    open(os.path.join(OUT, "demo.html"), "w").write(page(
        "Lloyd vs e-means demo", "Lloyd's k-means and e-means run side by side on the same 2-D points, with sliders.", "demo.html",
        open(os.path.join(ROOT, "site", "demo.html"), encoding="utf-8").read(), wide=True, scripts=("demo-core.js", "demo.js")))
    body = open(os.path.join(ROOT, "site", "followups.html"), encoding="utf-8").read()
    for key, fig in charts.all_charts().items():
        body = body.replace("{" + key + "}", fig)
    open(os.path.join(OUT, "followups.html"), "w").write(page(
        "e-means follow-ups", "The e-means recipe applied to mixtures, NMF, topic models, hidden Markov models, ALS and PageRank, with an animated mixture model.",
        "followups.html", body, wide=True, scripts=("demo-core.js", "gmm-core.js", "gmm-demo.js")))
    for f in ("demo-core.js", "demo.js", "gmm-core.js", "gmm-demo.js"):
        shutil.copy(os.path.join(ROOT, "site", f), os.path.join(OUT, f))
    print("wrote", ", ".join(sorted(os.listdir(OUT))))


if __name__ == "__main__":
    main()
