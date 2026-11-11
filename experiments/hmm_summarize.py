"""Usage: python hmm_summarize.py <log file> [more log files]. Markdown tables from hmm_run.py logs."""
import sys, json, collections, numpy as np
rows = [json.loads(l) for p in sys.argv[1:] for l in open(p) if l.strip()]
def ms(v, d=4): return f"{np.mean(v):.{d}f}" if len(v) else "–"
def pm(v): return f"{np.mean(v):+.4f} ± {np.std(v):.4f}" if len(v) else "–"
for ds, K in sorted({(r["ds"], r["K"]) for r in rows}):
    rs = [r for r in rows if r["ds"] == ds and r["K"] == K]; n = rs[0]["n"]
    bw = {}
    for r in rs:
        if r["m"] == "bw": bw[(r["seed"], r["it"])] = r
    seeds = sorted({s for s, _ in bw}); curve = {s: [bw[(s, i)]["bits"] for i in range(1, 1 + max(i for t, i in bw if t == s))] for s in seeds}
    print(f"\n### {ds}, K = {K}, {n:,} training rows, seeds {seeds}\n")
    for r in rs:
        if r["m"] == "reference": print("Reference, held-out bits/symbol: unigram %.4f, order-1 Markov %.4f%s\n" % (r["unigram"], r["bigram"], ", true model %.4f" % r["truth"] if r.get("truth") else ""))
    def match(seed, b): return next((i + 1 for i, v in enumerate(curve.get(seed, [])) if v <= b), None)
    if seeds:
        top = min(len(c) for c in curve.values())
        print("| method | passes | train s | held-out bits | Baum-Welch passes to match (per seed) |\n|---|---|---|---|---|")
        for it in (1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 75, 100, 150):
            if it <= top: print(f"| Baum-Welch | {it} | {ms([bw[(s, it)]['time'] for s in seeds], 1)} | {ms([curve[s][it-1] for s in seeds])} | – |")
        if top not in (1, 2, 3, 5, 10, 20, 30, 40, 50, 60, 75, 100, 150): print(f"| Baum-Welch | {top} | {ms([bw[(s, top)]['time'] for s in seeds], 1)} | {ms([curve[s][top-1] for s in seeds])} | – |")
        for m, label in (("online", "online, formula with n = rows"), ("online_sym", "online, formula with n = symbols")):
          for T in (1, 2, 3, 5, 10):
            d = sorted([r for r in rs if r["m"] == m and r["T"] == T], key=lambda r: r["seed"])
            if d: print(f"| {label} (B={d[0]['B']} rows, step {d[0]['a0']}) | {T} | {ms([r['time'] for r in d], 1)} | {ms([r['bits'] for r in d])} | {[match(r['seed'], r['bits']) or ('>%d' % len(curve.get(r['seed'], []))) for r in d]} |")
    ab = [r for r in rs if r["m"] == "ablate"]
    if ab:
        base = {r["seed"]: r for r in ab if r["variant"] == "default"}
        print("\n| 3-pass variant | held-out bits | vs default (negative = better) | train s | revived | seeds |\n|---|---|---|---|---|---|")
        for v in dict.fromkeys(r["variant"] for r in ab):
            d = [r for r in ab if r["variant"] == v and r["seed"] in base]
            print(f"| {v} | {ms([r['bits'] for r in d])} | {pm([r['bits'] - base[r['seed']]['bits'] for r in d])} | {ms([r['time'] for r in d], 1)} | {ms([r['revived'] for r in d], 0)} | {len(d)} |")
    def base3(seed, sym=False):
        d = [r for r in rs if r["seed"] == seed and ((r["m"] == "ablate" and r["variant"] == ("n = symbols" if sym else "default")) or (r["m"] == ("online_sym" if sym else "online") and r["T"] == 3))]
        return d[0]["bits"] if d else None
    def vs_sym(d): return pm([r["bits"] - base3(r["seed"], True) for r in d if base3(r["seed"], True) is not None])
    tb = [r for r in rs if r["m"] == "textbook" and base3(r["seed"]) is not None]
    if tb:
        print("\n| textbook step (t+2)^-kappa, 3 passes | batch rows | revival | held-out bits | vs recipe, n = rows (negative = better) | vs recipe, n = symbols | train s | revived | seeds |\n|---|---|---|---|---|---|---|---|---|")
        for key in dict.fromkeys((r["bdiv"], r["kappa"], r["revive"]) for r in tb):
            d = [r for r in tb if (r["bdiv"], r["kappa"], r["revive"]) == key]
            print(f"| kappa {key[1]} | {d[0]['B']} | {'yes' if key[2] else 'no'} | {ms([r['bits'] for r in d])} | {pm([r['bits'] - base3(r['seed']) for r in d])} | {vs_sym(d)} | {ms([r['time'] for r in d], 1)} | {ms([r['revived'] for r in d], 0)} | {len(d)} |")
    gr = [r for r in rs if r["m"] == "grid" and base3(r["seed"]) is not None]
    if gr:
        law = next(r for r in rs if r["m"] == "law")
        print(f"\nSettings grid, 3 passes (formula: B = {law['B']}, step = {law['a0']}); change in held-out bits vs the formula's run, negative = better\n")
        print("| batch rows | peak step | held-out bits | vs formula | train s | seeds |\n|---|---|---|---|---|---|")
        for key in dict.fromkeys((r["B"], r["a0"]) for r in gr):
            d = [r for r in gr if (r["B"], r["a0"]) == key]
            print(f"| {key[0]} | {key[1]} | {ms([r['bits'] for r in d])} | {pm([r['bits'] - base3(r['seed']) for r in d])} | {ms([r['time'] for r in d], 1)} | {len(d)} |")
    gw = [r for r in rs if r["m"] == "grow"]
    if gw:
        print("\n| state splitting | passes | held-out bits | train s | seeds |\n|---|---|---|---|---|")
        for key in dict.fromkeys((r["variant"], r["T"]) for r in gw):
            d = [r for r in gw if (r["variant"], r["T"]) == key]; print(f"| {key[0]} | {key[1]} | {ms([r['bits'] for r in d])} | {ms([r['time'] for r in d], 1)} | {len(d)} |")
    for mode in ("buffer", "short"):
        b = [r for r in rs if r["m"] == mode + "_bw"]
        if b:
            print(f"\n| rows of {b[0]['rows']} symbols + {b[0]['buf']} of context each side | passes | train s | held-out bits | seeds |\n|---|---|---|---|---|")
            for it in (10, 20, 40): print(f"| Baum-Welch | {it} | {ms([r['time'] * it / 40 for r in b], 1)} | {ms([r['curve'][it-1] for r in b])} | {len(b)} |")
            for T in (1, 2, 3, 5):
                d = [r for r in rs if r["m"] == mode + "_online" and r["T"] == T]
                if d: print(f"| online (B={d[0]['B']}, step {d[0]['a0']}) | {T} | {ms([r['time'] for r in d], 1)} | {ms([r['bits'] for r in d])} | {len(d)} |")
