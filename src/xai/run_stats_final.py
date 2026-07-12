"""Final statistics: 5 methods (4 + random). Friedman + pairwise Wilcoxon.
In particular, is each real method significantly better than random? (anti-faithfulness check)"""
import os, json, itertools
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

ROOT = r"."
raw = pd.read_csv(rf"{ROOT}\results\faithfulness_full_raw.csv")
methods = ["raw_attention", "attention_rollout", "integrated_gradients", "chefer_relevance", "random"]
out = {}
for metric, better in [("comprehensiveness", "higher better"), ("sufficiency", "lower better")]:
    piv = raw.pivot(index="idx", columns="method", values=metric)[methods].dropna()
    n = len(piv)
    fr = friedmanchisquare(*[piv[m] for m in methods])
    print(f"\n===== {metric} ({better}) | n={n} =====")
    print("means:", {m: round(float(piv[m].mean()), 4) for m in methods})
    print(f"Friedman: chi2={fr.statistic:.2f}, p={fr.pvalue:.4g}")
    out[metric] = {"n": int(n), "means": {m: float(piv[m].mean()) for m in methods},
                   "friedman_p": float(fr.pvalue), "pairs": {}}
    print(f"{'pair':42s} {'Δmean':>8s} {'p':>9s}  sig")
    for a, b in itertools.combinations(methods, 2):
        d = piv[a] - piv[b]
        try:
            p = float(wilcoxon(piv[a], piv[b]).pvalue)
        except ValueError:
            p = float("nan")
        sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
        star = "  <-- vs random" if b == "random" else ""
        print(f"{a+' - '+b:42s} {float(d.mean()):+8.4f} {p:9.4g}  {sig}{star}")
        out[metric]["pairs"][f"{a}__vs__{b}"] = {"mean_diff": float(d.mean()), "p": p, "sig": sig}

json.dump(out, open(rf"{ROOT}\results\faithfulness_stats_final.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\nSaved: results/faithfulness_stats_final.json")
