"""Paired statistical tests on the faithfulness results.
Friedman (omnibus, 4 methods) + pairwise Wilcoxon signed-rank + paired mean difference."""
import os, sys, json, itertools
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

ROOT = r"."
raw = pd.read_csv(rf"{ROOT}\results\faithfulness_subset_raw.csv")
methods = ["raw_attention", "attention_rollout", "integrated_gradients", "chefer_relevance"]

out = {}
for metric, better in [("comprehensiveness", "↑"), ("sufficiency", "↓")]:
    piv = raw.pivot(index="idx", columns="method", values=metric)[methods].dropna()
    n = len(piv)
    # Friedman omnibus
    fr = friedmanchisquare(*[piv[m] for m in methods])
    print(f"\n===== {metric} ({better} better) | n={n} =====")
    print(f"Friedman omnibus: chi2={fr.statistic:.3f}, p={fr.pvalue:.4f}")
    out[metric] = {"n": int(n), "friedman_chi2": float(fr.statistic),
                   "friedman_p": float(fr.pvalue),
                   "means": {m: float(piv[m].mean()) for m in methods}, "pairs": {}}
    print(f"{'pair':45s} {'Δmean':>8s} {'W':>8s} {'p':>8s}  significant?")
    for a, b in itertools.combinations(methods, 2):
        d = piv[a] - piv[b]
        mean_d = float(d.mean())
        try:
            w = wilcoxon(piv[a], piv[b])
            W, p = float(w.statistic), float(w.pvalue)
        except ValueError:
            W, p = float("nan"), float("nan")
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{a+' - '+b:45s} {mean_d:+8.4f} {W:8.1f} {p:8.4f}  {sig}")
        out[metric]["pairs"][f"{a}__vs__{b}"] = {"mean_diff": mean_d, "W": W, "p": p, "sig": sig}

json.dump(out, open(rf"{ROOT}\results\faithfulness_stats.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\nSaved: results/faithfulness_stats.json")
