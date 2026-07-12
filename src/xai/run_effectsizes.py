"""Effect sizes: mean±std, 95% bootstrap CI, Holm-adjusted p, rank-biserial effect
size. Reads faithfulness_rev_raw.csv; builds the pairwise comparison table for
comprehensiveness under [MASK]."""
import os, sys, json, itertools
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, rankdata

ROOT = r"."
raw = pd.read_csv(rf"{ROOT}\results\faithfulness_rev_raw.csv")
methods = ["raw_attention","attention_rollout","integrated_gradients","chefer_relevance","random"]
short = {"raw_attention":"Raw attention","attention_rollout":"Attention rollout",
         "integrated_gradients":"Integrated Gradients","chefer_relevance":"Chefer","random":"Random"}
rng = np.random.default_rng(7)

def boot_ci(x, n=10000):
    x = np.asarray(x); idx = rng.integers(0, len(x), (n, len(x)))
    bs = x[idx].mean(1); return np.percentile(bs, [2.5, 97.5])

def rank_biserial(a, b):
    d = np.asarray(a) - np.asarray(b); d = d[d != 0]
    if len(d) == 0: return 0.0
    r = rankdata(np.abs(d)); Wp = r[d>0].sum(); Wn = r[d<0].sum()
    return float((Wp - Wn)/(Wp + Wn))

def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i]); m=len(ps); adj=[None]*m; prev=0
    for rank,i in enumerate(order):
        v=min(1.0,(m-rank)*ps[i]); v=max(v,prev); prev=v; adj[i]=v
    return adj

out = {}
# --- per-method mean±std + 95% CI (each operator x metric) ---
summary = {}
for op in ["mask","delete"]:
    for metric in ["comprehensiveness","sufficiency"]:
        piv = raw[raw.operator==op].pivot(index="idx",columns="method",values=metric)[methods].dropna()
        summary[f"{metric}_{op}"] = {}
        for m in methods:
            lo,hi = boot_ci(piv[m].values)
            summary[f"{metric}_{op}"][short[m]] = {
                "mean":round(float(piv[m].mean()),3),"std":round(float(piv[m].std()),3),
                "ci":[round(lo,3),round(hi,3)]}
print("=== per-method mean±std + 95% CI ===")
for k,v in summary.items():
    print(f"\n[{k}]")
    for m,s in v.items(): print(f"  {m:22s} {s['mean']:.3f} ± {s['std']:.3f}  CI[{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]")
out["per_method"] = summary

# --- pairwise comparison table (comprehensiveness, mask) ---
op, metric = "mask", "comprehensiveness"
piv = raw[raw.operator==op].pivot(index="idx",columns="method",values=metric)[methods].dropna()
pairs = list(itertools.combinations(methods,2))
rows=[]; ps=[]
for a,b in pairs:
    d = (piv[a]-piv[b]).values
    lo,hi = boot_ci(d)
    p = float(wilcoxon(piv[a],piv[b]).pvalue)
    ps.append(p)
    rows.append({"pair":f"{short[a]} − {short[b]}","delta":float(d.mean()),
                 "ci_lo":lo,"ci_hi":hi,"p":p,"r":rank_biserial(piv[a],piv[b])})
adj = holm(ps)
print("\n=== pairwise comparison (comprehensiveness, [MASK]) ===")
print(f"{'Comparison':40s} {'Δ':>8s} {'95% CI':>18s} {'p_adj':>8s} {'r':>6s}")
tbl=[]
for row,pa in zip(rows,adj):
    row["p_adj"]=pa
    print(f"{row['pair']:40s} {row['delta']:+8.3f} [{row['ci_lo']:+.3f},{row['ci_hi']:+.3f}] {pa:8.4f} {row['r']:+6.2f}")
    tbl.append(row)
out["pairwise_comp_mask"]=tbl
json.dump(out, open(rf"{ROOT}\results\effectsizes.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
print("\nSaved: results/effectsizes.json")
