"""Final statistics: integrate the reps-random baseline; Friedman chi2 + full
pairwise (4 families) + stratum + mean±std. Writes results/final_stats.json."""
import os, json, itertools
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata

ROOT = r"."
raw = pd.read_csv(rf"{ROOT}\results\faithfulness_rev_raw.csv")
reps = pd.read_csv(rf"{ROOT}\results\random_reps.csv")
methods = ["raw_attention","attention_rollout","integrated_gradients","chefer_relevance","random"]
short = {"raw_attention":"Raw attention","attention_rollout":"Attention rollout",
         "integrated_gradients":"Integrated Gradients","chefer_relevance":"Chefer","random":"Random"}
idx2strat = raw[raw.method=="raw_attention"].set_index("idx")["stratum"].to_dict()
rng = np.random.default_rng(3)

# --- reshape reps-random to long format and merge with the non-random results ---
rr = []
for _, r in reps.iterrows():
    rr.append({"idx": r["idx"], "stratum": idx2strat[r["idx"]], "method": "random",
               "operator": "mask", "comprehensiveness": r["comp_mask"], "sufficiency": r["suff_mask"]})
    rr.append({"idx": r["idx"], "stratum": idx2strat[r["idx"]], "method": "random",
               "operator": "delete", "comprehensiveness": r["comp_delete"], "sufficiency": r["suff_delete"]})
data = pd.concat([raw[raw.method != "random"], pd.DataFrame(rr)], ignore_index=True)

def boot(x, n=10000):
    x=np.asarray(x); i=rng.integers(0,len(x),(n,len(x))); return np.percentile(x[i].mean(1),[2.5,97.5])
def rb(a,b):
    d=np.asarray(a)-np.asarray(b); d=d[d!=0]
    if len(d)==0: return 0.0
    r=rankdata(np.abs(d)); return float((r[d>0].sum()-r[d<0].sum())/(r[d>0].sum()+r[d<0].sum()))
def holm(ps):
    o=sorted(range(len(ps)),key=lambda i:ps[i]); m=len(ps); adj=[None]*m; pv=0
    for k,i in enumerate(o): v=min(1,(m-k)*ps[i]); v=max(v,pv); pv=v; adj[i]=v
    return adj

out = {"means": {}, "friedman": {}, "pairwise": {}, "stratum_random": {}}
for metric in ["comprehensiveness","sufficiency"]:
    for op in ["mask","delete"]:
        piv = data[data.operator==op].pivot(index="idx",columns="method",values=metric)[methods].dropna()
        key = f"{metric}_{op}"
        out["means"][key] = {short[m]: [round(float(piv[m].mean()),3), round(float(piv[m].std()),3)] for m in methods}
        fr = friedmanchisquare(*[piv[m] for m in methods])
        out["friedman"][key] = {"chi2": round(float(fr.statistic),2), "p": float(fr.pvalue), "df": len(methods)-1}
        pairs=list(itertools.combinations(methods,2)); rows=[]; ps=[]
        for a,b in pairs:
            d=(piv[a]-piv[b]).values; lo,hi=boot(d); p=float(wilcoxon(piv[a],piv[b]).pvalue)
            ps.append(p); rows.append([f"{short[a]} - {short[b]}", round(float(d.mean()),3), round(lo,3), round(hi,3), round(rb(piv[a],piv[b]),2)])
        adj=holm(ps)
        out["pairwise"][key] = [{"pair":r[0],"delta":r[1],"ci":[r[2],r[3]],"p_adj":round(pa,4),"r":r[4]} for r,pa in zip(rows,adj)]

# --- stratum comprehensiveness (mask) with reps-random ---
piv2 = data[data.operator=="mask"].pivot_table(index="stratum",columns="method",values="comprehensiveness",aggfunc="mean")[methods]
out["stratum_comp_mask"] = {st: {short[m]: round(float(piv2.loc[st,m]),3) for m in methods} for st in piv2.index}

json.dump(out, open(rf"{ROOT}\results\final_stats.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
# print summary
print("=== MEANS (reps-random) ===")
for k,v in out["means"].items(): print(k, {m:v[m][0] for m in v})
print("\n=== FRIEDMAN ===")
for k,v in out["friedman"].items(): print(k, f"chi2({v['df']})={v['chi2']}, p={v['p']:.2g}")
print("\n=== rollout vs random (key claim) ===")
for op in ["mask","delete"]:
    for row in out["pairwise"][f"comprehensiveness_{op}"]:
        if row["pair"]=="Attention rollout - Random": print(f"  comp_{op}:", row)
print("\n=== stratum comp (mask) ===")
for st,v in out["stratum_comp_mask"].items(): print(f"  {st}: {v}")
print("\nSaved: results/final_stats.json")
