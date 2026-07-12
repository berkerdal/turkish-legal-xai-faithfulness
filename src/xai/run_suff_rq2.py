"""Sufficiency pairwise (delete, primary separation) + RQ2 number-column check."""
import os, json, re, itertools
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, rankdata

ROOT = r"."
raw = pd.read_csv(rf"{ROOT}\results\faithfulness_rev_raw.csv")
methods = ["raw_attention","attention_rollout","integrated_gradients","chefer_relevance","random"]
short = {"raw_attention":"Raw attention","attention_rollout":"Attention rollout",
         "integrated_gradients":"Integrated Gradients","chefer_relevance":"Chefer","random":"Random"}
rng = np.random.default_rng(11)
def boot_ci(x,n=10000):
    x=np.asarray(x); idx=rng.integers(0,len(x),(n,len(x))); return np.percentile(x[idx].mean(1),[2.5,97.5])
def rb(a,b):
    d=np.asarray(a)-np.asarray(b); d=d[d!=0]
    if len(d)==0: return 0.0
    r=rankdata(np.abs(d)); return float((r[d>0].sum()-r[d<0].sum())/(r[d>0].sum()+r[d<0].sum()))
def holm(ps):
    o=sorted(range(len(ps)),key=lambda i:ps[i]); m=len(ps); adj=[None]*m; pv=0
    for k,i in enumerate(o): v=min(1,(m-k)*ps[i]); v=max(v,pv); pv=v; adj[i]=v
    return adj

# --- Sufficiency pairwise (delete operator) ---
op="delete"
piv=raw[raw.operator==op].pivot(index="idx",columns="method",values="sufficiency")[methods].dropna()
pairs=list(itertools.combinations(methods,2)); rows=[]; ps=[]
for a,b in pairs:
    d=(piv[a]-piv[b]).values; lo,hi=boot_ci(d); p=float(wilcoxon(piv[a],piv[b]).pvalue)
    ps.append(p); rows.append({"pair":f"{short[a]} − {short[b]}","delta":float(d.mean()),"lo":lo,"hi":hi,"p":p,"r":rb(piv[a],piv[b])})
adj=holm(ps)
print("=== Sufficiency pairwise (delete, n=60); LOWER sufficiency = more faithful ===")
print(f"{'Comparison':44s} {'Δ':>8s} {'95% CI':>18s} {'p_adj':>8s} {'r':>6s}")
res=[]
for row,pa in zip(rows,adj):
    sig="***" if pa<.001 else "**" if pa<.01 else "*" if pa<.05 else "ns"
    print(f"{row['pair']:44s} {row['delta']:+8.3f} [{row['lo']:+.3f},{row['hi']:+.3f}] {pa:8.4f} {row['r']:+6.2f} {sig}")
    res.append({**row,"p_adj":pa})
json.dump(res, open(rf"{ROOT}\results\suff_pairwise_delete.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)

# --- RQ2 with the number column (revision sample) ---
data=json.load(open(rf"{ROOT}\results\attributions_rev.json",encoding="utf-8"))
TR_STOP=set("ve bir bu ile için olarak da de ki mi mı mu mü ya veya ama ise ne gibi kadar daha en çok az her hiç şey o şu".split())
PUNCT=set(".,;:!?()[]{}\"'`«»—–-…/\\|"); types=["content","punctuation","subword","stopword","number"]
def tt(t):
    if t in ("[CLS]","[SEP]","[PAD]"): return "special"
    if t.startswith("##"): return "subword"
    if all(c in PUNCT for c in t): return "punctuation"
    if re.fullmatch(r"\d+([.,]\d+)?",t): return "number"
    if t.lower() in TR_STOP: return "stopword"
    return "content"
pm={m:{ty:[] for ty in types} for m in methods}
for idx,rec in data.items():
    tp=[tt(t) for t in rec["tokens"]]
    for m in methods:
        s=np.array(rec["scores"][m],float); s[[i for i,x in enumerate(tp) if x=="special"]]=0
        if s.sum()<=0: continue
        for ty in types: pm[m][ty].append(s[[i for i,x in enumerate(tp) if x==ty]].sum()/s.sum())
tbl=pd.DataFrame({short[m]:{ty:round(100*np.mean(pm[m][ty]),1) for ty in types} for m in methods}).T[types]
print("\n=== RQ2 (n=60) with number column ===")
print(tbl.to_string())
tbl.to_csv(rf"{ROOT}\results\rq2_rev_full.csv")
print("\nSaved: suff_pairwise_delete.json + rq2_rev_full.csv")
