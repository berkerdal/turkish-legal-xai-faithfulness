"""Revizyon istatistikleri: iki operatör × Friedman + Holm-düzeltmeli Wilcoxon,
katman kırılımı, ve geniş örneklemde RQ2 token-tipi."""
import os, sys, json, re, itertools
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

ROOT = r"."
raw = pd.read_csv(rf"{ROOT}\results\faithfulness_rev_raw.csv")
methods = ["raw_attention", "attention_rollout", "integrated_gradients", "chefer_relevance", "random"]
short = {"raw_attention":"raw","attention_rollout":"rollout","integrated_gradients":"IG",
         "chefer_relevance":"chefer","random":"random"}

def holm(pairs):  # pairs: list of (label, p) -> adjusted
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][1])
    m = len(pairs); adj = [None]*m; prev = 0
    for rank, i in enumerate(order):
        a = min(1.0, (m-rank)*pairs[i][1]); a = max(a, prev); prev = a
        adj[i] = a
    return adj

out = {}
print(f"Örneklem n={raw.idx.nunique()} | katmanlar: {raw.groupby('stratum').idx.nunique().to_dict()}")
for op in ["mask", "delete"]:
    out[op] = {}
    sub = raw[raw.operator == op]
    for metric in ["comprehensiveness", "sufficiency"]:
        piv = sub.pivot(index="idx", columns="method", values=metric)[methods].dropna()
        n = len(piv)
        fr = friedmanchisquare(*[piv[m] for m in methods])
        means = {short[m]: round(float(piv[m].mean()),4) for m in methods}
        print(f"\n=== operator={op} | {metric} | n={n} ===")
        print("ort:", means, f"| Friedman p={fr.pvalue:.3g}")
        pairs = []
        for a,b in itertools.combinations(methods, 2):
            try: p = float(wilcoxon(piv[a], piv[b]).pvalue)
            except ValueError: p = 1.0
            pairs.append((f"{short[a]}-{short[b]}", p, float((piv[a]-piv[b]).mean())))
        adj = holm([(lbl,p) for lbl,p,_ in pairs])
        out[op][metric] = {"n":n,"friedman_p":float(fr.pvalue),"means":means,"pairs":{}}
        for (lbl,p,d),pa in zip(pairs,adj):
            sig = "***" if pa<.001 else "**" if pa<.01 else "*" if pa<.05 else "ns"
            tag = " <vs random" if lbl.endswith("random") else ""
            print(f"  {lbl:20s} Δ={d:+.4f}  p_holm={pa:.4f} {sig}{tag}")
            out[op][metric]["pairs"][lbl] = {"delta":d,"p_holm":pa,"sig":sig}

# --- Katman kırılımı (mask, comprehensiveness) — sıralama sağlamlığı ---
print("\n=== Katman kırılımı: comprehensiveness ort. (operator=mask) ===")
piv2 = raw[raw.operator=="mask"].pivot_table(index="stratum", columns="method",
        values="comprehensiveness", aggfunc="mean")[methods]
piv2.columns = [short[m] for m in methods]
print(piv2.round(3).to_string())
out["stratum_comp_mask"] = piv2.round(4).to_dict()

# --- RQ2 geniş örneklem ---
data = json.load(open(rf"{ROOT}\results\attributions_rev.json", encoding="utf-8"))
TR_STOP=set("ve bir bu ile için olarak da de ki mi mı mu mü ya veya ama ise ne gibi kadar daha en çok az her hiç şey o şu".split())
PUNCT=set(".,;:!?()[]{}\"'`«»—–-…/\\|"); types=["content","punctuation","subword","stopword","number"]
def ttype(t):
    if t in ("[CLS]","[SEP]","[PAD]"): return "special"
    if t.startswith("##"): return "subword"
    if all(c in PUNCT for c in t): return "punctuation"
    if re.fullmatch(r"\d+([.,]\d+)?",t): return "number"
    if t.lower() in TR_STOP: return "stopword"
    return "content"
pm={m:{ty:[] for ty in types} for m in methods}
for idx,rec in data.items():
    tt=[ttype(t) for t in rec["tokens"]]
    for m in methods:
        s=np.array(rec["scores"][m],float); s[[i for i,x in enumerate(tt) if x=="special"]]=0
        if s.sum()<=0: continue
        for ty in types: pm[m][ty].append(s[[i for i,x in enumerate(tt) if x==ty]].sum()/s.sum())
rq2=pd.DataFrame({short[m]:{ty:round(100*np.mean(pm[m][ty]),1) for ty in types} for m in methods}).T[types]
print("\n=== RQ2 (geniş örneklem, n=%d): önem kütlesi %% ===" % len(data))
print(rq2.to_string())
out["rq2"]=rq2.to_dict()

json.dump(out, open(rf"{ROOT}\results\faithfulness_rev_stats.json","w",encoding="utf-8"), ensure_ascii=False, indent=2)
rq2.to_csv(rf"{ROOT}\results\rq2_rev.csv")
print("\nKaydedildi: faithfulness_rev_stats.json + rq2_rev.csv")
