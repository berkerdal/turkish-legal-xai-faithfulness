"""Token-type attribution analysis (RQ2). For each method, how much of the total
importance mass falls on each token type (punctuation / subword / stopword / number /
content)? Reads attributions_pool.json; no model needed."""
import os, json, re
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np, pandas as pd

ROOT = r"."
data = json.load(open(rf"{ROOT}\results\attributions_pool.json", encoding="utf-8"))

TR_STOP = set("""ve bir bu ile için olarak da de ki mi mı mu mü ya veya ancak ama çünkü
ise ne gibi kadar daha en çok az her hiç şey o bu şu ben sen biz siz onlar ki nin nın
nun nün ının inin dır dir tir ilgili sonra önce üzere göre""".split())
PUNCT = set(".,;:!?()[]{}\"'`«»—–-…/\\|")

def token_type(t):
    if t in ("[CLS]", "[SEP]", "[PAD]"):
        return "special"
    if t.startswith("##"):
        return "subword"
    if all(c in PUNCT for c in t):
        return "punctuation"
    if re.fullmatch(r"\d+([.,]\d+)?", t):
        return "number"
    if t.lower() in TR_STOP:
        return "stopword"
    return "content"

methods = ["raw_attention", "attention_rollout", "integrated_gradients", "chefer_relevance", "random"]
types = ["content", "punctuation", "subword", "stopword", "number"]

# per instance: method x type mass share, then averaged over instances
per_method = {m: {ty: [] for ty in types} for m in methods}
for idx, rec in data.items():
    toks = rec["tokens"]
    tt = [token_type(t) for t in toks]
    for m in methods:
        s = np.array(rec["scores"][m], dtype=float)
        s[[i for i, x in enumerate(tt) if x == "special"]] = 0.0  # drop special tokens
        tot = s.sum()
        if tot <= 0:
            continue
        for ty in types:
            mass = s[[i for i, x in enumerate(tt) if x == ty]].sum()
            per_method[m][ty].append(mass / tot)

tbl = pd.DataFrame({m: {ty: round(100 * np.mean(per_method[m][ty]), 1) for ty in types}
                    for m in methods}).T[types]
print("=== RQ2: importance mass by token type, per method (%) ===")
print("(special tokens excluded; rows sum to ~100)\n")
print(tbl.to_string())

# summary: content vs surface (punctuation + subword + stopword)
tbl["surface(punct+subword+stop)"] = (tbl["punctuation"] + tbl["subword"] + tbl["stopword"]).round(1)
print("\n--- content vs surface ---")
print(tbl[["content", "surface(punct+subword+stop)"]].to_string())

tbl.to_csv(rf"{ROOT}\results\rq2_token_type_shares.csv")
print("\nSaved: results/rq2_token_type_shares.csv")
