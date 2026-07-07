"""RQ2 — token-tipi atıf analizi. Her yöntem, toplam önem kütlesinin ne kadarını
hangi token tipine (noktalama / alt-kelime / durak-kelime / sayı / anlamlı içerik) veriyor?
attributions_pool.json'dan; model gerektirmez."""
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

# her örnek için: yöntem×tip kütle payı; sonra örnekler üzerinde ortalama
per_method = {m: {ty: [] for ty in types} for m in methods}
for idx, rec in data.items():
    toks = rec["tokens"]
    tt = [token_type(t) for t in toks]
    for m in methods:
        s = np.array(rec["scores"][m], dtype=float)
        s[[i for i, x in enumerate(tt) if x == "special"]] = 0.0  # özel tokenları çıkar
        tot = s.sum()
        if tot <= 0:
            continue
        for ty in types:
            mass = s[[i for i, x in enumerate(tt) if x == ty]].sum()
            per_method[m][ty].append(mass / tot)

tbl = pd.DataFrame({m: {ty: round(100 * np.mean(per_method[m][ty]), 1) for ty in types}
                    for m in methods}).T[types]
print("=== RQ2: Yöntem başına önem kütlesinin token-tipine dağılımı (%) ===")
print("(özel tokenlar hariç; satır ~100'e toplanır)\n")
print(tbl.to_string())

# özet: içerik vs yüzeysel (noktalama+altkelime+durak) oranı
tbl["yuzeysel(punct+subword+stop)"] = (tbl["punctuation"] + tbl["subword"] + tbl["stopword"]).round(1)
print("\n--- İçerik vs Yüzeysel ---")
print(tbl[["content", "yuzeysel(punct+subword+stop)"]].to_string())

tbl.to_csv(rf"{ROOT}\results\rq2_token_type_shares.csv")
print("\nKaydedildi: results/rq2_token_type_shares.csv")
