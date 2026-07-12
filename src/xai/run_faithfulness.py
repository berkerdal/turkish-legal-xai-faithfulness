"""Runner: comprehensiveness/sufficiency for the four methods on a subset of the pool."""
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"; os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, r".\src")
import numpy as np, pandas as pd
from xai.explainer import Explainer, load_model_for_xai
from xai import faithfulness as F

ROOT = r"."
N = int(sys.argv[1]) if len(sys.argv) > 1 else 10

model, tok = load_model_for_xai(rf"{ROOT}\models", num_labels=2)
exp = Explainer(model, tok, device="cpu", max_len=512)
df = pd.read_csv(rf"{ROOT}\results\test_predictions.csv").set_index("idx")
pool = json.load(open(rf"{ROOT}\results\example_pool.json", encoding="utf-8"))[:N]

methods = ["raw_attention", "attention_rollout", "integrated_gradients", "chefer_relevance"]
rows = []
t0 = time.time()
for n, e in enumerate(pool, 1):
    text = str(df.loc[e["idx"], "text"])
    atts = exp.all_methods(text)
    for m in methods:
        r = F.evaluate(exp, text, atts[m])
        rows.append({"idx": e["idx"], "method": m,
                     "comprehensiveness": r["comprehensiveness"], "sufficiency": r["sufficiency"]})
    print(f"  {n}/{len(pool)} (idx={e['idx']}) done | elapsed {time.time()-t0:.0f}s")

res = pd.DataFrame(rows)
agg = res.groupby("method").agg(
    comp_mean=("comprehensiveness", "mean"), comp_std=("comprehensiveness", "std"),
    suff_mean=("sufficiency", "mean"), suff_std=("sufficiency", "std")).reindex(methods)

print("\n=== FAITHFULNESS (n=%d instances) ===" % len(pool))
print("comprehensiveness ↑ better | sufficiency ↓ better\n")
print(agg.round(4).to_string())

res.to_csv(rf"{ROOT}\results\faithfulness_subset_raw.csv", index=False)
agg.to_csv(rf"{ROOT}\results\faithfulness_subset_agg.csv")
print("\nSaved: results/faithfulness_subset_{raw,agg}.csv")
