"""Revision run: stratified sample (high/low-confidence correct + incorrect) x two
operators (mask, delete).
Writes results/faithfulness_rev_raw.csv, results/attributions_rev.json."""
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"; os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, r".\src")
import numpy as np, pandas as pd
from xai.explainer import Explainer, load_model_for_xai, Attribution
from xai import faithfulness as F

ROOT = r"."
rng = np.random.default_rng(42)
model, tok = load_model_for_xai(rf"{ROOT}\models", num_labels=2)
exp = Explainer(model, tok, device="cpu", max_len=512)
df = pd.read_csv(rf"{ROOT}\results\test_predictions.csv")

# --- stratified sample (~20+20+20) ---
correct = df[df.correct].copy(); errors = df[~df.correct].copy()
hi = correct.sort_values("confidence", ascending=False).head(20).assign(stratum="high_confidence_correct")
lo = correct.sort_values("confidence", ascending=True).head(20).assign(stratum="low_confidence_correct")
# balance FP/FN among the errors
fp = errors[errors.pred == 1].head(10); fn = errors[errors.pred == 0].head(10)
err = pd.concat([fp, fn]).assign(stratum="incorrect")
sample = pd.concat([hi, lo, err]).drop_duplicates("idx").reset_index(drop=True)
print(f"Sample: {len(sample)} | strata: {sample.stratum.value_counts().to_dict()}")

methods = ["raw_attention", "attention_rollout", "integrated_gradients", "chefer_relevance"]
OPS = ["mask", "delete"]
rows, saved = [], {}
t0 = time.time()
for n, r in sample.iterrows():
    idx = int(r["idx"]); text = str(r["text"])
    atts = exp.all_methods(text)
    ref = next(iter(atts.values()))
    atts["random"] = Attribution("random", ref.tokens, rng.random(len(ref.tokens)), ref.pred, ref.prob)
    saved[str(idx)] = {"stratum": r["stratum"], "true": int(r["true"]), "pred": ref.pred,
                       "tokens": ref.tokens,
                       "scores": {m: [round(float(x), 5) for x in a.scores] for m, a in atts.items()}}
    for m, a in atts.items():
        for op in OPS:
            res = F.evaluate(exp, text, a, operator=op)
            rows.append({"idx": idx, "stratum": r["stratum"], "method": m, "operator": op,
                         "comprehensiveness": res["comprehensiveness"], "sufficiency": res["sufficiency"]})
    print(f"  {n+1}/{len(sample)} (idx={idx}, {r['stratum']}) | {time.time()-t0:.0f}s")

pd.DataFrame(rows).to_csv(rf"{ROOT}\results\faithfulness_rev_raw.csv", index=False)
json.dump(saved, open(rf"{ROOT}\results\attributions_rev.json", "w", encoding="utf-8"), ensure_ascii=False)
print("\nDone. Saved: faithfulness_rev_raw.csv + attributions_rev.json")
