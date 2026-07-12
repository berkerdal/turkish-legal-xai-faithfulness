"""AOPC comprehensiveness curve: reuses the saved attributions and runs only the
[MASK] perturbation passes. Writes results/aopc_curve.json."""
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"; os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, r".\src")
import numpy as np, pandas as pd
from xai.explainer import load_model_for_xai, Explainer

ROOT = r"."
BINS = [0.01, 0.05, 0.10, 0.20, 0.50]
methods = ["raw_attention", "attention_rollout", "integrated_gradients", "chefer_relevance", "random"]
model, tok = load_model_for_xai(rf"{ROOT}\models", num_labels=2)
exp = Explainer(model, tok, device="cpu", max_len=512)
df = pd.read_csv(rf"{ROOT}\results\test_predictions.csv").set_index("idx")
data = json.load(open(rf"{ROOT}\results\attributions_rev.json", encoding="utf-8"))
mask_id = tok.mask_token_id
special = {"[CLS]", "[SEP]", "[PAD]"}

acc = {m: {k: [] for k in BINS} for m in methods}
t0 = time.time()
for n, (idx, rec) in enumerate(data.items(), 1):
    text = str(df.loc[int(idx), "text"]); pred = rec["pred"]
    enc = exp._encode(text); input_ids = enc["input_ids"]; attn = enc["attention_mask"]
    toks = rec["tokens"]
    p_orig = exp.prob_from_ids(input_ids, attn, pred)
    content = np.array([i for i, t in enumerate(toks) if t not in special])
    for m in methods:
        s = np.array(rec["scores"][m], float)
        ranked = content[np.argsort(-s[content])]
        nc = len(ranked)
        for k in BINS:
            mm = max(1, int(round(k * nc)))
            ids = input_ids.clone(); ids[0, ranked[:mm]] = mask_id
            acc[m][k].append(p_orig - exp.prob_from_ids(ids, attn, pred))
    if n % 10 == 0:
        print(f"  {n}/{len(data)} | {time.time()-t0:.0f}s")

curve = {m: {str(k): round(float(np.mean(v)), 4) for k, v in acc[m].items()} for m in methods}
json.dump(curve, open(rf"{ROOT}\results\aopc_curve.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("Saved: results/aopc_curve.json"); print(json.dumps(curve, indent=2))
