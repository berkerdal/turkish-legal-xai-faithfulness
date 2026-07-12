"""Stabilise the random baseline with 10 independent rankings.
For each instance, comp/suff (mask & delete) averaged over 10 repetitions. Writes results/random_reps.csv."""
import os, sys, json, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"; os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, r".\src")
import numpy as np, pandas as pd
from xai.explainer import load_model_for_xai, Explainer, Attribution
from xai import faithfulness as F

ROOT = r"."; REPS = 10
model, tok = load_model_for_xai(rf"{ROOT}\models", num_labels=2)
exp = Explainer(model, tok, device="cpu", max_len=512)
df = pd.read_csv(rf"{ROOT}\results\test_predictions.csv").set_index("idx")
data = json.load(open(rf"{ROOT}\results\attributions_rev.json", encoding="utf-8"))
rows = []; t0 = time.time()
for n, (idx, rec) in enumerate(data.items(), 1):
    text = str(df.loc[int(idx), "text"]); tokens = rec["tokens"]; pred = rec["pred"]
    cm, cd, sm, sd = [], [], [], []
    for r in range(REPS):
        rng = np.random.default_rng(1000*int(idx) + r)
        a = Attribution("random", tokens, rng.random(len(tokens)), pred, 1.0)
        em = F.evaluate(exp, text, a, operator="mask"); ed = F.evaluate(exp, text, a, operator="delete")
        cm.append(em["comprehensiveness"]); sm.append(em["sufficiency"])
        cd.append(ed["comprehensiveness"]); sd.append(ed["sufficiency"])
    rows.append({"idx": int(idx), "comp_mask": np.mean(cm), "comp_delete": np.mean(cd),
                 "suff_mask": np.mean(sm), "suff_delete": np.mean(sd)})
    if n % 10 == 0: print(f"  {n}/{len(data)} | {time.time()-t0:.0f}s")
out = pd.DataFrame(rows)
out.to_csv(rf"{ROOT}\results\random_reps.csv", index=False)
print("\n=== Random baseline (mean over %d rankings) ===" % REPS)
for c in ["comp_mask", "comp_delete", "suff_mask", "suff_delete"]:
    print(f"  {c}: {out[c].mean():.4f} ± {out[c].std():.4f}")
print("Saved: results/random_reps.csv")
