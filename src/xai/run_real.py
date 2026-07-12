"""Load the fine-tuned model, check that local predictions match the Kaggle ones,
and run the four explanation methods on one high-confidence instance."""
import os, sys, json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, r".\src")
import pandas as pd, numpy as np
from xai.explainer import Explainer, load_model_for_xai

ROOT = r"."
model, tok = load_model_for_xai(rf"{ROOT}\models", num_labels=2)
exp = Explainer(model, tok, device="cpu", max_len=512)

df = pd.read_csv(rf"{ROOT}\results\test_predictions.csv")
pool = json.load(open(rf"{ROOT}\results\example_pool.json", encoding="utf-8"))
print(f"Test rows: {len(df)} | pool: {len(pool)} instances")

# --- check: local prediction == Kaggle prediction (first 15 pool instances) ---
ok = 0
for e in pool[:15]:
    row = df[df.idx == e["idx"]].iloc[0]
    pred, probs, _ = exp.predict(str(row["text"]))
    if pred == int(row["pred"]) and abs(float(probs[1]) - float(row["prob_1"])) < 0.02:
        ok += 1
print(f"Local vs Kaggle agreement: {ok}/15 (pred matched & prob_1 diff <0.02)")

# --- first attribution: the highest-confidence correct instance ---
e = pool[0]
row = df[df.idx == e["idx"]].iloc[0]
text = str(row["text"])
print(f"\n=== instance idx={e['idx']} | true={int(row['true'])} pred={int(row['pred'])} "
      f"confidence={float(row['confidence']):.3f} | rights={row['Haklar']} ===")
print("Text (first 200 chars):", text[:200])

atts = exp.all_methods(text)
out = {"idx": int(e["idx"]), "true": int(row["true"]), "pred": int(row["pred"]),
       "confidence": float(row["confidence"]), "rights": str(row["Haklar"]), "methods": {}}
for name, a in atts.items():
    top = a.top_k(10)
    toptok = [a.tokens[i] for i in top]
    out["methods"][name] = {"top10_tokens": toptok,
                            "top10_idx": [int(i) for i in top]}
    print(f"\n[{name}] top-10: {toptok}")

# overlap between methods (top-20 Jaccard)
names = list(atts)
print("\n--- top-20 Jaccard overlap between methods ---")
for i in range(len(names)):
    for j in range(i+1, len(names)):
        s1 = set(atts[names[i]].top_k(20)); s2 = set(atts[names[j]].top_k(20))
        jac = len(s1 & s2) / len(s1 | s2)
        print(f"  {names[i]:20s} vs {names[j]:20s}: {jac:.2f}")

json.dump(out, open(rf"{ROOT}\results\sample_attributions.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\nSaved: results/sample_attributions.json")
