"""Fine-tuned modeli yükle, yerel tahminin Kaggle ile tutarlılığını doğrula,
gerçek bir yüksek-güvenli örnek üzerinde 4 XAI yöntemini çalıştır."""
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
print(f"Test satır: {len(df)} | havuz: {len(pool)} örnek")

# --- Doğrulama: yerel tahmin == Kaggle tahmini (ilk 15 havuz örneği) ---
ok = 0
for e in pool[:15]:
    row = df[df.idx == e["idx"]].iloc[0]
    pred, probs, _ = exp.predict(str(row["text"]))
    if pred == int(row["pred"]) and abs(float(probs[1]) - float(row["prob_1"])) < 0.02:
        ok += 1
print(f"Yerel↔Kaggle tutarlılık: {ok}/15 (pred eşleşti & prob_1 farkı <0.02)")

# --- İlk gerçek attribution: en yüksek güvenli doğru örnek ---
e = pool[0]
row = df[df.idx == e["idx"]].iloc[0]
text = str(row["text"])
print(f"\n=== Örnek idx={e['idx']} | gerçek={int(row['true'])} pred={int(row['pred'])} "
      f"güven={float(row['confidence']):.3f} | Haklar={row['Haklar']} ===")
print("Metin (ilk 200 krktr):", text[:200])

atts = exp.all_methods(text)
out = {"idx": int(e["idx"]), "true": int(row["true"]), "pred": int(row["pred"]),
       "confidence": float(row["confidence"]), "haklar": str(row["Haklar"]), "methods": {}}
for name, a in atts.items():
    top = a.top_k(10)
    toptok = [a.tokens[i] for i in top]
    out["methods"][name] = {"top10_tokens": toptok,
                            "top10_idx": [int(i) for i in top]}
    print(f"\n[{name}] top-10: {toptok}")

# yöntemler arası örtüşme (top-20 Jaccard)
names = list(atts)
print("\n--- Yöntemler arası top-20 Jaccard örtüşme ---")
for i in range(len(names)):
    for j in range(i+1, len(names)):
        s1 = set(atts[names[i]].top_k(20)); s2 = set(atts[names[j]].top_k(20))
        jac = len(s1 & s2) / len(s1 | s2)
        print(f"  {names[i]:20s} vs {names[j]:20s}: {jac:.2f}")

json.dump(out, open(rf"{ROOT}\results\sample_attributions.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print("\nKaydedildi: results/sample_attributions.json")
