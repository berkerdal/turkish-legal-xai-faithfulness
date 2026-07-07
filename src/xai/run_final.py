"""Faz 4 nihai koşu: 4 yöntem + RASTGELE baseline; attributions'ı kaydet (RQ2/ısı haritası için).
Çıktı: results/faithfulness_full_raw.csv, results/attributions_pool.json"""
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
df = pd.read_csv(rf"{ROOT}\results\test_predictions.csv").set_index("idx")
pool = json.load(open(rf"{ROOT}\results\example_pool.json", encoding="utf-8"))

rows, saved = [], {}
t0 = time.time()
for n, e in enumerate(pool, 1):
    idx = e["idx"]; text = str(df.loc[idx, "text"])
    atts = exp.all_methods(text)
    # rastgele baseline: gerçek tahmin + rastgele skorlar
    ref = next(iter(atts.values()))
    rand = Attribution("random", ref.tokens, rng.random(len(ref.tokens)), ref.pred, ref.prob)
    atts["random"] = rand

    saved[str(idx)] = {"true": int(df.loc[idx, "true"]), "pred": ref.pred,
                       "haklar": str(df.loc[idx, "Haklar"]), "tokens": ref.tokens,
                       "scores": {m: [round(float(x), 5) for x in a.scores] for m, a in atts.items()}}
    for m, a in atts.items():
        r = F.evaluate(exp, text, a)
        rows.append({"idx": idx, "method": m,
                     "comprehensiveness": r["comprehensiveness"], "sufficiency": r["sufficiency"]})
    print(f"  {n}/{len(pool)} (idx={idx}) | {time.time()-t0:.0f}s")

pd.DataFrame(rows).to_csv(rf"{ROOT}\results\faithfulness_full_raw.csv", index=False)
json.dump(saved, open(rf"{ROOT}\results\attributions_pool.json", "w", encoding="utf-8"),
          ensure_ascii=False)
print("\nKaydedildi: faithfulness_full_raw.csv + attributions_pool.json")
