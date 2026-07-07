"""Katman kompozisyonu (sınıf sayıları) — Supplementary Table için."""
import os, json
os.environ["PYTHONIOENCODING"] = "utf-8"
import pandas as pd
ROOT = r"."
df = pd.read_csv(rf"{ROOT}\results\test_predictions.csv")
correct = df[df.correct].copy(); errors = df[~df.correct].copy()
hi = correct.sort_values("confidence", ascending=False).head(20).assign(stratum="High-confidence correct")
lo = correct.sort_values("confidence", ascending=True).head(20).assign(stratum="Low-confidence correct")
fp = errors[errors.pred == 1].head(10); fn = errors[errors.pred == 0].head(10)
err = pd.concat([fp, fn]).assign(stratum="Incorrect")
sample = pd.concat([hi, lo, err]).drop_duplicates("idx")
print("n =", len(sample))
for st in ["High-confidence correct", "Low-confidence correct", "Incorrect"]:
    s = sample[sample.stratum == st]
    c0 = int((s.true == 0).sum()); c1 = int((s.true == 1).sum())
    conf = s.confidence
    print(f"{st}: n={len(s)} | class0={c0} class1={c1} | conf [{conf.min():.3f}, {conf.max():.3f}]")
print("\nIncorrect breakdown: FP(pred=1)=%d, FN(pred=0)=%d" %
      (int((err.pred==1).sum()), int((err.pred==0).sum())))
