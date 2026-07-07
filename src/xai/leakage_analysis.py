"""Leakage analizi (eleştiri §3.1): (1) keyword cue taraması, (2) TF-IDF+LogReg baseline,
(3) truncation (128/256/512) macro-F1 eğrisi. Hepsi yerel; GPU gerekmez.
Çıktı: results/leakage.json"""
import os, sys, json, re, time
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"; os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, r".\src")
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

ROOT = r"."
MODELDIR = rf"{ROOT}\models"
ds = load_dataset("icgcihan/Turkish_Constutional_Court_Decisions")
tok = AutoTokenizer.from_pretrained(MODELDIR)
out = {}

def head_text(txt, n=512):
    ids = tok(str(txt), truncation=True, max_length=n)["input_ids"]
    return tok.decode(ids, skip_special_tokens=True)

# --- head-512 metinleri (modelin gördüğü pencere) ---
print("head-512 metinleri hazırlanıyor..."); t0 = time.time()
tr_head = [head_text(t) for t in ds["train"]["text"]]
te_head = [head_text(t) for t in ds["test"]["text"]]
tr_y = np.array(ds["train"]["labels"]); te_y = np.array(ds["test"]["labels"])
print(f"  bitti {time.time()-t0:.0f}s")

# --- (1) Keyword cue taraması: cue'lar ilk 512 token içinde ne sıklıkta ve label ile ilişkisi ---
cues = ["ihlal edildiğine", "ihlal edilmediğine", "ihlal edilmiştir", "ihlal edilmemiştir",
        "kabul edilemez", "kabul edilebilir", "başvurunun kabul", "oybirliğiyle", "oyçokluğuyla",
        "karar verilmiştir", "tazminat"]
def norm(s): return s.lower()
te_norm = [norm(t) for t in te_head]
cue_stats = {}
for c in cues:
    present = np.array([c in t for t in te_norm])
    p_all = present.mean()
    # label ile ilişki (present olanlarda label dağılımı)
    if present.sum() > 0:
        p_lab1_given = te_y[present].mean()
    else:
        p_lab1_given = None
    cue_stats[c] = {"prevalence_test": round(float(p_all), 3),
                    "P(label=1|cue)": None if p_lab1_given is None else round(float(p_lab1_given), 3),
                    "n_present": int(present.sum())}
out["keyword_cues_test"] = cue_stats
out["test_base_rate_label1"] = round(float(te_y.mean()), 3)

# --- (2) TF-IDF + Logistic Regression baseline (head-512) ---
print("TF-IDF baseline..."); t0 = time.time()
vec = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), min_df=3)
Xtr = vec.fit_transform(tr_head); Xte = vec.transform(te_head)
clf = LogisticRegression(max_iter=2000, C=1.0, class_weight=None)
clf.fit(Xtr, tr_y)
tfidf_pred = clf.predict(Xte)
out["tfidf_logreg"] = {
    "macro_f1": round(float(f1_score(te_y, tfidf_pred, average="macro")), 3),
    "note": "head-512 metinde TF-IDF(1-2gram)+LogReg; BERT=0.797 ile kıyas"}
print(f"  TF-IDF macro-F1={out['tfidf_logreg']['macro_f1']} ({time.time()-t0:.0f}s)")

# --- (3) Truncation macro-F1 eğrisi (fine-tuned model, CPU inference) ---
print("truncation inference..."); import torch
from xai.explainer import load_model_for_xai
model, _ = load_model_for_xai(MODELDIR, num_labels=2)
model.eval()
def eval_len(L):
    preds = []
    for i in range(0, len(ds["test"]), 16):
        batch = ds["test"]["text"][i:i+16]
        enc = tok([str(x) for x in batch], truncation=True, max_length=L, padding=True, return_tensors="pt")
        with torch.no_grad():
            lg = model(**enc).logits
        preds.extend(lg.argmax(-1).tolist())
    return round(float(f1_score(te_y, preds, average="macro")), 3)
out["truncation_macroF1"] = {}
for L in [128, 256, 512]:
    t0 = time.time(); f = eval_len(L)
    out["truncation_macroF1"][str(L)] = f
    print(f"  L={L}: macro-F1={f} ({time.time()-t0:.0f}s)")

json.dump(out, open(rf"{ROOT}\results\leakage.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("\nKaydedildi: results/leakage.json")
