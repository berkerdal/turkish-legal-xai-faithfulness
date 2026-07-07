"""Fig 3 örneği için yöntem başına en yüksek atıflı token'lar (top-token tablosu için)."""
import os, json
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np
ROOT = r"."
data = json.load(open(rf"{ROOT}\results\attributions_rev.json", encoding="utf-8"))
pick = next((k for k, v in data.items() if v["stratum"] == "yuksek_guven_dogru"), list(data)[0])
rec = data[pick]; toks = rec["tokens"]
methods = {"Raw attention":"raw_attention","Attention rollout":"attention_rollout",
           "Integrated Gradients":"integrated_gradients","Chefer":"chefer_relevance"}
special = {"[CLS]","[SEP]","[PAD]"}
print("idx:", pick, "| true/pred:", rec["true"], rec["pred"])
for name,key in methods.items():
    s=np.array(rec["scores"][key],float)
    order=[i for i in np.argsort(-s) if toks[i] not in special][:6]
    print(f"{name:22s}:", [f"{toks[i]}({s[i]:.2f})" for i in order])
