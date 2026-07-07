"""Figure S1: AOPC comprehensiveness curves (fraction removed vs probability drop)."""
import os, json
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = r"."
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})
c = json.load(open(rf"{ROOT}\results\aopc_curve.json", encoding="utf-8"))
ks = [1, 5, 10, 20, 50]
style = {"integrated_gradients":("Integrated Gradients","#56B4E9","o","-"),
         "chefer_relevance":("Chefer","#0072B2","s","-"),
         "raw_attention":("Raw attention","#E69F00","^","-"),
         "attention_rollout":("Attention rollout","#D55E00","v","-"),
         "random":("Random","#999999","D","--")}
fig, ax = plt.subplots(figsize=(7.0,4.4))
for m,(name,col,mk,ls) in style.items():
    y = [c[m][str(k/100)] for k in ks]
    ax.plot(ks, y, ls, color=col, marker=mk, markersize=7, linewidth=2, label=name,
            markeredgecolor="white", markeredgewidth=0.6)
ax.set_xticks(ks); ax.set_xlabel("Fraction of most-important tokens removed (%)")
ax.set_ylabel("Comprehensiveness (probability drop)")
ax.set_title("Figure S1. Comprehensiveness perturbation curves ([MASK] operator)", fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(rf"{ROOT}\results\figS1_aopc_en.png", dpi=600, bbox_inches="tight"); plt.close(fig)
print("Saved: figS1_aopc_en.png")
