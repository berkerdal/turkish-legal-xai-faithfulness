"""English publication figures (300 dpi). Same data/colors as make_figures.py, English labels.
Token text in Fig 3 stays in Turkish (it is the actual data)."""
import os, json
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = r"."
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.6, "axes.axisbelow": True})
C = {"Chefer": "#0072B2", "Integrated Gradients": "#56B4E9",
     "Raw attention": "#E69F00", "Attention rollout": "#D55E00", "Random": "#999999"}

# ---- Figure 1: Comprehensiveness, two operators ----
methods = ["Integrated Gradients", "Chefer", "Raw attention", "Attention rollout", "Random"]
comp_mask = [0.240, 0.239, 0.106, 0.079, 0.056]
comp_del  = [0.251, 0.261, 0.113, 0.076, 0.049]
x = np.arange(len(methods)); w = 0.38
fig, ax = plt.subplots(figsize=(7.2, 4.0))
b1 = ax.bar(x - w/2, comp_mask, w, color=[C[m] for m in methods], edgecolor="white", linewidth=0.6)
b2 = ax.bar(x + w/2, comp_del, w, color=[C[m] for m in methods], edgecolor="white",
            linewidth=0.6, hatch="////")
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}", (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, color="#333")
ax.set_xticks(x); ax.set_xticklabels(methods, rotation=18, ha="right")
ax.set_ylabel("Comprehensiveness (higher = more faithful)")
ax.set_title("Figure 1. Comprehensiveness faithfulness of explanation methods", fontsize=11, loc="left")
ax.legend([Patch(facecolor="#777"), Patch(facecolor="#777", hatch="////")],
          ["[MASK] operator", "delete operator"], frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(rf"{ROOT}\results\fig1_faithfulness_en.png", dpi=600); plt.close(fig)

# ---- Figure 2: Robustness across strata ----
strata = ["High-confidence\ncorrect", "Low-confidence\ncorrect", "Incorrect"]
vals = {"Raw attention": [0.162, 0.020, 0.136], "Attention rollout": [0.143, -0.014, 0.108],
        "Integrated Gradients": [0.225, 0.216, 0.278], "Chefer": [0.252, 0.190, 0.276],
        "Random": [0.051, 0.044, 0.065]}
order = ["Integrated Gradients", "Chefer", "Raw attention", "Attention rollout", "Random"]
x = np.arange(len(strata)); w = 0.16
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for i, m in enumerate(order):
    bars = ax.bar(x + (i-2)*w, vals[m], w, color=C[m], edgecolor="white", linewidth=0.6, label=m)
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}", (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom" if b.get_height()>=0 else "top", fontsize=6.5, color="#333")
ax.axhline(0, color="#888", linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(strata)
ax.set_ylabel("Comprehensiveness (higher = more faithful)")
ax.set_title("Figure 2. Comprehensiveness across confidence and error strata", fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(); fig.savefig(rf"{ROOT}\results\fig2_stratum_en.png", dpi=600, bbox_inches="tight"); plt.close(fig)

# ---- Figure 3: Heatmap ----
data = json.load(open(rf"{ROOT}\results\attributions_rev.json", encoding="utf-8"))
pick = next((k for k, v in data.items() if v["stratum"] == "high_confidence_correct"), list(data)[0])
rec = data[pick]; N = 40
toks = rec["tokens"][:N]
rows = ["Raw attention", "Attention rollout", "Integrated Gradients", "Chefer"]
key = {"Raw attention": "raw_attention", "Attention rollout": "attention_rollout",
       "Integrated Gradients": "integrated_gradients", "Chefer": "chefer_relevance"}
M = np.array([rec["scores"][key[r]][:N] for r in rows])
fig, ax = plt.subplots(figsize=(min(0.28*N, 12), 2.8))
im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
ax.set_xticks(range(N)); ax.set_xticklabels([t.replace("##", "") for t in toks], rotation=90, fontsize=6)
ax.set_title("Figure 3. Token-level importance heatmap (first %d tokens)" % N, fontsize=11, loc="left")
fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="importance (0–1)")
fig.tight_layout(); fig.savefig(rf"{ROOT}\results\fig3_heatmap_en.png", dpi=600, bbox_inches="tight"); plt.close(fig)
print("Saved: fig1_faithfulness_en.png, fig2_stratum_en.png, fig3_heatmap_en.png")
