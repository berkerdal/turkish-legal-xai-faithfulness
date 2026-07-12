"""Supplementary figures (English, 600 dpi): comprehensiveness distribution,
token-type stacked bar, and the leakage keyword evidence. Okabe-Ito palette,
consistent with the main figures."""
import os, json
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"."
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True})
CM = {"Chefer":"#0072B2","Integrated Gradients":"#56B4E9","Raw attention":"#E69F00",
      "Attention rollout":"#D55E00","Random":"#999999"}
short = {"raw_attention":"Raw attention","attention_rollout":"Attention rollout",
         "integrated_gradients":"Integrated Gradients","chefer_relevance":"Chefer","random":"Random"}

# ---- A2: comprehensiveness distribution (mask) ----
raw = pd.read_csv(rf"{ROOT}\results\faithfulness_rev_raw.csv")
sub = raw[raw.operator=="mask"]
order = ["Integrated Gradients","Chefer","Raw attention","Attention rollout","Random"]
inv = {v:k for k,v in short.items()}
fig, ax = plt.subplots(figsize=(7.2,4.2))
for i,name in enumerate(order):
    vals = sub[sub.method==inv[name]]["comprehensiveness"].values
    ax.boxplot(vals, positions=[i], widths=0.5, patch_artist=True,
               boxprops=dict(facecolor=CM[name], alpha=0.5, edgecolor=CM[name]),
               medianprops=dict(color="#222"), showfliers=False)
    ax.scatter(np.random.normal(i,0.06,len(vals)), vals, s=10, color=CM[name], alpha=0.6, edgecolor="white", linewidth=0.3)
ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=18, ha="right")
ax.axhline(0, color="#888", lw=0.8)
ax.set_ylabel("Comprehensiveness ([MASK])")
ax.set_title("Figure S2. Distribution of comprehensiveness across the 60 instances", fontsize=11, loc="left")
fig.tight_layout(); fig.savefig(rf"{ROOT}\results\figA1_dist_en.png", dpi=600, bbox_inches="tight"); plt.close(fig)

# ---- A2: token-type stacked bar ----
rq2 = pd.read_csv(rf"{ROOT}\results\rq2_rev_full.csv", index_col=0)
types = ["content","punctuation","subword","stopword","number"]
tcol = {"content":"#88a0b8","punctuation":"#D55E00","subword":"#56B4E9","stopword":"#CC79A7","number":"#009E73"}
rows = ["Raw attention","Attention rollout","Integrated Gradients","Chefer","Random"]
fig, ax = plt.subplots(figsize=(7.6,3.8))
left = np.zeros(len(rows))
for ty in types:
    vals = [rq2.loc[r, ty] for r in rows]
    ax.barh(rows, vals, left=left, color=tcol[ty], edgecolor="white", linewidth=0.6, label=ty)
    left += np.array(vals)
ax.set_xlabel("Share of importance mass (%)")
ax.set_title("Figure S3. Attribution mass by token type", fontsize=11, loc="left")
ax.legend(ncol=5, frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5,-0.15))
ax.invert_yaxis()
fig.tight_layout(); fig.savefig(rf"{ROOT}\results\figA2_tokentype_en.png", dpi=600, bbox_inches="tight"); plt.close(fig)

# ---- A3: leakage keyword evidence ----
lk = json.load(open(rf"{ROOT}\results\leakage.json", encoding="utf-8"))
base = lk["test_base_rate_label1"]
cues = {k:v for k,v in lk["keyword_cues_test"].items() if v["prevalence_test"]>0}
labels = list(cues.keys())
prev = [cues[c]["prevalence_test"] for c in labels]
pcond = [cues[c]["P(label=1|cue)"] for c in labels]
verdict = {"ihlal edildiğine","kabul edilemez","oybirliğiyle","oyçokluğuyla"}
fig, (a1,a2) = plt.subplots(1,2, figsize=(9.5,3.8))
cols = ["#D55E00" if l in verdict else "#56B4E9" for l in labels]
a1.barh(labels, prev, color=cols, edgecolor="white")
a1.set_xlabel("Prevalence in first 512 tokens"); a1.invert_yaxis()
a1.set_title("Figure S4. Outcome-cue prevalence", fontsize=10, loc="left")
a2.scatter(pcond, range(len(labels)), color=cols, s=45, zorder=3)
a2.axvline(base, color="#333", ls="--", lw=1, label=f"base rate = {base:.2f}")
a2.set_yticks(range(len(labels))); a2.set_yticklabels([]); a2.invert_yaxis()
a2.set_xlim(0,1); a2.set_xlabel("P(violation | cue present)")
a2.set_title("Cue vs. outcome (verdict cues in orange)", fontsize=10, loc="left")
a2.legend(frameon=False, fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(rf"{ROOT}\results\figA3_leakage_en.png", dpi=600, bbox_inches="tight"); plt.close(fig)
print("Saved: figA1_dist_en.png, figA2_tokentype_en.png, figA3_leakage_en.png")
