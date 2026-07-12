"""Main-text figure for the generative-model self-explanation result (RQ4), 600 dpi.
Per-instance comprehensiveness (n=60) for the model's self-explanation, an occlusion
reference, and a random baseline, under both deletion operators. Same palette/typography
as make_figures_en.py so it reads as one figure family with Figures 1 and 2.
Run from the repository root."""
import os, csv
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"."
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.6, "axes.axisbelow": True})

# Colour grammar carried over from Figures 1-2: blue = faithful reference,
# orange = the user-facing signal that turns out weak, grey = random floor.
COL = {"Self-explanation": "#E69F00", "Occlusion": "#0072B2", "Random": "#999999"}
CONDS = ["Self-explanation", "Occlusion", "Random"]
KEY = {"Self-explanation": "self_comp_{op}", "Occlusion": "occ_comp_{op}", "Random": "rand_comp_{op}"}

rows = list(csv.DictReader(open(rf"{ROOT}\results\llm_perinstance.csv", encoding="utf-8")))
def arr(cond, op): return np.array([float(r[KEY[cond].format(op=op)]) for r in rows])

fig, axes = plt.subplots(1, 2, figsize=(7.6, 4.0), sharey=True)
panels = [("mask", "(a) [MASK] operator"), ("delete", "(b) delete operator")]
rng = np.random.default_rng(42)

for ax, (op, ptitle) in zip(axes, panels):
    data = [arr(c, op) for c in CONDS]
    pos = np.arange(1, len(CONDS) + 1)
    bp = ax.boxplot(data, positions=pos, widths=0.55, showfliers=False,
                    medianprops=dict(color="#222", linewidth=1.4),
                    boxprops=dict(linewidth=1.0), whiskerprops=dict(linewidth=1.0),
                    capprops=dict(linewidth=1.0), patch_artist=True)
    for patch, c in zip(bp["boxes"], CONDS):
        patch.set_facecolor(COL[c]); patch.set_alpha(0.28); patch.set_edgecolor(COL[c])
    for i, (c, d) in enumerate(zip(CONDS, data), start=1):
        jit = rng.uniform(-0.16, 0.16, size=len(d))
        ax.scatter(np.full(len(d), i) + jit, d, s=12, color=COL[c],
                   alpha=0.75, edgecolor="white", linewidth=0.3, zorder=3)
    # random mean as a shared reference line so the self/random overlap is visible
    rmean = arr("Random", op).mean()
    ax.axhline(rmean, color="#888", ls="--", lw=0.9, zorder=1)
    ax.set_xticks(pos); ax.set_xticklabels(CONDS, rotation=12, ha="right")
    ax.set_title(ptitle, fontsize=10, loc="left")

axes[0].set_ylabel("Comprehensiveness (higher = more faithful)")
fig.suptitle("Figure 3. Faithfulness of the generative model's self-explanation (n = 60)",
             fontsize=11, x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(rf"{ROOT}\results\figures\fig_llm_en.png", dpi=600, bbox_inches="tight")
plt.close(fig)
print("Saved: results/figures/fig_llm_en.png")
