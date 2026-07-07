"""Yayın figürleri (300 dpi): (1) faithfulness, (2) katman sağlamlığı, (3) ısı haritası.
Okabe-Ito renk-körü-güvenli palet; operatör = hatch (print-güvenli); renk = yöntem (varlık)."""
import os, json
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"."
plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
                     "grid.linewidth": 0.6, "axes.axisbelow": True})
# yöntem → renk (sabit, varlığı izler)
C = {"Chefer": "#0072B2", "Bütünleşik Gradyanlar": "#56B4E9",
     "Ham dikkat": "#E69F00", "Dikkat yayılımı": "#D55E00", "Rastgele": "#999999"}

# ---------- Şekil 1: Kapsayıcılık, iki operatör ----------
methods = ["Bütünleşik Gradyanlar", "Chefer", "Ham dikkat", "Dikkat yayılımı", "Rastgele"]
comp_mask = [0.240, 0.239, 0.106, 0.079, 0.056]
comp_del  = [0.251, 0.261, 0.113, 0.076, 0.049]
x = np.arange(len(methods)); w = 0.38
fig, ax = plt.subplots(figsize=(7.2, 4.0))
b1 = ax.bar(x - w/2, comp_mask, w, color=[C[m] for m in methods], edgecolor="white", linewidth=0.6)
b2 = ax.bar(x + w/2, comp_del, w, color=[C[m] for m in methods], edgecolor="white",
            linewidth=0.6, hatch="////", alpha=0.99)
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}", (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=8, color="#333")
ax.set_xticks(x); ax.set_xticklabels(methods, rotation=18, ha="right")
ax.set_ylabel("Kapsayıcılık (yüksek = daha sadık)")
ax.set_title("Şekil 1. Açıklama yöntemlerinin kapsayıcılık sadakati", fontsize=11, loc="left")
from matplotlib.patches import Patch
ax.legend([Patch(facecolor="#777"), Patch(facecolor="#777", hatch="////")],
          ["[MASK] operatörü", "delete operatörü"], frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(rf"{ROOT}\results\fig1_faithfulness.png", dpi=600); plt.close(fig)

# ---------- Şekil 2: Katman sağlamlığı (kapsayıcılık, mask) ----------
strata = ["Yüksek-güvenli\ndoğru", "Düşük-güvenli\ndoğru", "Yanlış"]
vals = {"Ham dikkat": [0.162, 0.020, 0.136], "Dikkat yayılımı": [0.143, -0.014, 0.108],
        "Bütünleşik Gradyanlar": [0.225, 0.216, 0.278], "Chefer": [0.252, 0.190, 0.276],
        "Rastgele": [0.051, 0.044, 0.065]}
order = ["Bütünleşik Gradyanlar", "Chefer", "Ham dikkat", "Dikkat yayılımı", "Rastgele"]
x = np.arange(len(strata)); w = 0.16
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for i, m in enumerate(order):
    bars = ax.bar(x + (i-2)*w, vals[m], w, color=C[m], edgecolor="white", linewidth=0.6, label=m)
    for b in bars:
        ax.annotate(f"{b.get_height():.2f}", (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom" if b.get_height()>=0 else "top", fontsize=7, color="#333")
ax.axhline(0, color="#888", linewidth=0.8)
ax.set_xticks(x); ax.set_xticklabels(strata)
ax.set_ylabel("Kapsayıcılık (yüksek = daha sadık)")
ax.set_title("Şekil 2. Güven ve hata katmanlarına göre kapsayıcılık", fontsize=11, loc="left")
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
fig.tight_layout(); fig.savefig(rf"{ROOT}\results\fig2_stratum.png", dpi=600, bbox_inches="tight"); plt.close(fig)

# ---------- Şekil 3: Isı haritası (nitel) ----------
data = json.load(open(rf"{ROOT}\results\attributions_rev.json", encoding="utf-8"))
pick = next((k for k, v in data.items() if v["stratum"] == "yuksek_guven_dogru"), list(data)[0])
rec = data[pick]; N = 40
toks = rec["tokens"][:N]
rows = ["Ham dikkat", "Dikkat yayılımı", "Bütünleşik Gradyanlar", "Chefer"]
key = {"Ham dikkat": "raw_attention", "Dikkat yayılımı": "attention_rollout",
       "Bütünleşik Gradyanlar": "integrated_gradients", "Chefer": "chefer_relevance"}
M = np.array([rec["scores"][key[r]][:N] for r in rows])
fig, ax = plt.subplots(figsize=(min(0.28*N, 12), 2.8))
im = ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows)
ax.set_xticks(range(N)); ax.set_xticklabels([t.replace("##", "") for t in toks], rotation=90, fontsize=6)
ax.set_title("Şekil 3. Token-düzeyi önem ısı haritası (ilk %d belirteç)" % N, fontsize=11, loc="left")
fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="önem (0–1)")
fig.tight_layout(); fig.savefig(rf"{ROOT}\results\fig3_heatmap.png", dpi=600, bbox_inches="tight"); plt.close(fig)

print("Kaydedildi: fig1_faithfulness.png, fig2_stratum.png, fig3_heatmap.png")
print("Isı haritası örnek idx:", pick, "| stratum:", rec["stratum"])
