"""Qualitative heatmap (HTML): 2 instances x 4 methods, tokens coloured by
importance score. Reads attributions_pool.json; no model needed."""
import os, json, html
os.environ["PYTHONIOENCODING"] = "utf-8"
import numpy as np

ROOT = r"."
data = json.load(open(rf"{ROOT}\results\attributions_pool.json", encoding="utf-8"))
METHODS = ["raw_attention", "attention_rollout", "integrated_gradients", "chefer_relevance"]
LABELS = {"raw_attention": "Raw attention", "attention_rollout": "Attention rollout",
          "integrated_gradients": "Integrated Gradients", "chefer_relevance": "Chefer relevance"}
NCAP = 70   # number of leading tokens to show

def pick(label):
    for idx, rec in data.items():
        if rec["true"] == label and rec["pred"] == label:
            return idx, rec
    return None, None

def tok_html(t):
    return html.escape(t[2:] if t.startswith("##") else (" " + t))

def render(idx, rec):
    toks = rec["tokens"][:NCAP]
    parts = [f'<h3>Instance idx={idx}: true/pred={rec["true"]} · <span class="rights">{html.escape(rec["haklar"])}</span></h3>']
    for m in METHODS:
        s = np.array(rec["scores"][m][:NCAP], dtype=float)
        spans = []
        for t, v in zip(toks, s):
            if t in ("[CLS]", "[SEP]", "[PAD]"):
                continue
            a = float(np.clip(v, 0, 1))
            spans.append(f'<span style="background:rgba(214,40,40,{a:.2f})" title="{v:.2f}">{tok_html(t)}</span>')
        parts.append(f'<div class="row"><div class="name">{LABELS[m]}</div>'
                     f'<div class="txt">{"".join(spans)}</div></div>')
    return "\n".join(parts)

body = []
for lab in (1, 0):
    idx, rec = pick(lab)
    if rec:
        body.append(render(idx, rec))

htmldoc = f"""<style>
body{{font-family:Georgia,serif;max-width:1000px;margin:24px auto;color:#1a1a1a;line-height:2.1}}
h2{{border-bottom:2px solid #333}}
.row{{display:flex;gap:12px;margin:10px 0;align-items:baseline}}
.name{{flex:0 0 160px;font-family:sans-serif;font-size:13px;font-weight:600;color:#444;text-align:right}}
.txt{{flex:1;font-size:15px}}
.txt span{{padding:1px 0;border-radius:2px}}
.rights{{font-size:13px;color:#666;font-style:italic}}
h3{{font-family:sans-serif;font-size:15px;margin-top:28px}}
.note{{font-family:sans-serif;font-size:13px;color:#666;background:#f5f5f5;padding:10px;border-radius:6px}}
</style>
<h2>Turkish Constitutional Court decision classification: XAI heatmaps</h2>
<p class="note">Red intensity = token importance score for the predicted class (0-1, normalised). First {NCAP} tokens.
Note how raw attention favours punctuation and surface tokens while Chefer and Integrated Gradients focus on meaningful legal terms.</p>
{"".join(body)}
"""
open(rf"{ROOT}\results\heatmaps.html", "w", encoding="utf-8").write(htmldoc)
print("Saved: results/heatmaps.html")
