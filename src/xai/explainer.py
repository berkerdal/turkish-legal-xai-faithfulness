"""Faz 3 — XAI ortak arayüzü.

Tüm yöntemler aynı `Attribution` nesnesini döndürür: tahmin edilen sınıf için
token-düzeyi önem skorları. Böylece Faz 4'te faithfulness (comprehensiveness/
sufficiency, perturbation) tek bir arayüz üzerinden ölçülür.

Yöntemler:
  - raw_attention      : son katman, kafa-ortalaması, [CLS] satırı  (alt-sınır baseline)
  - attention_rollout  : Abnar & Zuidema (2020)
  - integrated_gradients : Captum LayerIntegratedGradients (embedding katmanı)
  - lrp / chefer       : ayrı modül (spike sonrası) — bkz. explainer_lrp.py
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Anaconda+pip-torch OpenMP çakışması
from dataclasses import dataclass
from typing import List
import numpy as np
import torch


def load_model_for_xai(model_path: str, num_labels: int = 2):
    """XAI için model+tokenizer yükle. attention çıkarımı için attn_implementation='eager'
    ŞARTTIR (sdpa/flash output_attentions desteklemez)."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels, attn_implementation="eager")
    return model, tok


@dataclass
class Attribution:
    method: str
    tokens: List[str]      # özel tokenlar dahil (wordpiece)
    scores: np.ndarray     # (seq_len,) tahmin edilen sınıf için önem
    pred: int
    prob: float

    def top_k(self, k: int, exclude_special: bool = True):
        """En önemli k token'ın indekslerini döndür (özel tokenları hariç tut)."""
        mask = np.ones(len(self.scores), dtype=bool)
        if exclude_special:
            for i, t in enumerate(self.tokens):
                if t in ("[CLS]", "[SEP]", "[PAD]"):
                    mask[i] = False
        idx = np.where(mask)[0]
        order = idx[np.argsort(-self.scores[idx])]
        return order[:k]


class Explainer:
    def __init__(self, model, tokenizer, device: str = "cpu", max_len: int = 512):
        self.model = model.to(device).eval()
        self.tok = tokenizer
        self.device = device
        self.max_len = max_len
        self.special_ids = set(tokenizer.all_special_ids)

    # ---- ortak yardımcılar ----
    def _encode(self, text: str):
        enc = self.tok(text, truncation=True, max_length=self.max_len,
                       return_tensors="pt")
        return {k: v.to(self.device) for k, v in enc.items()}

    def _tokens(self, input_ids) -> List[str]:
        return self.tok.convert_ids_to_tokens(input_ids[0].tolist())

    @torch.no_grad()
    def predict(self, text: str):
        enc = self._encode(text)
        logits = self.model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0]
        pred = int(probs.argmax())
        return pred, probs.detach().cpu().numpy(), enc

    @torch.no_grad()
    def prob_from_ids(self, input_ids, attention_mask, target: int) -> float:
        """Verilen input_ids için hedef sınıf olasılığı (faithfulness için)."""
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return float(torch.softmax(logits, dim=-1)[0, target])

    def _finalize(self, method, tokens, scores, pred, prob):
        scores = np.asarray(scores, dtype=np.float64)
        # özel tokenları sıfırla (önem taşımasınlar)
        for i, t in enumerate(tokens):
            if t in ("[CLS]", "[SEP]", "[PAD]"):
                scores[i] = 0.0
        # negatifleri kırp + normalize (0-1) — yöntemler arası kıyas için
        scores = np.clip(scores, 0, None)
        if scores.max() > 0:
            scores = scores / scores.max()
        return Attribution(method, tokens, scores, pred, float(prob))

    # ---- Yöntem A: Ham attention ----
    @torch.no_grad()
    def raw_attention(self, text: str) -> Attribution:
        enc = self._encode(text)
        out = self.model(**enc, output_attentions=True)
        pred = int(torch.softmax(out.logits, -1).argmax())
        prob = float(torch.softmax(out.logits, -1)[0, pred])
        att = out.attentions[-1][0]              # (heads, seq, seq)
        att = att.mean(0)                        # kafa ortalaması (seq, seq)
        cls_to_tokens = att[0]                   # [CLS]'in her token'a dikkati
        tokens = self._tokens(enc["input_ids"])
        return self._finalize("raw_attention", tokens, cls_to_tokens.cpu().numpy(), pred, prob)

    # ---- Yöntem B: Attention rollout (Abnar & Zuidema 2020) ----
    @torch.no_grad()
    def attention_rollout(self, text: str) -> Attribution:
        enc = self._encode(text)
        out = self.model(**enc, output_attentions=True)
        pred = int(torch.softmax(out.logits, -1).argmax())
        prob = float(torch.softmax(out.logits, -1)[0, pred])
        atts = [a[0].mean(0) for a in out.attentions]   # her katman: (seq, seq)
        seq = atts[0].size(0)
        eye = torch.eye(seq, device=atts[0].device)
        rollout = eye.clone()
        for a in atts:
            a_res = 0.5 * a + 0.5 * eye                  # residual bağlantı
            a_res = a_res / a_res.sum(dim=-1, keepdim=True)
            rollout = a_res @ rollout
        cls_to_tokens = rollout[0]
        tokens = self._tokens(enc["input_ids"])
        return self._finalize("attention_rollout", tokens, cls_to_tokens.cpu().numpy(), pred, prob)

    # ---- Yöntem C: Integrated Gradients (Captum) ----
    def integrated_gradients(self, text: str, n_steps: int = 50) -> Attribution:
        from captum.attr import LayerIntegratedGradients
        enc = self._encode(text)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        pred, probs, _ = self.predict(text)
        prob = float(probs[pred])

        # baseline: özel tokenları koru, gerisini [PAD]
        pad_id = self.tok.pad_token_id
        ref = input_ids.clone()
        for i, tid in enumerate(input_ids[0].tolist()):
            if tid not in self.special_ids:
                ref[0, i] = pad_id

        def forward_func(ids, mask):
            return self.model(input_ids=ids, attention_mask=mask).logits

        emb_layer = self.model.base_model.embeddings
        lig = LayerIntegratedGradients(forward_func, emb_layer)
        attributions = lig.attribute(inputs=input_ids, baselines=ref,
                                     additional_forward_args=(attention_mask,),
                                     target=pred, n_steps=n_steps,
                                     internal_batch_size=8)
        scores = attributions.sum(dim=-1).squeeze(0)    # embedding boyutunu topla
        tokens = self._tokens(input_ids)
        return self._finalize("integrated_gradients", tokens, scores.detach().cpu().numpy(), pred, prob)

    # ---- Yöntem D: Chefer grad-ağırlıklı attention relevance (2021) ----
    def chefer_relevance(self, text: str) -> Attribution:
        """Chefer et al. (2021), 'Generic Attention-model Explainability'.
        Her katmanda (attention ⊙ ∂logit/∂attention)+ kafa-ortalaması alınıp rollout edilir.
        Vanilla-LRP'nin transformer'daki propagation sorunlarını aşan, atıf-sadık yöntem."""
        enc = self._encode(text)
        out = self.model(**enc, output_attentions=True)
        probs = torch.softmax(out.logits, -1)[0]
        pred = int(probs.argmax()); prob = float(probs[pred])
        logit = out.logits[0, pred]
        grads = torch.autograd.grad(logit, out.attentions, retain_graph=False)

        seq = out.attentions[0].size(-1)
        R = torch.eye(seq, device=out.attentions[0].device)
        for A, G in zip(out.attentions, grads):
            cam = (G[0] * A[0]).clamp(min=0).mean(0)     # (seq, seq), pozitif katkı, kafa ort.
            R = R + cam @ R                              # residual rollout
        rel = R[0]                                       # [CLS] satırı
        tokens = self._tokens(enc["input_ids"])
        return self._finalize("chefer_relevance", tokens, rel.detach().cpu().numpy(), pred, prob)

    def all_methods(self, text: str):
        """Dört yöntemi de tek çağrıda döndür (Faz 4 için)."""
        return {a.method: a for a in [
            self.raw_attention(text), self.attention_rollout(text),
            self.integrated_gradients(text), self.chefer_relevance(text)]}
