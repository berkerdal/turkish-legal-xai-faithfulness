"""Explanation methods with a common interface.

Every method returns an `Attribution`: token-level importance scores for the
predicted class, so the faithfulness metrics (comprehensiveness, sufficiency)
are computed the same way for all of them.

Methods:
  - raw_attention        : last layer, head-averaged, [CLS] row (lower-bound baseline)
  - attention_rollout    : Abnar & Zuidema (2020)
  - integrated_gradients : Captum LayerIntegratedGradients over the embedding layer
  - chefer_relevance     : Chefer et al. (2021), gradient-weighted attention relevance
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # avoid an OpenMP clash between Anaconda and pip torch
from dataclasses import dataclass
from typing import List
import numpy as np
import torch


def load_model_for_xai(model_path: str, num_labels: int = 2):
    """Load the model and tokenizer. attn_implementation='eager' is required
    because sdpa/flash attention do not support output_attentions."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=num_labels, attn_implementation="eager")
    return model, tok


@dataclass
class Attribution:
    method: str
    tokens: List[str]      # wordpiece tokens, special tokens included
    scores: np.ndarray     # (seq_len,) importance for the predicted class
    pred: int
    prob: float

    def top_k(self, k: int, exclude_special: bool = True):
        """Indices of the k most important tokens (special tokens excluded)."""
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

    # ---- shared helpers ----
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
        """Target-class probability for the given input_ids (used by the metrics)."""
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return float(torch.softmax(logits, dim=-1)[0, target])

    def _finalize(self, method, tokens, scores, pred, prob):
        scores = np.asarray(scores, dtype=np.float64)
        # zero out special tokens so they carry no importance
        for i, t in enumerate(tokens):
            if t in ("[CLS]", "[SEP]", "[PAD]"):
                scores[i] = 0.0
        # clip negatives and normalise to 0-1 so methods are comparable
        scores = np.clip(scores, 0, None)
        if scores.max() > 0:
            scores = scores / scores.max()
        return Attribution(method, tokens, scores, pred, float(prob))

    # ---- Method A: raw attention ----
    @torch.no_grad()
    def raw_attention(self, text: str) -> Attribution:
        enc = self._encode(text)
        out = self.model(**enc, output_attentions=True)
        pred = int(torch.softmax(out.logits, -1).argmax())
        prob = float(torch.softmax(out.logits, -1)[0, pred])
        att = out.attentions[-1][0]              # (heads, seq, seq)
        att = att.mean(0)                        # average over heads (seq, seq)
        cls_to_tokens = att[0]                   # attention from [CLS] to each token
        tokens = self._tokens(enc["input_ids"])
        return self._finalize("raw_attention", tokens, cls_to_tokens.cpu().numpy(), pred, prob)

    # ---- Method B: attention rollout (Abnar & Zuidema 2020) ----
    @torch.no_grad()
    def attention_rollout(self, text: str) -> Attribution:
        enc = self._encode(text)
        out = self.model(**enc, output_attentions=True)
        pred = int(torch.softmax(out.logits, -1).argmax())
        prob = float(torch.softmax(out.logits, -1)[0, pred])
        atts = [a[0].mean(0) for a in out.attentions]   # per layer: (seq, seq)
        seq = atts[0].size(0)
        eye = torch.eye(seq, device=atts[0].device)
        rollout = eye.clone()
        for a in atts:
            a_res = 0.5 * a + 0.5 * eye                  # residual connection
            a_res = a_res / a_res.sum(dim=-1, keepdim=True)
            rollout = a_res @ rollout
        cls_to_tokens = rollout[0]
        tokens = self._tokens(enc["input_ids"])
        return self._finalize("attention_rollout", tokens, cls_to_tokens.cpu().numpy(), pred, prob)

    # ---- Method C: Integrated Gradients (Captum) ----
    def integrated_gradients(self, text: str, n_steps: int = 50) -> Attribution:
        from captum.attr import LayerIntegratedGradients
        enc = self._encode(text)
        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]
        pred, probs, _ = self.predict(text)
        prob = float(probs[pred])

        # baseline: keep special tokens, replace the rest with [PAD]
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
        scores = attributions.sum(dim=-1).squeeze(0)    # sum over the embedding dimension
        tokens = self._tokens(input_ids)
        return self._finalize("integrated_gradients", tokens, scores.detach().cpu().numpy(), pred, prob)

    # ---- Method D: Chefer gradient-weighted attention relevance (2021) ----
    def chefer_relevance(self, text: str) -> Attribution:
        """Chefer et al. (2021), 'Generic Attention-model Explainability'.
        At each layer, (attention * d logit / d attention)+ is averaged over heads
        and rolled out. This avoids the propagation problems of vanilla LRP in
        transformers."""
        enc = self._encode(text)
        out = self.model(**enc, output_attentions=True)
        probs = torch.softmax(out.logits, -1)[0]
        pred = int(probs.argmax()); prob = float(probs[pred])
        logit = out.logits[0, pred]
        grads = torch.autograd.grad(logit, out.attentions, retain_graph=False)

        seq = out.attentions[0].size(-1)
        R = torch.eye(seq, device=out.attentions[0].device)
        for A, G in zip(out.attentions, grads):
            cam = (G[0] * A[0]).clamp(min=0).mean(0)     # (seq, seq), positive contribution, head-averaged
            R = R + cam @ R                              # residual rollout
        rel = R[0]                                       # [CLS] row
        tokens = self._tokens(enc["input_ids"])
        return self._finalize("chefer_relevance", tokens, rel.detach().cpu().numpy(), pred, prob)

    def all_methods(self, text: str):
        """Return all four attributions in one call."""
        return {a.method: a for a in [
            self.raw_attention(text), self.attention_rollout(text),
            self.integrated_gradients(text), self.chefer_relevance(text)]}
