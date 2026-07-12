"""Faithfulness metrics (DeYoung et al., ERASER 2020).

comprehensiveness: mask the top k% most important tokens and measure how much the
    predicted-class probability drops (higher = more faithful).
sufficiency:       keep only the top k% and mask the rest, and measure how much the
    probability changes (lower = more faithful).

AOPC is the mean of (original - perturbed) probability over the bins. As an
out-of-distribution control, tokens are replaced with [MASK] rather than deleted:
this stays close to the BERT MLM distribution and keeps the sequence length fixed.
"""
import numpy as np
import torch

BINS = (0.01, 0.05, 0.10, 0.20, 0.50)


def _content_positions(attribution):
    """Non-special positions, ordered by decreasing score."""
    toks = attribution.tokens
    idx = [i for i, t in enumerate(toks) if t not in ("[CLS]", "[SEP]", "[PAD]")]
    idx = np.array(idx)
    order = idx[np.argsort(-attribution.scores[idx])]
    return order


def _remove(exp, input_ids, attn, positions, operator, mask_id):
    """Remove `positions` according to `operator`: mask replaces them with [MASK];
    delete drops them from the sequence."""
    if operator == "mask":
        ids = input_ids.clone()
        if len(positions):
            ids[0, list(positions)] = mask_id
        return ids, attn
    elif operator == "delete":
        rem = set(int(p) for p in positions)
        keep = [i for i in range(input_ids.size(1)) if i not in rem]
        ids = input_ids[0, keep].unsqueeze(0)
        return ids, torch.ones_like(ids)
    raise ValueError(operator)


def evaluate(exp, text, attribution, bins=BINS, operator="mask"):
    """Comprehensiveness and sufficiency AOPC for one instance and one method.
    operator: 'mask' (replace with [MASK]) or 'delete' (drop from the sequence,
    ERASER-canonical)."""
    enc = exp._encode(text)
    input_ids = enc["input_ids"]
    attn = enc["attention_mask"]
    target = attribution.pred
    mask_id = exp.tok.mask_token_id

    p_orig = exp.prob_from_ids(input_ids, attn, target)
    ranked = _content_positions(attribution)
    n = len(ranked)

    comp, suff = [], []
    for k in bins:
        m = max(1, int(round(k * n)))
        topk = ranked[:m]
        # comprehensiveness: remove the top-k
        ids_c, attn_c = _remove(exp, input_ids, attn, topk, operator, mask_id)
        p_c = exp.prob_from_ids(ids_c, attn_c, target)
        comp.append(p_orig - p_c)
        # sufficiency: keep only the top-k, remove the remaining content tokens
        keep = set(topk.tolist())
        drop = [pos for pos in ranked if pos not in keep]
        ids_s, attn_s = _remove(exp, input_ids, attn, drop, operator, mask_id)
        p_s = exp.prob_from_ids(ids_s, attn_s, target)
        suff.append(p_orig - p_s)

    return {
        "comprehensiveness": float(np.mean(comp)),   # higher is better
        "sufficiency": float(np.mean(suff)),          # lower is better
        "operator": operator,
        "p_orig": float(p_orig),
        "n_content": int(n),
    }
