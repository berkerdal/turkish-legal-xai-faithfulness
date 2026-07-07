"""Faz 4 — Faithfulness metrikleri (DeYoung/ERASER 2020).

comprehensiveness: en önemli %k token'ı [MASK]'la → hedef sınıf olasılığı NE KADAR DÜŞER?
    (yüksek = sadık: o tokenlar gerçekten önemliydi)
sufficiency:       yalnız en önemli %k token'ı TUT, gerisini [MASK]'la → olasılık ne kadar değişir?
    (düşük = sadık: o tokenlar tek başına kararı taşıyor)

AOPC = bin'ler üzerinde ortalama (orijinal - değiştirilmiş) olasılık.
OOD notu: silme yerine [MASK] ile değiştiriyoruz (BERT MLM dağılımına yakın, sekans uzunluğu sabit).
"""
import numpy as np
import torch

BINS = (0.01, 0.05, 0.10, 0.20, 0.50)


def _content_positions(attribution):
    """Özel token olmayan pozisyonlar (skorlarına göre azalan sırada)."""
    toks = attribution.tokens
    idx = [i for i, t in enumerate(toks) if t not in ("[CLS]", "[SEP]", "[PAD]")]
    idx = np.array(idx)
    order = idx[np.argsort(-attribution.scores[idx])]
    return order


def _remove(exp, input_ids, attn, positions, operator, mask_id):
    """positions'ı 'operator'e göre kaldır. mask=[MASK] ile değiştir; delete=diziden çıkar."""
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
    """Bir örnek + bir yöntem için comprehensiveness & sufficiency AOPC döndür.
    operator: 'mask' ([MASK] ile değiştir) veya 'delete' (diziden çıkar, ERASER-kanonik)."""
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
        # comprehensiveness: top-k'yı kaldır
        ids_c, attn_c = _remove(exp, input_ids, attn, topk, operator, mask_id)
        p_c = exp.prob_from_ids(ids_c, attn_c, target)
        comp.append(p_orig - p_c)
        # sufficiency: yalnız top-k'yı tut → geri kalan içerik tokenlarını kaldır
        keep = set(topk.tolist())
        drop = [pos for pos in ranked if pos not in keep]
        ids_s, attn_s = _remove(exp, input_ids, attn, drop, operator, mask_id)
        p_s = exp.prob_from_ids(ids_s, attn_s, target)
        suff.append(p_orig - p_s)

    return {
        "comprehensiveness": float(np.mean(comp)),   # yüksek = iyi
        "sufficiency": float(np.mean(suff)),          # düşük = iyi
        "operator": operator,
        "p_orig": float(p_orig),
        "n_content": int(n),
    }
