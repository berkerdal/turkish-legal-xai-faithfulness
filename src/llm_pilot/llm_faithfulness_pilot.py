"""
LLM self-explanation faithfulness pilot.

Extends the encoder-level faithfulness study to a generative model: an open
instruct LLM classifies each Turkish Constitutional Court decision and states the
phrases it relied on, and the faithfulness of that self-explanation is measured
with the same ERASER-style comprehensiveness and sufficiency metrics (two
operators, random baseline) used for the BERT explanation methods. Outputs are
written in a form that lines up with the existing result tables so the two
settings are directly comparable.

Designed to run on a single Kaggle GPU (T4/P100, 16 GB) with 4-bit inference.
Run in SMOKE mode first (N_INSTANCES small, one model) to validate the
environment, then scale up.

Required inputs (upload to the Kaggle notebook as datasets, or set paths below):
  - the frozen dataset snapshot  turkish_constitutional_court_decisions.csv.gz
  - the evaluated-instance list   sample_ids.csv   (id, stratum, label, prediction, confidence)

All outputs are written under OUT_DIR and zipped to llm_pilot_outputs.zip so the
run does not have to be repeated.
"""

import os
import re
import gc
import json
import glob
import random
import shutil
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

CONFIG = {
    # Models (all verified public Hugging Face ids). Comment any out to skip.
    "models": [
        "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1",   # focus model (competent classifier)
        # "Qwen/Qwen2.5-7B-Instruct",                      # general multilingual (zero-shot echoes the allegation)
        # "WiroAI/wiroai-turkish-llm-9b",                  # Turkish SFT on Gemma-2 (also echoes the allegation)
    ],
    "smoke": False,         # True -> tiny run (2 instances) to validate the pipeline first
    "n_instances_smoke": 2,
    "n_instances_full": 20,     # raise to 60 for the full stratified sample
    "n_models_smoke": 1,        # only the first model in smoke mode
    "max_input_tokens": 512,    # parity with the BERT head-512 truncation
    "max_rationale_phrases": 5,
    "max_phrase_chars": 250,    # per-sentence rationale cap (drop degenerate blobs)
    "n_random_reps": 5,         # random-baseline repetitions
    "seed": 42,

    # Label surface forms scored as the two classes.
    "label_texts": {"violation": "İHLAL VAR", "no_violation": "İHLAL YOK"},
    # Maps the dataset `labels` value to the class name above.
    "label_map": {1: "violation", 0: "no_violation"},

    # Input locations. The script searches /kaggle/input recursively if these are
    # left as basenames.
    "dataset_csv": "turkish_constitutional_court_decisions.csv",
    "sample_ids_csv": "sample_ids.csv",
    "text_column": "text",
    "label_column": "labels",
    "split_column": "split",     # set to None if the CSV has no split column
    "test_split_value": "test",

    # sample_ids.csv column names (this file uses: idx, stratum, true, pred, confidence)
    "sample_id_col": "idx",
    "sample_stratum_col": "stratum",
    "sample_true_col": "true",
    "sample_pred_col": "pred",
    "sample_conf_col": "confidence",

    # Path (or Kaggle dataset) of a LoRA adapter to apply to the first model.
    # Leave None for the base model; set to the fine-tuned adapter for the
    # competent-classifier faithfulness run.
    "adapter_path": None,

    "out_dir": "/kaggle/working/llm_pilot_out",
}

TURKISH_STOPWORDS = {
    "ve", "bir", "bu", "ile", "için", "da", "de", "ki", "ya", "ama", "veya",
    "en", "gibi", "kadar", "daha", "çok", "olarak", "olan", "ise", "ancak",
}

# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_input(name):
    """Return an existing path for `name`, searching /kaggle/input if needed.
    Tolerant to the .csv / .csv.gz extension difference (Kaggle may decompress)."""
    base = os.path.basename(name)
    candidates = [base]
    if base.endswith(".gz"):
        candidates.append(base[:-3])
    elif base.endswith(".csv"):
        candidates.append(base + ".gz")
    for c in [name] + candidates:
        if os.path.exists(c):
            return c
    for c in candidates:
        hits = glob.glob(os.path.join("/kaggle/input", "**", c), recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"Could not find '{name}' (tried {candidates}). "
        "Upload it as a Kaggle dataset or set the path in CONFIG."
    )


def token_type(token):
    t = token.strip()
    if not t:
        return None
    if all(not ch.isalnum() for ch in t):
        return "punctuation"
    if re.fullmatch(r"\d+([.,]\d+)*", t):
        return "number"
    if t.lower() in TURKISH_STOPWORDS:
        return "stopword"
    return "content"


def token_type_shares(text):
    counts = {"content": 0, "punctuation": 0, "number": 0, "stopword": 0}
    total = 0
    for tok in re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE):
        tt = token_type(tok)
        if tt is None:
            continue
        counts[tt] += 1
        total += 1
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: 100.0 * v / total for k, v in counts.items()}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #

def load_data(cfg):
    df = pd.read_csv(find_input(cfg["dataset_csv"]))
    if cfg["split_column"] and cfg["split_column"] in df.columns:
        df = df[df[cfg["split_column"]] == cfg["test_split_value"]].reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    sample = pd.read_csv(find_input(cfg["sample_ids_csv"]))
    # the id column is the 0-based row index into the ordered test split.
    rows = []
    for _, s in sample.iterrows():
        idx = int(s[cfg["sample_id_col"]])
        if idx < 0 or idx >= len(df):
            raise IndexError(f"sample id {idx} out of range for test split of size {len(df)}")
        try:
            bert_pred = cfg["label_map"][int(s[cfg["sample_pred_col"]])]
        except (ValueError, TypeError, KeyError):
            bert_pred = str(s.get(cfg["sample_pred_col"], ""))
        rows.append({
            "id": idx,
            "stratum": s.get(cfg["sample_stratum_col"], ""),
            "gold": cfg["label_map"][int(df.loc[idx, cfg["label_column"]])],
            "bert_pred": bert_pred,
            "bert_conf": s.get(cfg["sample_conf_col"], ""),
            "sample_label": s.get(cfg["sample_true_col"], None),
            "text": str(df.loc[idx, cfg["text_column"]]),
        })
    out = pd.DataFrame(rows)

    # Self-check: gold labels reconstructed from the dataset must match the
    # labels recorded in sample_ids.csv. A mismatch means the id join is wrong.
    if "sample_label" in out.columns and out["sample_label"].notna().all():
        try:
            recon = out["gold"].map({v: k for k, v in cfg["label_map"].items()})
            mism = (recon.astype(int) != out["sample_label"].astype(int)).sum()
            print(f"[data] id-join self-check: {mism} label mismatches out of {len(out)}")
            assert mism == 0, "id join produced label mismatches; check the id scheme"
        except (ValueError, TypeError):
            print("[data] id-join self-check skipped (non-integer sample labels)")
    print(f"[data] loaded {len(out)} evaluated instances")
    return out


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #

def load_model(model_id, cfg):
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    print(f"[model] loading {model_id}")
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto", torch_dtype=torch.float16,
    )
    adapter = cfg.get("adapter_path")
    if adapter:
        from peft import PeftModel
        if not os.path.isdir(adapter):
            hits = glob.glob("/kaggle/input/**/adapter_config.json", recursive=True)
            adapter = os.path.dirname(hits[0]) if hits else adapter
        model = PeftModel.from_pretrained(model, adapter)
        print(f"[model] applied LoRA adapter from {adapter}")
    model.eval()
    return tok, model


def truncate_to_tokens(tok, text, max_tokens):
    ids = tok(text, add_special_tokens=False).input_ids[:max_tokens]
    return tok.decode(ids)


def encode_chat(tok, model, user_content):
    """Return input_ids for a one-turn user prompt, robust across transformers
    versions (apply_chat_template may return a tensor or a BatchEncoding)."""
    enc = tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )
    return enc["input_ids"].to(model.device)


@torch.no_grad()
def continuation_logprob(model, tok, prompt_text, continuation_text):
    """Length-normalized log-probability (mean over continuation tokens) of
    `continuation_text` following `prompt_text`. Length normalization avoids the
    bias that a raw sum introduces when the two class labels tokenize to a
    different number of tokens under a given tokenizer."""
    prompt_ids = encode_chat(tok, model, prompt_text)
    cont_ids = tok(continuation_text, add_special_tokens=False,
                   return_tensors="pt").input_ids.to(model.device)
    input_ids = torch.cat([prompt_ids, cont_ids], dim=1)
    logits = model(input_ids).logits[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    p_len = prompt_ids.shape[1]
    n = cont_ids.shape[1]
    total = 0.0
    for i in range(n):
        pos = p_len + i - 1
        tok_id = input_ids[0, p_len + i]
        total += logprobs[pos, tok_id].item()
    return total / max(n, 1)


@torch.no_grad()
def class_probs(model, tok, text, cfg):
    """Two-way class probability scored at the first token where the two label
    strings diverge, at the same sequence position. This is unbiased with respect
    to label token length (unlike a summed score) and preserves full dynamic range
    (unlike a length-averaged score), which matters for the faithfulness deltas."""
    prompt = (
        f"{text}\n\n"
        "Yukarıdaki Anayasa Mahkemesi kararına göre, başvuruda bir temel hak "
        "ihlali tespit edilmiş midir? Cevap yalnızca \"İHLAL VAR\" veya "
        "\"İHLAL YOK\" olsun."
    )
    prompt_ids = encode_chat(tok, model, prompt)
    keys = list(cfg["label_texts"])
    lab_ids = {c: tok(" " + cfg["label_texts"][c], add_special_tokens=False).input_ids
               for c in keys}
    a, b = lab_ids[keys[0]], lab_ids[keys[1]]
    j = 0
    while j < len(a) and j < len(b) and a[j] == b[j]:
        j += 1
    if a[:j]:
        shared = torch.tensor([a[:j]], device=model.device)
        inp = torch.cat([prompt_ids, shared], dim=1)
    else:
        inp = prompt_ids
    logits = model(inp).logits[0, -1]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    ta = a[j] if j < len(a) else tok.eos_token_id
    tb = b[j] if j < len(b) else tok.eos_token_id
    la, lb = logprobs[ta].item(), logprobs[tb].item()
    m = max(la, lb)
    ea, eb = np.exp(la - m), np.exp(lb - m)
    s = ea + eb
    return {keys[0]: float(ea / s), keys[1]: float(eb / s)}


def stop_token_ids(tok):
    ids = []
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    for t in ("<|eot_id|>", "<|im_end|>", "<end_of_turn>"):
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0 and tid != tok.unk_token_id:
            ids.append(tid)
    return list(dict.fromkeys(ids))


@torch.no_grad()
def generate_rationale(model, tok, text, cfg):
    prompt = (
        f"{text}\n\n"
        "Yukarıdaki Anayasa Mahkemesi kararında başvuruda temel hak ihlali olup "
        "olmadığına karar verdin. Bu kararının gerekçesini, karardaki SOMUT "
        "OLGULARA ve ifadelere atıfla en fazla 3 kısa cümleyle açıkla."
    )
    ids = encode_chat(tok, model, prompt)
    out = model.generate(ids, max_new_tokens=160, do_sample=False,
                         pad_token_id=tok.pad_token_id, eos_token_id=stop_token_ids(tok))
    completion = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    # remove any leaked role header the model may emit after its turn
    completion = re.split(r"assistant\b|<\|", completion)[0]
    return completion.strip()


@torch.no_grad()
def generate_selection(model, tok, text, sent_spans, cfg):
    """Extractive self-rationale: present the input as numbered sentences and ask
    the model which ones its decision rests on, plus a one-line justification.
    Returns (selected_sentence_indices, justification, raw_completion)."""
    numbered = "\n".join(f"[{i + 1}] {text[s:e].strip()}"
                         for i, (s, e) in enumerate(sent_spans))
    prompt = (
        "Aşağıda bir Anayasa Mahkemesi kararının cümleleri numaralandırılmıştır:\n\n"
        f"{numbered}\n\n"
        "Soru: Başvuruda temel hak ihlali var mı, yok mu? Ve kararını EN ÇOK hangi "
        f"cümlelere dayandırıyorsun (en fazla {cfg['max_rationale_phrases']} numara)? "
        "Yanıtı tam olarak şu biçimde ver:\n"
        "KARAR: <VAR veya YOK>\n"
        "CÜMLELER: <numaralar, virgülle>\n"
        "GEREKÇE: <tek cümle>"
    )
    ids = encode_chat(tok, model, prompt)
    out = model.generate(ids, max_new_tokens=120, do_sample=False,
                         pad_token_id=tok.pad_token_id, eos_token_id=stop_token_ids(tok))
    completion = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    completion = re.split(r"assistant\b|<\|", completion)[0].strip()
    m = re.search(r"C[ÜU]MLELER\s*:?([^\n]*)", completion, flags=re.IGNORECASE)
    nums_str = m.group(1) if m else completion
    idxs = []
    for num in re.findall(r"\d+", nums_str):
        k = int(num) - 1
        if 0 <= k < len(sent_spans) and k not in idxs:
            idxs.append(k)
    km = re.search(r"KARAR\s*:?([^\n]*)", completion, flags=re.IGNORECASE)
    decision = None
    if km:
        line = km.group(1).upper()
        if "YOK" in line:
            decision = "no_violation"
        elif "VAR" in line:
            decision = "violation"
    jm = re.search(r"GEREK[ÇC]E\s*:?(.*)", completion, flags=re.IGNORECASE | re.DOTALL)
    justification = (jm.group(1).strip() if jm else "")[:300]
    return decision, sorted(idxs[: cfg["max_rationale_phrases"]]), justification, completion


def parse_rationale(completion, cfg):
    """Split the model's free-form rationale into sentence-level phrases; matching
    to the input sentences is handled separately."""
    body = completion
    m = re.search(r"GEREK[ÇC]E\s*:?(.*)", completion, flags=re.IGNORECASE | re.DOTALL)
    if m:
        body = m.group(1)
    phrases = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", body):
        p = re.sub(r'^[\s\-\*•\d\.\)"“”\']+', "", part).strip(' "“”\'\t')
        if not p or len(p) < 8 or len(p) > cfg["max_phrase_chars"]:
            continue
        if p.upper().startswith(("KARAR", "GEREK", "ÖRNEK", "ASSISTANT", "BİÇİM", "KURAL")):
            continue
        phrases.append(p)
    return phrases[: cfg["max_rationale_phrases"]]


# --------------------------------------------------------------------------- #
# Faithfulness
# --------------------------------------------------------------------------- #

def _normalize_with_map(text):
    """Collapse whitespace runs to a single space; return the normalized string
    and a map from normalized index to original index."""
    chars, idx_map, prev_space = [], [], False
    for i, ch in enumerate(text):
        if ch.isspace():
            if prev_space:
                continue
            chars.append(" "); idx_map.append(i); prev_space = True
        else:
            chars.append(ch); idx_map.append(i); prev_space = False
    idx_map.append(len(text))
    return "".join(chars), idx_map


def phrase_spans(text, phrases):
    """Locate each cited phrase in the text with whitespace-insensitive matching,
    mapping matches back to original character offsets."""
    norm, idx_map = _normalize_with_map(text)
    norm_l = norm.casefold()
    spans = []
    for p in phrases:
        pn = re.sub(r"\s+", " ", p).strip().casefold()
        if len(pn) < 3:
            continue
        pos = norm_l.find(pn)
        if pos >= 0:
            spans.append((idx_map[pos], idx_map[pos + len(pn)]))
    return merge_spans(spans)


def merge_spans(spans):
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def apply_operator(text, spans, keep, operator):
    """keep=False -> remove the spans (comprehensiveness); keep=True -> keep only
    the spans (sufficiency). operator in {delete, mask}."""
    if keep:
        pieces = [text[s:e] for s, e in spans]
        return " ".join(pieces) if operator == "delete" else \
            "[...] " + " [...] ".join(pieces) + " [...]"
    out, prev = [], 0
    for s, e in spans:
        out.append(text[prev:s])
        if operator == "mask":
            out.append(" [...] ")
        prev = e
    out.append(text[prev:])
    return "".join(out)


def faithfulness(model, tok, text, spans, pred_class, base_prob, cfg):
    res = {}
    for operator in ("delete", "mask"):
        p_removed = class_probs(model, tok, apply_operator(text, spans, False, operator), cfg)[pred_class]
        p_kept = class_probs(model, tok, apply_operator(text, spans, True, operator), cfg)[pred_class]
        res[f"comp_{operator}"] = base_prob - p_removed
        res[f"suff_{operator}"] = base_prob - p_kept
    return res


def random_spans(text, total_len, seed):
    rng = random.Random(seed)
    words = list(re.finditer(r"\S+", text))
    if not words:
        return []
    picked, acc = [], 0
    order = list(range(len(words)))
    rng.shuffle(order)
    for i in order:
        w = words[i]
        picked.append((w.start(), w.end()))
        acc += w.end() - w.start()
        if acc >= total_len:
            break
    return merge_spans(picked)


def random_baseline(model, tok, text, total_len, pred_class, base_prob, cfg):
    acc = {"comp_delete": [], "comp_mask": [], "suff_delete": [], "suff_mask": []}
    for r in range(cfg["n_random_reps"]):
        spans = random_spans(text, total_len, cfg["seed"] + r)
        if not spans:
            continue
        f = faithfulness(model, tok, text, spans, pred_class, base_prob, cfg)
        for k in acc:
            acc[k].append(f[k])
    return {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #

def macro_f1(gold, pred, classes):
    f1s = []
    for c in classes:
        tp = sum((g == c and p == c) for g, p in zip(gold, pred))
        fp = sum((g != c and p == c) for g, p in zip(gold, pred))
        fn = sum((g == c and p != c) for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s))


_SENT_RE = re.compile(r"[^.!?]+[.!?]?", re.DOTALL)


def split_sentences(text):
    spans = []
    for m in _SENT_RE.finditer(text):
        if text[m.start():m.end()].strip():
            spans.append((m.start(), m.end()))
    return spans


def _tokset(s):
    toks = re.findall(r"\w+", s.casefold(), flags=re.UNICODE)
    return set(t for t in toks if len(t) > 2 and t not in TURKISH_STOPWORDS)


def match_rationale_to_sentences(phrases, text, sent_spans, thr=0.34):
    """Map each paraphrased rationale phrase to the input sentence that contains
    the largest fraction of the phrase's content words."""
    sent_toks = [_tokset(text[s:e]) for s, e in sent_spans]
    cited = set()
    for p in phrases:
        pt = _tokset(p)
        if len(pt) < 2:
            continue
        best_i, best_score = -1, 0.0
        for i, st in enumerate(sent_toks):
            if not st:
                continue
            score = len(pt & st) / len(pt)   # fraction of phrase content words in sentence
            if score > best_score:
                best_score, best_i = score, i
        if best_i >= 0 and best_score >= thr:
            cited.add(best_i)
    return sorted(cited)


def occlusion_importance(model, tok, text, sent_spans, pred_class, base_prob, cfg):
    """Leave-one-out importance per sentence: probability drop when it is removed."""
    imp = []
    for s, e in sent_spans:
        perturbed = (text[:s] + " " + text[e:]).strip()
        p = class_probs(model, tok, perturbed, cfg)[pred_class]
        imp.append(base_prob - p)
    return imp


def run_model(model_id, data, cfg, out_dir):
    tok, model = load_model(model_id, cfg)
    per_rows, raw_rows = [], []
    for _, row in data.iterrows():
        text = truncate_to_tokens(tok, row["text"], cfg["max_input_tokens"])
        sent_spans = split_sentences(text)

        # (1) one generation gives both the decision and the extractive self-rationale,
        # so the rationale is coupled to the answer whose faithfulness we test.
        stated, cited, justification, completion = generate_selection(model, tok, text, sent_spans, cfg)
        probs = class_probs(model, tok, text, cfg)
        scored = max(probs, key=probs.get)
        pred_class = stated if stated in probs else scored
        base_prob = probs[pred_class]
        self_spans = merge_spans([sent_spans[i] for i in cited])

        # (2) occlusion attribution: importance = drop when a sentence is removed
        imp = occlusion_importance(model, tok, text, sent_spans, pred_class, base_prob, cfg)
        k = len(cited) if cited else max(1, round(0.2 * len(sent_spans)))
        occ_idx = sorted(range(len(imp)), key=lambda i: imp[i], reverse=True)[:k]
        occ_spans = merge_spans([sent_spans[i] for i in occ_idx])

        if self_spans:
            self_ff = faithfulness(model, tok, text, self_spans, pred_class, base_prob, cfg)
        else:
            self_ff = {kk: float("nan") for kk in
                       ("comp_delete", "comp_mask", "suff_delete", "suff_mask")}
        occ_ff = faithfulness(model, tok, text, occ_spans, pred_class, base_prob, cfg)
        occ_len = sum(e - s for s, e in occ_spans)
        rand = random_baseline(model, tok, text, occ_len, pred_class, base_prob, cfg)

        self_text = " ".join(text[s:e] for s, e in self_spans)
        if cfg.get("smoke"):
            sc = self_ff["comp_delete"]
            print(f"  [smoke] id={row['id']} pred={pred_class} sents={len(sent_spans)} "
                  f"selected={cited} | self_comp(del)={sc} "
                  f"occ_comp(del)={occ_ff['comp_delete']:.3f} rand={rand['comp_delete']}")
            print(f"           justification={justification[:100]}")

        per_rows.append({
            "id": row["id"], "stratum": row["stratum"], "gold": row["gold"],
            "bert_pred": row["bert_pred"], "bert_conf": row["bert_conf"],
            "llm_pred": pred_class, "llm_pconf": round(base_prob, 4),
            "correct": int(pred_class == row["gold"]),
            "stated_decision": stated, "decision_consistency": int(stated == scored),
            "n_sentences": len(sent_spans), "n_cited_sentences": len(cited),
            "rationale_frac": round(sum(e - s for s, e in self_spans) / max(len(text), 1), 4),
            "rationale_text": self_text,
            **{f"self_{k}": round(v, 4) if v == v else v for k, v in self_ff.items()},
            **{f"occ_{k}": round(v, 4) if v == v else v for k, v in occ_ff.items()},
            **{f"rand_{k}": round(v, 4) if v == v else v for k, v in rand.items()},
        })
        raw_rows.append({
            "id": row["id"], "model": model_id, "completion": completion,
            "justification": justification, "cited_sentences": cited, "class_probs": probs,
        })

    per = pd.DataFrame(per_rows)
    classes = list(cfg["label_texts"])
    agg = {
        "model": model_id,
        "n": len(per),
        "accuracy": float((per["llm_pred"] == per["gold"]).mean()),
        "macro_f1": macro_f1(per["gold"].tolist(), per["llm_pred"].tolist(), classes),
        "bert_agreement": float((per["llm_pred"].astype(str) == per["bert_pred"].astype(str)).mean()),
        "accuracy_by_stratum": {
            s: float((g["llm_pred"] == g["gold"]).mean())
            for s, g in per.groupby("stratum")
        },
        "self_comp_delete": float(per["self_comp_delete"].mean(skipna=True)),
        "self_comp_mask": float(per["self_comp_mask"].mean(skipna=True)),
        "self_suff_delete": float(per["self_suff_delete"].mean(skipna=True)),
        "occ_comp_delete": float(per["occ_comp_delete"].mean(skipna=True)),
        "occ_comp_mask": float(per["occ_comp_mask"].mean(skipna=True)),
        "occ_suff_delete": float(per["occ_suff_delete"].mean(skipna=True)),
        "rand_comp_delete": float(per["rand_comp_delete"].mean(skipna=True)),
        "rand_comp_mask": float(per["rand_comp_mask"].mean(skipna=True)),
        "rand_suff_delete": float(per["rand_suff_delete"].mean(skipna=True)),
        "mean_rationale_frac": float(per["rationale_frac"].mean(skipna=True)),
        "mean_cited_sentences": float(per["n_cited_sentences"].mean(skipna=True)),
        "decision_consistency": float(per["decision_consistency"].mean()),
        "pred_distribution": per["llm_pred"].value_counts().to_dict(),
        "rationale_token_types": token_type_shares(" ".join(per["rationale_text"].fillna("").tolist())),
        "self_vs_random_signal": float(
            per["self_comp_delete"].mean(skipna=True) - per["rand_comp_delete"].mean(skipna=True)),
        "occ_vs_random_signal": float(
            per["occ_comp_delete"].mean(skipna=True) - per["rand_comp_delete"].mean(skipna=True)),
    }

    safe = model_id.replace("/", "__")
    per.to_csv(os.path.join(out_dir, f"perinstance_{safe}.csv"), index=False)
    with open(os.path.join(out_dir, f"raw_{safe}.jsonl"), "w", encoding="utf-8") as fh:
        for r in raw_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    del model, tok
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return per, agg


def main():
    cfg = CONFIG
    set_seed(cfg["seed"])
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    data = load_data(cfg)
    n = cfg["n_instances_smoke"] if cfg["smoke"] else cfg["n_instances_full"]
    # keep strata balanced when subsetting
    per_stratum = max(1, n // max(data["stratum"].nunique(), 1))
    data = (data.groupby("stratum", group_keys=False)
                .head(per_stratum)).head(n).reset_index(drop=True)
    print(f"[run] evaluating {len(data)} instances")

    models = cfg["models"][: cfg["n_models_smoke"]] if cfg["smoke"] else cfg["models"]
    aggregates, qualit = [], []
    for mid in models:
        try:
            per, agg = run_model(mid, data, cfg, out_dir)
            aggregates.append(agg)
            for _, r in per.head(5).iterrows():
                qualit.append((mid, r))
            print(f"[run] {mid}: acc={agg['accuracy']:.3f} | "
                  f"self_comp={agg['self_comp_delete']:.3f} "
                  f"occ_comp={agg['occ_comp_delete']:.3f} "
                  f"rand={agg['rand_comp_delete']:.3f}")
        except Exception as exc:  # keep going if one model fails to load
            print(f"[run] FAILED {mid}: {exc}")
            aggregates.append({"model": mid, "error": str(exc)})

    with open(os.path.join(out_dir, "aggregate.json"), "w", encoding="utf-8") as fh:
        json.dump(aggregates, fh, ensure_ascii=False, indent=2)

    tt_rows = [{"model": a["model"], **a["rationale_token_types"]}
               for a in aggregates if "rationale_token_types" in a]
    if tt_rows:
        pd.DataFrame(tt_rows).to_csv(os.path.join(out_dir, "rationale_token_types.csv"), index=False)

    with open(os.path.join(out_dir, "qualitative.md"), "w", encoding="utf-8") as fh:
        fh.write("# Qualitative examples (LLM prediction + self-explanation)\n\n")
        for mid, r in qualit:
            fh.write(f"## {mid} — instance {r['id']} (stratum: {r['stratum']})\n")
            fh.write(f"- gold: {r['gold']} | BERT: {r['bert_pred']} | "
                     f"LLM: {r['llm_pred']} (p={r['llm_pconf']})\n")
            fh.write(f"- rationale: {r['rationale_text']}\n\n")

    with open(os.path.join(out_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "env.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"timestamp_utc: {datetime.now(timezone.utc).isoformat()}\n")
        fh.write(f"python: {platform.python_version()}\n")
        fh.write(f"torch: {torch.__version__}\n")
        fh.write(f"cuda: {torch.cuda.is_available()} "
                 f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}\n")
    os.system(f"pip freeze > {os.path.join(out_dir, 'pip_freeze.txt')}")

    zip_base = "/kaggle/working/llm_pilot_outputs"
    shutil.make_archive(zip_base, "zip", out_dir)
    print("\n[done] outputs in", out_dir)
    print("[done] download:", zip_base + ".zip")
    print("\n=== SUMMARY ===")
    for a in aggregates:
        if "error" in a:
            print(f"{a['model']}: ERROR {a['error']}")
        else:
            print(f"{a['model']}: acc={a['accuracy']:.3f} f1={a['macro_f1']:.3f} "
                  f"cons={a['decision_consistency']:.2f} preds={a['pred_distribution']} | "
                  f"self_comp(del)={a['self_comp_delete']:.3f} "
                  f"occ_comp(del)={a['occ_comp_delete']:.3f} "
                  f"rand={a['rand_comp_delete']:.3f} | "
                  f"self_signal={a['self_vs_random_signal']:+.3f} "
                  f"occ_signal={a['occ_vs_random_signal']:+.3f}")
    print("\nBERT reference (existing study): macro-F1 0.797; "
          "comp[MASK] IG 0.240 / Chefer 0.239 / raw-attn 0.106 / rollout 0.079 / random 0.053")


if __name__ == "__main__":
    main()
