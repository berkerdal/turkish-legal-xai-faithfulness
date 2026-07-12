"""
QLoRA fine-tuning of the Turkish LLM for the rights-violation classification task.

Purpose: make the decoder a competent classifier (comparable to the fine-tuned
BERT), so its self-explanation faithfulness can be studied without the zero-shot
class-prior confound. The base model is loaded in 4-bit; only LoRA adapters are
trained. The adapter is saved and zipped for download; upload it as a Kaggle
dataset and point the evaluation notebook's CONFIG["adapter_path"] at it.

Kaggle: single T4/P100 (16 GB). Run in SMOKE mode first (tiny subset, a few steps)
to validate the pipeline, then set CONFIG["smoke"] = False.

Required input: the frozen dataset snapshot (turkish_constitutional_court_decisions.csv[.gz]).
"""

import os
import re
import glob
import shutil

import numpy as np
import pandas as pd
import torch

CONFIG = {
    "model_id": "ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1",
    "smoke": True,
    "n_train_smoke": 64,
    "n_train_full": 4000,     # balanced subset of the training split
    "n_val": 300,             # held-out check that the model actually learned
    "max_input_tokens": 512,  # parity with the BERT head-512 setup
    "epochs": 1,
    "lr": 2e-4,
    "batch_size": 1,
    "grad_accum": 16,
    "seed": 42,

    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],

    "label_texts": {"violation": "İHLAL VAR", "no_violation": "İHLAL YOK"},
    "label_map": {1: "violation", 0: "no_violation"},

    "dataset_csv": "turkish_constitutional_court_decisions.csv",
    "text_column": "text",
    "label_column": "labels",
    "split_column": "split",

    "out_dir": "/kaggle/working/llm_adapter",
}

QUESTION = ("Yukarıdaki Anayasa Mahkemesi kararına göre, başvuruda bir temel hak "
            "ihlali tespit edilmiş midir? Cevap yalnızca \"İHLAL VAR\" veya "
            "\"İHLAL YOK\" olsun.")


def find_input(name):
    base = os.path.basename(name)
    cands = [base] + ([base[:-3]] if base.endswith(".gz") else [base + ".gz"])
    for c in [name] + cands:
        if os.path.exists(c):
            return c
    for c in cands:
        hits = glob.glob(os.path.join("/kaggle/input", "**", c), recursive=True)
        if hits:
            return hits[0]
    raise FileNotFoundError(f"Could not find {name} (tried {cands}).")


def build_examples(cfg, tok, df):
    """Tokenize each (prompt -> label) pair with the prompt tokens masked to -100
    so the loss is computed on the label only."""
    rows = []
    for _, r in df.iterrows():
        text = tok.decode(tok(str(r[cfg["text_column"]]), add_special_tokens=False)
                          .input_ids[: cfg["max_input_tokens"]])
        cls = cfg["label_map"][int(r[cfg["label_column"]])]
        target = " " + cfg["label_texts"][cls]
        msgs = [{"role": "user", "content": f"{text}\n\n{QUESTION}"}]
        prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                             return_dict=True)["input_ids"]
        target_ids = tok(target, add_special_tokens=False).input_ids + [tok.eos_token_id]
        input_ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        rows.append({"input_ids": input_ids, "labels": labels})
    return rows


class Collator:
    def __init__(self, tok):
        self.pad = tok.pad_token_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids, labs, att = [], [], []
        for b in batch:
            p = n - len(b["input_ids"])
            ids.append(b["input_ids"] + [self.pad] * p)
            labs.append(b["labels"] + [-100] * p)
            att.append([1] * len(b["input_ids"]) + [0] * p)
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labs),
                "attention_mask": torch.tensor(att)}


def load_data(cfg):
    df = pd.read_csv(find_input(cfg["dataset_csv"]))
    tr = df[df[cfg["split_column"]] == "train"]
    va = df[df[cfg["split_column"]] == "validation"]
    n_tr = cfg["n_train_smoke"] if cfg["smoke"] else cfg["n_train_full"]
    # balance the training subset across the two classes
    per = n_tr // 2
    pos = tr[tr[cfg["label_column"]] == 1].sample(min(per, (tr[cfg["label_column"]] == 1).sum()),
                                                  random_state=cfg["seed"])
    neg = tr[tr[cfg["label_column"]] == 0].sample(min(per, (tr[cfg["label_column"]] == 0).sum()),
                                                  random_state=cfg["seed"])
    train = pd.concat([pos, neg]).sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
    val = va.sample(min(cfg["n_val"], len(va)), random_state=cfg["seed"]).reset_index(drop=True)
    print(f"[data] train={len(train)} (bal) val={len(val)}")
    return train, val


@torch.no_grad()
def predict_label(model, tok, text, cfg):
    """First-divergent-token scoring of the two labels (unbiased, dynamic range)."""
    msgs = [{"role": "user", "content": f"{text}\n\n{QUESTION}"}]
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                         return_dict=True, return_tensors="pt")["input_ids"].to(model.device)
    keys = list(cfg["label_texts"])
    lab = {c: tok(" " + cfg["label_texts"][c], add_special_tokens=False).input_ids for c in keys}
    a, b = lab[keys[0]], lab[keys[1]]
    j = 0
    while j < len(a) and j < len(b) and a[j] == b[j]:
        j += 1
    inp = torch.cat([prompt_ids, torch.tensor([a[:j]], device=model.device)], dim=1) if a[:j] else prompt_ids
    lp = torch.log_softmax(model(inp).logits[0, -1].float(), dim=-1)
    la = lp[a[j] if j < len(a) else tok.eos_token_id].item()
    lb = lp[b[j] if j < len(b) else tok.eos_token_id].item()
    return keys[0] if la >= lb else keys[1]


def evaluate(model, tok, val, cfg):
    model.eval()
    correct, preds = 0, {"violation": 0, "no_violation": 0}
    for _, r in val.iterrows():
        text = tok.decode(tok(str(r[cfg["text_column"]]), add_special_tokens=False)
                          .input_ids[: cfg["max_input_tokens"]])
        p = predict_label(model, tok, text, cfg)
        preds[p] += 1
        if p == cfg["label_map"][int(r[cfg["label_column"]])]:
            correct += 1
    acc = correct / max(len(val), 1)
    print(f"[eval] val accuracy={acc:.3f} pred_distribution={preds}")
    return acc, preds


def main():
    from transformers import (AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig,
                              Trainer, TrainingArguments)
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset

    cfg = CONFIG
    torch.manual_seed(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["model_id"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    train_df, val_df = load_data(cfg)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], quantization_config=bnb,
                                                 device_map="auto", torch_dtype=torch.float16)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(
        r=cfg["lora_r"], lora_alpha=cfg["lora_alpha"], lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_targets"], bias="none", task_type="CAUSAL_LM"))
    model.print_trainable_parameters()

    ds = Dataset.from_list(build_examples(cfg, tok, train_df))
    args = TrainingArguments(
        output_dir="/kaggle/working/_trainer", per_device_train_batch_size=cfg["batch_size"],
        gradient_accumulation_steps=cfg["grad_accum"], num_train_epochs=cfg["epochs"],
        learning_rate=cfg["lr"], warmup_ratio=0.03, lr_scheduler_type="cosine",
        logging_steps=10, save_strategy="no", fp16=True, optim="paged_adamw_8bit",
        gradient_checkpointing=True, report_to="none", seed=cfg["seed"])
    trainer = Trainer(model=model, args=args, train_dataset=ds, data_collator=Collator(tok))
    model.config.use_cache = False
    trainer.train()

    evaluate(model, tok, val_df, cfg)

    os.makedirs(cfg["out_dir"], exist_ok=True)
    model.save_pretrained(cfg["out_dir"])
    tok.save_pretrained(cfg["out_dir"])
    shutil.make_archive("/kaggle/working/llm_adapter", "zip", cfg["out_dir"])
    print("[done] adapter saved and zipped -> /kaggle/working/llm_adapter.zip")
    print("Upload it as a Kaggle dataset, then set CONFIG['adapter_path'] in the eval notebook.")


if __name__ == "__main__":
    main()
