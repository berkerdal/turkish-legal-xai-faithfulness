"""Build the Kaggle notebooks (install cell + script cell) from the source scripts."""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
NB_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "notebooks"))


def lines(text):
    parts = text.splitlines(keepends=True)
    return parts if parts else [""]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": lines(text)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": lines(text)}


def build(script_name, out_name, intro, install):
    with open(os.path.join(HERE, script_name), encoding="utf-8") as fh:
        script = fh.read()
    nb = {
        "cells": [md(intro), code(install), code(script)],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    os.makedirs(NB_DIR, exist_ok=True)
    out = os.path.join(NB_DIR, out_name)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(nb, fh, ensure_ascii=False, indent=1)
    print("wrote", out)


build(
    "llm_faithfulness_pilot.py", "llm_faithfulness_pilot.ipynb",
    intro=(
        "# LLM self-explanation faithfulness\n\n"
        "Runs the (base or fine-tuned) LLM as a classifier on the stratified sample, "
        "collects its extractive self-rationale, and measures comprehensiveness/"
        "sufficiency against occlusion and a random baseline.\n\n"
        "**Inputs:** add `turkish_constitutional_court_decisions.csv[.gz]` and "
        "`sample_ids.csv` as Kaggle datasets. For the fine-tuned run, also add the "
        "LoRA adapter dataset and set `CONFIG['adapter_path']`.\n\n"
        "Enable GPU. Run the install cell, then the pilot cell."
    ),
    install="!pip -q install -U transformers accelerate bitsandbytes sentencepiece peft\n",
)

build(
    "llm_finetune.py", "llm_finetune_kaggle.ipynb",
    intro=(
        "# QLoRA fine-tuning: Turkish LLM classifier\n\n"
        "Fine-tunes the Turkish LLM (4-bit base + LoRA) on the rights-violation "
        "classification task so it becomes a competent classifier. Saves the adapter "
        "to `/kaggle/working/llm_adapter.zip`.\n\n"
        "**Input:** add `turkish_constitutional_court_decisions.csv[.gz]` as a Kaggle "
        "dataset. Enable GPU (T4 or P100).\n\n"
        "Run the install cell, then the training cell. It starts in SMOKE mode "
        "(64 examples) to validate; when that works, set `CONFIG['smoke'] = False`.\n\n"
        "After training, download `llm_adapter.zip`, upload it as a Kaggle dataset, "
        "and point the evaluation notebook's `CONFIG['adapter_path']` at it."
    ),
    install="!pip -q install -U transformers accelerate bitsandbytes peft datasets\n",
)
