# Generative-model self-explanation faithfulness (RQ4)

This module extends the encoder-level faithfulness study to a generative model. A
Turkish legal large language model, fine-tuned to the rights-violation
classification task, classifies each Constitutional Court decision and lists the
input sentences its decision rests on. The faithfulness of that self-explanation
is measured with the same ERASER-style comprehensiveness and sufficiency metrics
(two deletion operators, random baseline) used for the BERT explanation methods,
on the same stratified sample.

## Files

- `llm_finetune.py` — QLoRA fine-tuning of the base model into a competent,
  class-balanced classifier. The base model is loaded in 4-bit and only LoRA
  adapters are trained; the adapter is saved as a zip.
- `llm_faithfulness_pilot.py` — the evaluation. It loads the fine-tuned model
  (base plus LoRA adapter), elicits the decision and the extractive
  self-explanation in a single generation so the two are coupled, computes an
  occlusion reference and a size-matched random baseline, and measures
  comprehensiveness and sufficiency under both operators.
- `build_pilot_notebook.py` — regenerates the Kaggle notebooks from these scripts.
- `../xai/make_llm_fig_en.py` — builds the main-text figure from the per-instance
  results.

## Reproducing on Kaggle (single GPU)

1. **Fine-tune.** New GPU notebook (T4 or P100); add the dataset as a Kaggle
   dataset and run `notebooks/llm_finetune_kaggle.ipynb`. It writes
   `llm_adapter.zip`.
2. **Evaluate.** Upload `llm_adapter.zip` as a Kaggle dataset, point
   `CONFIG["adapter_path"]` at it, and run
   `notebooks/llm_faithfulness_pilot.ipynb`. It writes `llm_pilot_outputs.zip`.
3. **Figure.** Run `make_llm_fig_en.py` on `results/llm_perinstance.csv` to
   produce `results/fig_llm_en.png`.

The evaluation reuses the stratified 60-instance sample from the encoder study
(`results/sample_ids.csv`), so the generative and encoder results are measured on
the same instances.

## Outputs and where they appear in the paper

| Output | Content | In the paper |
|---|---|---|
| `perinstance_<model>.csv` | per-instance prediction, self-explanation, comprehensiveness/sufficiency (both operators), random baseline | RQ4 table and figure |
| `aggregate.json` | accuracy, per-stratum accuracy, mean comprehensiveness/sufficiency against random | RQ4 table, Table S8 |
| `rationale_token_types.csv` | content/punctuation/subword/stopword shares of the self-explanation | content-word share reported in the text |
| `qualitative.md` | prediction and self-explanation per instance | qualitative discussion |
| `raw_<model>.jsonl`, `config.json`, `env.txt`, `pip_freeze.txt` | full provenance and run environment | reproducibility |

## Notes

- The self-explanation is a discrete sentence set, so its comprehensiveness and
  sufficiency are set-based (remove or keep the selected sentences), evaluated
  against a size-matched occlusion reference and random baseline; the encoder
  methods use AOPC curves over top-k fractions. Both follow the ERASER
  definitions.
- The predicted-class probability is read from the two label strings at the first
  token where they diverge, which is unbiased with respect to label token length.
- Without task fine-tuning the open Turkish models collapse to a single class,
  echoing the applicant's alleged violation rather than predicting the court's
  ruling (Table S8); fine-tuning removes this confound before faithfulness is
  measured.
