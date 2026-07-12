# Faithfulness of Explanations for Turkish Legal Judgment Classification in Decision Support

Code and results for the study comparing the faithfulness of four post-hoc explanation
methods (raw attention, attention rollout, Integrated Gradients, and Chefer-style
relevance propagation) on a Turkish BERT classifier fine-tuned to predict rights-violation
outcomes from Turkish Constitutional Court individual-application decisions.

The same faithfulness measurement is then extended to a generative model (RQ4): a Turkish
legal large language model is fine-tuned to the same task, and its natural-language
self-explanation is compared against an occlusion reference and a random baseline with the
same ERASER-style metrics. The generative-model code lives in `src/llm_pilot/`.

An archived version of this repository (code, results, and a frozen copy of the dataset)
is deposited at Zenodo: https://doi.org/10.5281/zenodo.21246640

## Data

The dataset is public and is not redistributed here. It is available on Hugging Face:

- `icgcihan/Turkish_Constutional_Court_Decisions` (licence: CC-BY-4.0)

The scripts and notebooks download it automatically. The 60 test instances used for the
explanation analysis are listed in `results/sample_ids.csv`.

## Model

The classifier is `dbmdz/bert-base-turkish-128k-cased` fine-tuned for binary classification
(head truncation to 512 tokens, learning rate 2e-5, three epochs, effective batch size 16,
seed 42). Training is reproduced by `notebooks/train_kaggle.ipynb`. The fine-tuned checkpoint
is not included here because of its size; it can be regenerated from the notebook or obtained
from the author on request.

For the generative-model analysis (RQ4), `ytu-ce-cosmos/Turkish-Llama-8b-Instruct-v0.1` is
fine-tuned to the same classification task with QLoRA (4-bit base, LoRA rank 16), reproduced
by `notebooks/llm_finetune_kaggle.ipynb`. The self-explanation faithfulness evaluation is run
by `notebooks/llm_faithfulness_pilot.ipynb`. See `src/llm_pilot/README.md` for the full
procedure.

## Contents

```
src/xai/                      analysis code
  explainer.py                the four explanation methods (common interface)
  faithfulness.py             ERASER comprehensiveness / sufficiency metrics
  run_revision.py             faithfulness run (two operators, stratified sample)
  run_random_reps.py          random baseline over 10 rankings per instance
  run_final_stats.py          Friedman + Holm-corrected pairwise + effect sizes
  run_effectsizes.py          bootstrap CIs and rank-biserial effect sizes
  run_suff_rq2.py             sufficiency pairwise and token-type shares
  run_aopc.py                 comprehensiveness perturbation curves
  leakage_analysis.py         verdict-cue scan, lexical baseline, truncation curve
  stratum_composition.py      class composition of the stratified sample
  make_figures_en.py          main-text figures
  make_appendix_figs.py       supplementary figures
  make_aopc_fig.py            perturbation-curve figure
  make_llm_fig_en.py          generative-model self-explanation figure (RQ4)
src/llm_pilot/                generative-model analysis (RQ4)
  llm_finetune.py             QLoRA fine-tuning of the base model
  llm_faithfulness_pilot.py   self-explanation elicitation and faithfulness evaluation
  README.md                   procedure and mapping of outputs to the paper
notebooks/
  train_kaggle.ipynb          fine-tuning (GPU)
  seed_robustness_kaggle.ipynb  three-seed ranking check (GPU)
  llm_finetune_kaggle.ipynb   generative-model QLoRA fine-tuning (GPU)
  llm_faithfulness_pilot.ipynb  generative-model self-explanation evaluation (GPU)
results/                      computed outputs and figures
  sample_ids.csv              the 60 evaluated test instances (id, stratum, label, prediction, confidence)
  faithfulness_rev_raw.csv    per-instance comprehensiveness and sufficiency
  random_reps.csv             per-instance random-baseline values (10 repetitions)
  final_stats.json            means, Friedman tests, pairwise comparisons, stratum table
  rq2_rev_full.csv            token-type attribution shares
  leakage.json                cue prevalence, lexical baseline, truncation macro-F1
  seed_robustness.csv         three-seed comprehensiveness per method
  attributions_rev.json       token-level explanation scores per method
  llm_perinstance.csv         generative-model per-instance self-explanation faithfulness (RQ4)
  figures/                    PNG figures
requirements.txt
```

## Reproducing the analysis

1. Create the environment:

   ```
   pip install -r requirements.txt
   ```

2. Fine-tune the model with `notebooks/train_kaggle.ipynb` (a GPU is needed). Save the
   resulting checkpoint locally as `models/`.

3. Run the analysis scripts in `src/xai/` from the repository root; they use paths
   relative to it and expect the fine-tuned model under `models/`, reading and writing
   intermediate files under `results/`. On CPU the full explanation and faithfulness
   pipeline takes a few hours.

4. Generate figures with `src/xai/make_figures_en.py`, `src/xai/make_appendix_figs.py`,
   `src/xai/make_aopc_fig.py`, and `src/xai/make_llm_fig_en.py`.

5. For the generative-model analysis (RQ4), follow `src/llm_pilot/README.md`: fine-tune with
   `notebooks/llm_finetune_kaggle.ipynb`, then evaluate with
   `notebooks/llm_faithfulness_pilot.ipynb`. The per-instance results are already provided in
   `results/llm_perinstance.csv`.

The `results/` directory already contains the outputs reported in the paper, so the
statistics and figures can be regenerated without re-running the model.

## Citation

Dal, B. Faithfulness of explanations for Turkish legal judgment classification in
decision support. (Under review.)

## Licence

The code in this repository is released under the MIT Licence (see `LICENSE`). The dataset
is licensed separately under CC-BY-4.0 by its authors.
