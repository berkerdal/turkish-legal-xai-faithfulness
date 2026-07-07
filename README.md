# Faithfulness of Post-hoc Explanations for Turkish Legal Judgment Classification in Decision Support

Code and results for the study comparing the faithfulness of four post-hoc explanation
methods (raw attention, attention rollout, Integrated Gradients, and Chefer-style
relevance propagation) on a Turkish BERT classifier fine-tuned to predict rights-violation
outcomes from Turkish Constitutional Court individual-application decisions.

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
  rq2_token_types.py          attribution mass by token type
  make_figures_en.py          main-text figures
  make_appendix_figs.py       supplementary figures
  make_aopc_fig.py            perturbation-curve figure
notebooks/
  train_kaggle.ipynb          fine-tuning (GPU)
  seed_robustness_kaggle.ipynb  three-seed ranking check (GPU)
results/                      computed outputs and figures
  sample_ids.csv              the 60 evaluated test instances (id, stratum, label, prediction, confidence)
  faithfulness_rev_raw.csv    per-instance comprehensiveness and sufficiency
  random_reps.csv             per-instance random-baseline values (10 repetitions)
  final_stats.json            means, Friedman tests, pairwise comparisons, stratum table
  rq2_rev_full.csv            token-type attribution shares
  leakage.json                cue prevalence, lexical baseline, truncation macro-F1
  seed_robustness.csv         three-seed comprehensiveness per method
  attributions_rev.json       token-level explanation scores per method
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
   and `src/xai/make_aopc_fig.py`.

The `results/` directory already contains the outputs reported in the paper, so the
statistics and figures can be regenerated without re-running the model.

## Citation

Dal, B. Faithfulness of post-hoc explanations for Turkish legal judgment classification in
decision support. (Under review.)

## Licence

The code in this repository is released under the MIT Licence (see `LICENSE`). The dataset
is licensed separately under CC-BY-4.0 by its authors.
