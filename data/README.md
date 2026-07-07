# Data

## Source

The dataset is `icgcihan/Turkish_Constutional_Court_Decisions`, published on Hugging Face
under a CC-BY-4.0 licence:

- https://huggingface.co/datasets/icgcihan/Turkish_Constutional_Court_Decisions

It contains Turkish Constitutional Court individual-application decisions with a binary
rights-violation label. The scripts and notebooks download it directly from Hugging Face.

## Frozen copy

To keep the results reproducible even if the Hugging Face source changes, a frozen copy of
the dataset used in this study is archived here:

- `turkish_constitutional_court_decisions.csv.gz`

It contains all three splits (a `split` column marks train/validation/test) with the
original columns (`text`, `Haklar`, `Kararın Bağlantı Linki`, `Başvuru Konusu`, `labels`);
the `split` column is the only addition. This copy is redistributed under the dataset's
CC-BY-4.0 licence, with attribution to the original author.

## Evaluated instances

The 60 test instances used for the explanation analysis, together with their stratum, gold
label, model prediction, and confidence, are listed in `../results/sample_ids.csv`.
