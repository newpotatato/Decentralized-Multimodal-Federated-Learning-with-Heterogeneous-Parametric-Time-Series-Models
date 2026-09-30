# DataFusion-2023 experiments

Experiments of the manuscript *Decentralized multimodal federated forecasting via profile-aware local voting* (S. Slastnikov, E. Tarasova, K. Chernikov) on the DataFusion-2023 transaction data with exogenous news sentiment.

The code is in `../publication_release_20260414/federated_learning/`; this folder contains the pipelines, the data-preparation scripts, and the results reported in the manuscript.

## Contents

| Path | Purpose |
|---|---|
| `run_pipeline_v2.sh` | Main pipeline: ablation over the similarity threshold and step size, scenarios A–C, and reference runs (no exchange, no attack, no exogenous regressors); five seeds |
| `run_full_pipeline_exog.ps1` | Earlier pipeline (three seeds) that produced `artifacts_pipeline_exog/`; see the note below |
| `data_prep/fetch_reuters_gdelt.py` | Collects Reuters articles (URL, headline, day) for 2017-12..2021-08 from the GDELT 1.0 daily event exports |
| `data_prep/score_news_sentiment.py` | Per-article sentiment: Fontanka (`cointegrated/rubert-tiny-sentiment-balanced`), Reuters headlines (`ProsusAI/finbert`) |
| `data_prep/summarize_v2.py` | Prints the numbers reported in the manuscript from `artifacts_pipeline_v2/` |
| `data_prep/make_article_figures_v2.py` | Draws the manuscript figures from `artifacts_pipeline_v2/` |
| `artifacts_pipeline_v2/` | Results of `run_pipeline_v2.sh` (per-seed JSON histories, summaries, figures) |
| `artifacts_pipeline_exog/` | Results of `run_full_pipeline_exog.ps1` |
| `code_snapshots/variant1_exog_run/` | Versions of the source files that differ from the current code and were used for `artifacts_pipeline_exog/` |
| `aggregate_seeds.py` | Aggregates per-seed JSON files |

## Data

The raw data are not redistributed here.

- `01_data_transactions/dat_mcc.csv` — daily transaction amounts per merchant category built from the Data Fusion Contest 2023 "Attack" dataset (https://ods.ai/competitions/data-fusion2023-attack/Dataset); the dataset is available from ODS.ai after registration.
- `02_data_fontanka/news_clustered_final.csv` — Fontanka news leads with publication dates (columns `date`, `text`).
- `real_data_integration/reuters_2018_2021/` — produced by `data_prep/fetch_reuters_gdelt.py` (public GDELT exports, about 7 GB of downloads, processed in a streaming way).

Place the files under this folder with the paths above. The loaders look for them relative to `--base-path`, which the pipelines set to this folder.

## Reproduction

```bash
pip install -r ../publication_release_20260414/requirements.txt

# exogenous news sentiment (torch and transformers are needed only here)
python data_prep/fetch_reuters_gdelt.py
python data_prep/score_news_sentiment.py fontanka
python data_prep/score_news_sentiment.py reuters        # samples up to 50 headlines per day

# experiments (about 45 minutes on 12 CPU cores) and reported numbers
PYTHON=python bash run_pipeline_v2.sh
python data_prep/summarize_v2.py
python data_prep/make_article_figures_v2.py
```

## Setup in brief

- 10 clients; each client's target is the standardized daily sum over a random disjoint subset of merchant categories; one random partition per seed (seeds 42, 52, 62, 72, 82).
- Dynamic Linear Model (SARIMAX(1,0,0) with the two news-sentiment regressors lagged by the 10-day forecast horizon); 10 rounds, 1 local epoch with at most 5 optimizer iterations per round.
- Evaluation: MAE of a single 10-step-ahead forecast issued at the end of the training part (first 80% of each series), computed with the synchronized parameters and averaged over all clients.
- Attacks on the transmitted vectors of 40% of the clients: sign inversion (`-2.5 x`) or colluded noise (Gaussian noise with standard deviation `5 |x|` plus a shared-sign bias).

## Note on `artifacts_pipeline_exog/`

These results were produced before the following corrections to the code, which are included in the current version and in `artifacts_pipeline_v2/`:

- the attack noise was scaled by the within-parameter standard deviation, which is zero for the scalar parameters of the model, so `noise_colluded` was a negligible perturbation and `label_flip` a pure sign inversion;
- Push-Sum received the unperturbed local models of its neighbors;
- the update-direction cosine of the hybrid score was computed from local rather than transmitted vectors;
- the ablation used a single client partition for all seeds, the profiles contained an additional synthetic group tag, and the hybrid graph applied an extra cosine cut of 0.15 in the scenario runs.

To rerun this earlier pipeline, copy the files from `code_snapshots/variant1_exog_run/` over the corresponding files of `publication_release_20260414/federated_learning/` (core and experiments).
