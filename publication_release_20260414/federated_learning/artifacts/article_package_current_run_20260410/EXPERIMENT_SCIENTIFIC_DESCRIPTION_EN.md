# Decentralized Federated Forecasting of Financial Time Series: Scientific Description of the Current Scenario

## Abstract

This experiment studies decentralized federated forecasting on real financial time-series data using interpretable parametric models and a synchronization mechanism based on the Local Voting Protocol (LVP). The goal is to analyze model behavior under partial observability, heterogeneous data distribution across agents, and Byzantine corruption of local updates. Special attention is given to the construction of the consensus graph, the Jaccard similarity threshold, the Jaccard-cosine hybrid coefficient, and the new `self_weight` parameter, which introduces explicit damping of the local update.

The fixed scenario uses 20 clients, 10 communication rounds, one local epoch per round, 25% malicious clients, and the `noise_colluded` attack. The final comparison in the current package includes four interpretable models: `DynamicLinearModel`, `ARMAXModel`, `KalmanFilterModel`, and `StructuralTimeSeriesModel`. `MarkovSwitchingRegressionModel` is available in the repository as an expandable baseline, but it is excluded from the fixed final comparison due to computational cost.

## 1. Problem Setting

We consider collaborative forecasting where each client observes only a partial slice of a common MCC transaction matrix and additionally receives the same exogenous information channels. The objective is to train a shared parametric model for the target series `amt` and synchronize parameters across agents without a centralized server, using a decentralized LVP-style consensus process.

Unlike classical FedAvg, where averaging is executed on a central server, here interaction occurs on a local neighborhood graph, and the strength of exchange is determined by similarity between client information profiles. This setting allows us to study not only predictive accuracy, but also robustness under partial modality availability, sparse connectivity, and malicious participants.

## 2. Data Sources

### 2.1 Primary source: MCC transactions

The main endogenous signal is derived from the MCC transaction table. The repository loads the data from one of the following files:

- [01_data_transactions/dat_mcc_full.csv](../../../../01_data_transactions/dat_mcc_full.csv)
- [01_data_transactions/dat_mcc.csv](../../../../01_data_transactions/dat_mcc.csv)

For compatibility with archived layouts, mirrors under `data_LVP/01_data_transactions/` are also supported. The loader is implemented in [federated_learning/data_loaders/data_utils.py](../../../../federated_learning/data_loaders/data_utils.py).

The file contains a date index and multiple numeric MCC columns. These columns are treated as transaction categories rather than as independent clients.

### 2.2 Exogenous channels

The experiment uses external informational channels that are attached to every client as exogenous regressors:

- a Fontanka news signal, loaded from [02_data_fontanka/news_clustered_final.csv](../../../../02_data_fontanka/news_clustered_final.csv) or [02_data_fontanka/fontanka_news_result.csv](../../../../02_data_fontanka/fontanka_news_result.csv);
- when available, a Reuters-based daily signal from the `real_data_integration` pipeline.

In the current fixed scenario, these channels are aligned to the MCC date index through the `_build_exogenous(...)` routine in [federated_learning/experiments/run_real_experiments.py](../../../../federated_learning/experiments/run_real_experiments.py). Each client therefore receives the same exogenous time series, while heterogeneity is induced by the MCC partitioning.

## 3. How Data Are Split Across Agents

### 3.1 Partition principle

The current scenario uses 20 clients and `column_partition = contiguous`. This means that numeric MCC columns are split into consecutive blocks, and each agent receives its own block of categories. The client target series is constructed as the sum of the assigned columns:

$$
amt_i(t) = \sum_{c \in C_i} x_c(t),
$$

where $C_i$ denotes the MCC columns assigned to client $i$.

Thus, agents do not split rows among themselves; instead, they split the feature space and construct local time series with the same temporal axis but different category subsets. This is the main source of heterogeneity in the experiment.

### 3.2 What is shared and what is partitioned

The following are partitioned or shared as follows:

- MCC transaction categories are partitioned across agents and form the local target series `amt`;
- exogenous signals (`exog_news`, `exog_reuters`) are shared across all clients;
- source-column metadata are retained and later used to build information profiles and the neighbor graph.

In other words, the transaction layer is split, while exogenous channels remain common to all clients. This distinction is essential: clients observe different transaction subsets, but the same external signals.

### 3.3 Client information profiles

For the LVP graph, each client is assigned an information profile $S_i$. The implementation is in [federated_learning/data_loaders/data_utils.py](../../../../federated_learning/data_loaders/data_utils.py). The profile includes:

- base tags such as `base`, `target`, and `source:mcc`;
- the domain tag `domain:mcc`;
- discretized local statistics, including quantiles of the mean, standard deviation, nonzero ratio, linear trend, and the number of source columns;
- MCC identifier buckets from the source columns;
- modality tags for exogenous columns.

This profile makes Jaccard similarity informative rather than constant, because it depends on the local composition and statistical structure of each client series.

## 4. Models Used in the Experiment

The fixed comparison in the current package includes four interpretable forecasting models:

| Model | Purpose | Main feature |
|---|---|---|
| `DynamicLinearModel` | State-space forecasting model with a dynamic linear structure | Naturally compatible with iterative federated synchronization |
| `ARMAXModel` | Autoregression with exogenous variables | Highly interpretable, but sensitive to parameter aggregation |
| `KalmanFilterModel` | Classical state-space model | Usually provides smoother and more stable dynamics |
| `StructuralTimeSeriesModel` | Structural decomposition of the time series | Separates trend, seasonal structure, and noise |

The repository also contains `MarkovSwitchingRegressionModel`, which is a more expressive regime-switching baseline, but it is excluded from the fixed comparison in this package for computational reasons.

## 5. Byzantine Agents and Attacks

### 5.1 Fraction of malicious clients

The current scenario sets `malicious_frac = 0.25`, i.e. approximately 25% of clients are malicious. With 20 agents, this corresponds to roughly 5 Byzantine nodes. The exact set is selected randomly under a fixed seed unless the `hub_targeted` mode is enabled.

The malicious-node selection logic is implemented in [federated_learning/experiments/run_real_experiments.py](../../../../federated_learning/experiments/run_real_experiments.py). By default, the malicious subset is sampled uniformly; in `hub_targeted` mode, the highest-degree nodes in the graph are selected.

### 5.2 Attack strategies

The repository implements several corruption modes for local parameters:

- `label_flip` - the sign of numeric parameters is inverted and noise is added;
- `noise` - Gaussian noise is added to parameters;
- `noise_colluded` - Gaussian noise plus a coordinated bias per parameter key; this is the attack used in the current experiment;
- `noise_heavy_tail` - a heavy-tailed Student-t attack;
- `noise_colluded_heavy_tail` - a coordinated heavy-tailed attack;
- `random` - numeric parameters are replaced with random values drawn from a wide range.

The corruption logic is implemented in [federated_learning/core/aggregators.py](../../../../federated_learning/core/aggregators.py). In `noise_colluded`, the same key-dependent bias direction is applied to all malicious clients, which is much more harmful to simple averaging than independent noise, because the shared bias does not cancel out.

### 5.3 Why this matters for LVP

LVP is less vulnerable to naive parameter blurring than plain FedAvg, but it remains sensitive to persistent systematic bias in transmitted parameters, especially when the graph is dense and the LVP step size is too large. Therefore, robustness depends not only on the topology of the consensus graph, but also on the exact form of Byzantine corruption.

## 6. LVP Mechanism

### 6.1 Update equation

The decentralized LVP synchronization is implemented in [federated_learning/core/decentralized_lvp.py](../../../../federated_learning/core/decentralized_lvp.py). The update at round $t$ is:

$$
\theta_i^{t+1} = (1-\gamma)\left[\theta_i^t + \alpha \sum_{j \in N_i(t)} b_{ij}^t \left(\theta_j^{sent,t} - \theta_i^t\right)\right] + \gamma \theta_i^t,
$$

where:

- $\theta_i^t$ is the local parameter vector of client $i$ after local training;
- $\theta_j^{sent,t}$ is the vector broadcast by neighbor $j$;
- $N_i(t)$ is the neighbor set of client $i$ at round $t$;
- $\alpha$ is the synchronization step size;
- $\gamma$ is the `self_weight` parameter;
- $b_{ij}^t$ are normalized trust weights.

### 6.2 Jaccard graph and threshold

The base neighborhood graph is built using Jaccard similarity:

$$
\kappa_{ij}^{(J)} = \frac{|S_i \cap S_j|}{|S_i \cup S_j|}.
$$

An edge $i \to j$ is created if $\kappa_{ij} \ge \tau$, where $\tau$ is the similarity threshold. In the current fixed scenario, `similarity_mode = jaccard` and `similarity_tau = 0.35`.

### 6.3 Hybrid Jaccard-cosine similarity

The repository also supports the `jaccard_cosine_hybrid` mode. It is implemented in [federated_learning/core/decentralized_lvp.py](../../../../federated_learning/core/decentralized_lvp.py) and combines Jaccard similarity with cosine similarity between parameter increments:

$$
\kappa_{ij}^{(H)} = \lambda \cdot \kappa_{ij}^{(J)} + (1-\lambda) \cdot \frac{1 + \cos(\Delta_i, \Delta_j)}{2},
$$

where $\lambda = \lambda_{jaccard} \in [0,1]$.

Here, $\Delta_i$ and $\Delta_j$ are the local parameter deltas of clients measured across adjacent rounds. Cosine similarity is rescaled from $[-1,1]$ to $[0,1]$ so that it can be mixed with Jaccard in a common range.

A lower bound `tau_cos_min` may also be used: if the cosine similarity between increments is too low, the edge is rejected even if the mixed score is high. This makes the graph more conservative when topological similarity alone is not sufficient to guarantee directional agreement.

### 6.4 The `self_weight` parameter

The `self_weight` parameter introduces damping and corresponds to $\gamma$ in the update equation above. It is clipped to $[0,1]$ and controls how much of the local state is preserved instead of being replaced by neighbor-driven updates.

Interpretation:

- `self_weight = 0` means that the client fully accepts the LVP update;
- positive values preserve part of the local state;
- larger values reduce oscillations and can mitigate instability under noisy or colluded attacks.

In scientific terms, this acts as a stabilizer that reduces the effective movement toward the consensus direction. In the current package, tuning experiments selected `self_weight = 0.0` as the best compromise, so damping was not needed in the final fixed configuration.

### 6.5 Edge weights

Neighbor weights are normalized as:

$$
\tilde b_{ij} = \kappa_{ij} \mathbf{1}_{\{j \in N_i\}}, \qquad
b_{ij} = \frac{\tilde b_{ij}}{\max(1, \sum_k \tilde b_{ik})}.
$$

This means that the final influence of each neighbor depends both on the graph structure and on the local similarity coefficient. The implementation is consistent with the definition in [federated_learning/core/decentralized_lvp.py](../../../../federated_learning/core/decentralized_lvp.py).

## 7. Simulation Parameters

The final fixed scenario uses the following parameters:

| Parameter | Value |
|---|---:|
| Number of clients | 20 |
| Number of rounds | 10 |
| Local epochs per round | 1 |
| Byzantine fraction | 0.25 |
| Attack strategy | `noise_colluded` |
| Attack scale | 5.0 |
| Network evaluation mode | `proxy` |
| MCC partition | `contiguous` |
| similarity_mode | `jaccard` |
| similarity_tau | 0.35 |
| lambda_jaccard | 0.5 |
| tau_cos_min | -1.0 |
| lvp_alpha | 0.6 |
| lvp_self_weight | 0.0 |
| Final seed | 42 |

The hyperparameters were selected using dedicated ablations:

- joint tau/alpha sweep over three seeds: [federated_learning/artifacts/ablation_tau_alpha_article_rounds10/tau_alpha_ablation_rounds10_article_report.md](../ablation_tau_alpha_article_rounds10/tau_alpha_ablation_rounds10_article_report.md);
- multiseed alpha/self_weight tuning: [federated_learning/artifacts/article_package_current_run_20260410/plots/ablation/lvp_alpha_self_weight_tuning_report.md](plots/ablation/lvp_alpha_self_weight_tuning_report.md).

## 8. Results

### 8.1 Final model comparison

The final run is stored in [raw/all_models_lvp_seed42.json](raw/all_models_lvp_seed42.json). After 10 rounds on seed 42, the final network MAE values are:

| Model | Final MAE |
|---|---:|
| KalmanFilterModel | 356264.79 |
| DynamicLinearModel | 383871.77 |
| ARMAXModel | 503773.60 |
| StructuralTimeSeriesModel | 660136.42 |

The per-round learning curves are shown in [plots/dynamics/all_models_current_scenario_learning_curves.png](plots/dynamics/all_models_current_scenario_learning_curves.png), and the general comparison figure is in [plots/dynamics/model_comparison_learning_curves.png](plots/dynamics/model_comparison_learning_curves.png).

### 8.2 Interpretation of dynamics

1. `KalmanFilterModel` achieves the best final result in the current scenario, which is consistent with its state-space structure and smoothing behavior.
2. `DynamicLinearModel` remains stable and predictable, but slightly worse than the Kalman filter.
3. `ARMAXModel` no longer collapses to the `1e12` penalty after the forecasting fallback fix and parameter stabilization, but it is still the most sensitive to colluded corruption and parameter aggregation.
4. `StructuralTimeSeriesModel` shows a larger error in this setting, indicating that a more expressive structural decomposition does not automatically compensate for heterogeneity and Byzantine disturbance.

### 8.3 Coherence versus alpha

To visualize how the LVP step size influences internal network coherence, an additional alpha sweep was run for the fixed scenario. The resulting figure is [plots/coherence/lvp_coherence_by_alpha_current_scenario.png](plots/coherence/lvp_coherence_by_alpha_current_scenario.png).

This figure is scientifically relevant for two reasons:

- it shows how alpha affects round-by-round coherence, not just the final MAE;
- it illustrates that overly aggressive synchronization is not always beneficial under Byzantine noise and heterogeneous local series.

### 8.4 Reflection on ARMAX

ARMAX was originally unstable because an incorrect exogenous forecasting fallback could drive the model into penalty values. After fixing the fallback and stabilizing the parameters, the model became usable, but it remains the most sensitive to colluded corruption and parameter mixing across highly heterogeneous clients. This is expected: the interpretable coefficients and intercept of ARMAX are especially vulnerable to aggregation on an unstable graph.

## 9. Scientific Motivation

The experiment follows the logic of decentralized multimodal federated learning: data are distributed across autonomous participants, and parameter synchronization occurs through a local consensus process rather than through a central server. Unlike purely neural FL schemes, the models here are parametric and interpretable, which makes it possible to isolate the effect of:

- data structure;
- graph topology;
- Byzantine attacks;
- synchronization step size alpha;
- damping via `self_weight`;
- the choice of similarity metric.

This is precisely why the hybrid Jaccard, the threshold tau, and `self_weight` should be described explicitly in the article: they control not only learning quality, but also the geometry of inter-agent interaction.

## 10. Short Comparison to Analogs

- `ARMAXModel` is a classical baseline for time-series forecasting with exogenous inputs; it is highly interpretable but sensitive to aggregation.
- `DynamicLinearModel` is a state-space analogue that is more naturally aligned with iterative federated synchronization.
- `KalmanFilterModel` provides a more rigorous filtering formulation and is often robust to noise.
- `StructuralTimeSeriesModel` decomposes the series into components and can be more expressive, though not necessarily more robust.

`MarkovSwitchingRegressionModel` is available as an additional regime-switching baseline, but it is not part of the final fixed comparison in this package.

## 11. Reproducibility

Main entry points:

- run and comparison script: [federated_learning/experiments/run_all_models_current_scenario.py](../../../../federated_learning/experiments/run_all_models_current_scenario.py);
- LVP graph construction: [federated_learning/core/decentralized_lvp.py](../../../../federated_learning/core/decentralized_lvp.py);
- data partitioning and exogenous construction: [federated_learning/data_loaders/data_utils.py](../../../../federated_learning/data_loaders/data_utils.py) and [federated_learning/experiments/run_real_experiments.py](../../../../federated_learning/experiments/run_real_experiments.py);
- final results: [raw/all_models_lvp_seed42.json](raw/all_models_lvp_seed42.json).

Main figures:

- [plots/dynamics/all_models_current_scenario_learning_curves.png](plots/dynamics/all_models_current_scenario_learning_curves.png)
- [plots/dynamics/model_comparison_learning_curves.png](plots/dynamics/model_comparison_learning_curves.png)
- [plots/coherence/lvp_coherence_by_alpha_current_scenario.png](plots/coherence/lvp_coherence_by_alpha_current_scenario.png)
- [plots/ablation/tau_ablation_rounds10_article.png](plots/ablation/tau_ablation_rounds10_article.png)
- [plots/ablation/alpha_ablation_rounds10_article.png](plots/ablation/alpha_ablation_rounds10_article.png)

## 12. Conclusion

The results show that decentralized LVP synchronization can stabilize interpretable forecasting models under partial observability and Byzantine disturbance on a real financial multimodal dataset. In the current scenario, `KalmanFilterModel` is the strongest performer, while `ARMAXModel`, after the forecasting fix, becomes operational but still remains sensitive to colluded attack and parameter aggregation. The hybrid Jaccard and `self_weight` should be viewed as mechanisms for controlling graph structure and update damping, respectively; in this package they are real optimization hyperparameters rather than cosmetic settings.
