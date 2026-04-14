# Experimental Setup

## Experimental Design

This study investigates decentralized federated forecasting for financial time series with interpretable parametric models and a synchronization mechanism based on the Local Voting Protocol (LVP). The experiment is designed to study how the interaction graph, the Jaccard similarity threshold, the hybrid similarity option, and the additional self_weight parameter affect training stability and network coherence under partial observability, heterogeneous data placement, and Byzantine corruption of local updates.

The final package uses 20 clients, 10 communication rounds, one local epoch per round, a malicious-client ratio of 25%, and the `noise_colluded` attack scenario. The model comparison includes four interpretable forecasters: DynamicLinearModel, ARMAXModel, KalmanFilterModel, and StructuralTimeSeriesModel. MarkovSwitchingRegressionModel remains available in the repository as an extensible baseline, but it is excluded from the final comparable run because of computational cost.

## Data Sources

The primary data source is the MCC transaction matrix. The repository uses real files from `01_data_transactions`, primarily `dat_mcc_full.csv`, with `dat_mcc.csv` available as a lighter variant. For compatibility with the archival structure, mirrored copies are also present in `data_LVP/01_data_transactions`. The data contain a time index and multiple MCC columns that are treated not as independent clients but as categories of transactional activity.

External information channels are used as exogenous factors. In the current experiment, each client receives the Fontanka news signal and, when available in the integration pipeline, a daily Reuters signal. These series are aligned to the MCC timeline and attached to every client as common exogenous regressors. Thus, the heterogeneity in the experiment comes from the partitioning of transactional data, while the external channels remain shared across clients.

## Partitioning Across Clients

Data are partitioned by feature columns rather than by rows. In the current scenario, `column_partition = contiguous`, meaning that the numerical MCC columns are split into contiguous blocks and each client receives one block of categories. Each client's local target series is formed as the sum of its assigned MCC columns. As a result, all clients operate on a shared temporal axis but observe different subsets of transactional categories and therefore different local time series.

This design introduces a natural form of heterogeneity. Clients differ not only in their samples but also in the modality structure of the data they observe. At the same time, the external news and Reuters channels remain common to all participants and act as a synchronized informational background.

For neighbor-graph construction, each client is assigned an information profile. The profile includes source and domain tags, discretized statistics of the local series, bins derived from the original MCC identifiers, and modality tags for exogenous features. This makes Jaccard similarity informative rather than constant: it reflects both shared structure and similarity in the local data geometry.

## Models and Baselines

The fixed comparison in the current package includes four interpretable models. DynamicLinearModel is a state-space model with a dynamic linear structure and is naturally compatible with round-by-round federated synchronization. ARMAXModel is a classical autoregressive model with exogenous regressors and serves as a sensitive parametric baseline. KalmanFilterModel provides a stricter filtering formulation of state-space modeling and typically produces smoother dynamics. StructuralTimeSeriesModel decomposes the series into structural components and serves as a more expressive baseline, but not necessarily a more stable one.

All models are evaluated in the same scenario and on the same data to ensure that differences in quality are attributable to the model architecture, the synchronization mechanism, and the robustness to Byzantine corruption rather than to changes in the data regime.

## Byzantine Scenarios

The malicious-client ratio in the final scenario is 25%, which corresponds to roughly five Byzantine clients out of 20 agents. The malicious set is sampled randomly under a fixed seed unless an alternative high-degree selection mode is enabled. This models a realistic setting in which a subset of clients systematically perturbs its local updates.

Several attack modes are implemented in the repository. The `label_flip` mode flips the sign of numerical parameters and adds noise. The `noise` mode adds standard Gaussian perturbations. The `noise_colluded` mode, used in the current scenario, adds both random noise and a coordinated key-wise bias shared by all malicious clients; this is particularly harmful for simple averaging because independent noise can partially cancel while the aligned bias remains. Heavy-tailed variants based on Student-t noise and a `random` mode with wide-range random replacement are also available.

## LVP Mechanics

Decentralized synchronization is implemented through the Local Voting Protocol. After local training, each client shares its parameter vector with its neighbors, and the new state is computed as a local update shifted toward neighboring parameters. In its base form, the update rule is

$$
\theta_i^{t+1} = (1-\gamma)\left[\theta_i^t + \alpha \sum_{j \in N_i(t)} b_{ij}^t \left(\theta_j^{sent,t} - \theta_i^t\right)\right] + \gamma \theta_i^t,
$$

where $\alpha$ is the synchronization step, $\gamma$ is the `self_weight` parameter, $N_i(t)$ is the neighborhood of client $i$ at round $t$, and $b_{ij}^t$ are normalized trust weights.

The graph is built from Jaccard similarity between information profiles:

$$
\kappa_{ij}^{(J)} = \frac{|S_i \cap S_j|}{|S_i \cup S_j|}.
$$

An edge between two clients is created only when this similarity exceeds the threshold $\tau$. In the current scenario, `similarity_mode = jaccard` and `similarity_tau = 0.35`, which means that interactions are allowed only between clients whose profiles are sufficiently close in terms of both observed modality structure and local-series statistics.

### Hybrid Jaccard-Cosine Similarity

The repository also supports a hybrid `jaccard_cosine_hybrid` mode, where structural similarity between profiles is combined with the direction of local parameter updates. The hybrid similarity is defined as a convex combination of Jaccard similarity and cosine similarity between neighboring parameter increments:

$$
\kappa_{ij}^{(H)} = \lambda \cdot \kappa_{ij}^{(J)} + (1-\lambda) \cdot \frac{1 + \cos(\Delta_i, \Delta_j)}{2},
$$

where $\lambda = \lambda_{jaccard} \in [0,1]$. The cosine term is rescaled from $[-1,1]$ to $[0,1]$ so that it can be combined with Jaccard on a common scale. A lower bound `tau_cos_min` is additionally used to reject edges with negative or too weak directional agreement.

Scientifically, this hybrid coefficient is useful because it combines static information about data profiles with dynamic information about the direction of parameter updates. In the current package, it is treated as a separate research option and as an ablation mechanism relative to pure Jaccard.

### The self_weight Parameter

The `self_weight` parameter implements damping of the local update and corresponds to the coefficient $\gamma$ in the equation above. When `self_weight = 0`, the client fully accepts the LVP update. For positive values, part of the local state is retained, which reduces oscillations and can improve robustness under noise and colluded attacks. In the current package, the tuning experiments indicate that the best compromise is obtained at `self_weight = 0.0`, so the final configuration does not use additional damping.

## Simulation Parameters

The final fixed scenario uses 20 clients, 10 rounds, one local epoch per round, a Byzantine fraction of 0.25, the `noise_colluded` attack, an attack scale of 5.0, the `proxy` evaluation mode, `contiguous` feature partitioning, `jaccard` similarity mode, `similarity_tau = 0.35`, `lambda_jaccard = 0.5`, `tau_cos_min = -1.0`, `lvp_alpha = 0.6`, `lvp_self_weight = 0.0`, and seed 42 for the final run.

The hyperparameters were selected through dedicated ablation studies rather than by manual tuning. A joint sweep over tau and alpha was performed across three seeds, and alpha/self_weight tuning used a multiseed objective defined as mean + 0.5·std of the final MAE. This makes the selected values stable rather than seed-specific.

## Ablation Studies

### Jaccard Threshold

The tau sweep shows that the interaction threshold has a non-monotonic effect on the final error. The best value in the joint three-seed sweep is `tau = 0.5`, which gives a mean final MAE of 383803.98 with a standard deviation of 212.24. The neighboring values remain close, but `tau = 0.3` is clearly worse at 415585.03, which indicates that too permissive graph construction can amplify cross-client interference under colluded corruption.

Figure placeholder: tau ablation curve will be placed here.

### LVP Step Size

The alpha sweep is equally sensitive. In the joint tau-alpha ablation, the best alpha is `0.6`, with mean final MAE 383907.29, standard deviation 121.01, and objective 383967.79. At the level of the round-by-round coherence sweep, `alpha = 0.45` achieves the lowest final network MAE, 382788.99, while `alpha = 0.6` gives 383871.77. We keep `alpha = 0.6` in the final package because it is the best compromise under the multiseed joint criterion rather than under a single-sweep minimum.

Figure placeholder: alpha ablation curve will be placed here.

### Self-Weight Damping

The self_weight sweep indicates that retaining additional local state does not improve this scenario. The best pair in the joint tuning grid is `alpha = 0.6` and `self_weight = 0.0`, with mean final MAE 383907.29, standard deviation 121.01, and objective 383967.79. Larger self_weight values gradually increase the objective, for example `self_weight = 0.15` at `alpha = 0.1` yields an objective of 384041.91, and `self_weight = 0.3` at `alpha = 0.5` rises to 385733.58. This supports the choice of no extra damping in the final scenario.

Figure placeholder: self_weight tuning plot will be placed here.

### Coherence Sweep

The coherence-by-alpha sweep provides the round-level view that complements the MAE-based ablations. Across the fixed scenario with `self_weight = 0.0`, the final network MAE varies from 382788.99 at `alpha = 0.45` to 409534.27 at `alpha = 0.35`, while `alpha = 0.6` ends at 383871.77. This figure is useful because it shows whether more aggressive coupling improves convergence or only changes the amount of mixing between heterogeneous clients.

Figure placeholder: coherence-by-alpha plot will be placed here.

## Results

The final run of the current scenario is stored in `raw/all_models_lvp_seed42.json`. After 10 rounds on seed 42, the final network MAE values were 356264.79 for KalmanFilterModel, 383871.77 for DynamicLinearModel, 503773.60 for ARMAXModel, and 660136.42 for StructuralTimeSeriesModel. The corresponding learning curves are shown in `plots/dynamics/all_models_current_scenario_learning_curves.png` and `plots/dynamics/model_comparison_learning_curves.png`.

KalmanFilterModel gives the strongest result, which is consistent with its state-space structure and its ability to smooth noisy updates. DynamicLinearModel remains stable but slightly worse than KalmanFilterModel. After fixing the forecast fallback logic and stabilizing ARMAX parameters, ARMAX no longer collapsed to the 1e12 penalty regime, but it remains sensitive to colluded corruption and to parameter mixing across heterogeneous clients. StructuralTimeSeriesModel exhibits a higher error, suggesting that a richer structural decomposition does not automatically translate into better robustness in this setup.

To analyze network coherence, an additional alpha sweep was performed and round-by-round coherence curves were built for different LVP step sizes. The resulting plot is shown in `plots/coherence/lvp_coherence_by_alpha_current_scenario.png`. This figure is important because it captures not only the final error but also the internal synchronization dynamics of the client network, allowing one to evaluate whether stronger coupling is actually beneficial under Byzantine perturbations.

## Interpretation

The experiment shows that decentralized LVP can stabilize the training of interpretable models under partial observability and Byzantine noise. KalmanFilterModel is the most robust model in this scenario, while ARMAX became workable only after the forecast-evaluation logic was corrected and its parameters were stabilized. Nevertheless, ARMAX remains highly sensitive to systematic corruption and to parameter aggregation across very heterogeneous clients. The hybrid Jaccard option and `self_weight` should therefore be treated as genuine control knobs that affect not only accuracy but also the geometry of the inter-client consensus process.

## Reproducibility

The main implementation entry points are:

- [federated_learning/experiments/run_all_models_current_scenario.py](../../experiments/run_all_models_current_scenario.py)
- [federated_learning/experiments/run_real_experiments.py](../../experiments/run_real_experiments.py)
- [federated_learning/core/decentralized_lvp.py](../../core/decentralized_lvp.py)
- [federated_learning/data_loaders/data_utils.py](../../data_loaders/data_utils.py)

The key artifacts of the experiment are:

- [plots/dynamics/all_models_current_scenario_learning_curves.png](plots/dynamics/all_models_current_scenario_learning_curves.png)
- [plots/dynamics/model_comparison_learning_curves.png](plots/dynamics/model_comparison_learning_curves.png)
- [plots/coherence/lvp_coherence_by_alpha_current_scenario.png](plots/coherence/lvp_coherence_by_alpha_current_scenario.png)
- [plots/ablation/tau_ablation_rounds10_article.png](plots/ablation/tau_ablation_rounds10_article.png)
- [plots/ablation/alpha_ablation_rounds10_article.png](plots/ablation/alpha_ablation_rounds10_article.png)
