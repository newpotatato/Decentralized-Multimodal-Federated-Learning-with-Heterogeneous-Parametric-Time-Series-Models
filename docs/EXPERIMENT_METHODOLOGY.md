# Experiment Methodology: Byzantine Resilience in Federated Learning

## Scientific Documentation for Publication

**Experiment Title**: Byzantine-Resilient Aggregation for Federated Learning on Financial Time-Series Data

**Date**: December 2025  
**Version**: 1.0

---

## 1. Research Question

**Primary Hypothesis**: LVP (Limited Vector Projection) aggregation provides superior robustness to Byzantine attacks compared to standard FedAvg in federated learning scenarios with non-IID financial time-series data.

**Null Hypothesis**: There is no significant difference between FedAvg and LVP aggregation under Byzantine attacks.

---

## 2. Experimental Design

### 2.1 Data Sources

#### Primary Dataset: MCC Transaction Data
- **File**: `01_data_transactions/dat_mcc.csv`
- **Description**: Real-world credit card transaction amounts aggregated by merchant category code (MCC)
- **Properties**:
  - Time-series length: ~1500 observations
  - Temporal granularity: Daily aggregations
  - Data characteristics: Non-stationary, high volatility, heterogeneous patterns

#### Auxiliary Dataset: News Sentiment
- **File**: `02_data_fontanka/fontanka_news_result.csv`
- **Description**: Sentiment scores from Fontanka news articles
- **Usage**: Exogenous factors for ARMAX and regression models
- **Properties**: Daily sentiment scores aligned with transaction data

### 2.2 Client Data Distribution

**Heterogeneity Simulation**:
- Method: K-means clustering (k=20)
- Purpose: Create realistic non-IID data partitions representing different client behaviors
- Selection: 5 clients randomly sampled from 20 clusters per experiment
- Rationale: Simulates real federated settings where clients have different data distributions

**Mathematical Formulation**:
```
X_train → KMeans(k=20) → {C₁, C₂, ..., C₂₀}
Clients = random_sample(Clusters, n=5)
```

### 2.3 Models Tested

Five state-of-the-art time-series models:

#### Model 1: ARMAX
- **Class**: `ARMAXModel` from `07_prediction_models/arma_models.py`
- **Parameters**: Auto-selected orders (p, d, q) via AIC
- **Exogenous**: News sentiment features
- **Characteristics**: Linear, interpretable, fast convergence

#### Model 2: Dynamic Linear Model (DLM)
- **Class**: `DynamicLinearModel` from `state_space_models.py`
- **Type**: State-space model with Kalman filtering
- **Parameters**: State dimension = 2, observation noise variance
- **Characteristics**: Handles time-varying parameters

#### Model 3: Kalman Filter
- **Class**: `KalmanFilterModel`
- **Type**: Classical state-space model
- **Parameters**: Transition and observation matrices
- **Characteristics**: Optimal for Gaussian noise

#### Model 4: Structural Time Series
- **Class**: `StructuralTimeSeriesModel`
- **Components**: Trend + seasonal + irregular
- **Parameters**: Component variances
- **Characteristics**: Decomposable, interpretable

#### Model 5: Markov Switching Regression
- **Class**: `MarkovSwitchingRegressionModel`
- **Type**: Regime-switching model
- **Parameters**: 2 regimes (bull/bear market analogy)
- **Characteristics**: Captures non-linear dynamics

---

## 3. Byzantine Attack Protocol

### 3.1 Attack Strategy: Label Flipping

**Implementation**: `corrupt_params()` in `08_federated_learning/v2/aggregators.py`

```python
def corrupt_params(params, attack_strategy='label_flip', scale=2.5):
    """
    Byzantine attack: Invert parameter updates
    
    Malicious update: θ_malicious = -scale × θ_benign
    
    Effect: Pushes global model in opposite direction
    """
```

**Scale Factor**: 2.5
- Chosen empirically to cause significant but not catastrophic degradation
- Represents sophisticated adversary with knowledge of legitimate updates

### 3.2 Attack Intensities

Three scenarios tested:

| Scenario | Malicious % | Benign Clients | Malicious Clients |
|----------|-------------|----------------|-------------------|
| Baseline | 0%          | 5              | 0                 |
| Moderate | 20%         | 4              | 1                 |
| Severe   | 40%         | 3              | 2                 |

**Rationale**: 40% represents near-maximum Byzantine tolerance (theoretical limit ~50%)

### 3.3 Attack Timing

- **When**: Every training round (rounds 0-7)
- **Target**: Model parameters after local training
- **Persistence**: Malicious clients remain malicious throughout experiment

---

## 4. Aggregation Methods

### 4.1 FedAvg (Baseline)

**Algorithm**: Weighted average by client data size

```
θ_global^{t+1} = Σᵢ (nᵢ/N) × θᵢ^t

where:
  nᵢ = data size of client i
  N = total data across all clients
  θᵢ^t = parameters from client i at round t
```

**Vulnerability**: Malicious parameters directly influence global model proportionally to their weight.

### 4.2 LVP (Proposed Robust Method)

**Algorithm**: Limited Vector Projection with quality scoring

```
1. Compute quality scores qᵢ for each client (based on validation loss)
2. For each parameter dimension:
   a. Compute weighted median: θ_med = weighted_median({θᵢ}, {qᵢ})
   b. Project outliers: θᵢ' = project(θᵢ, θ_med, radius=β×std)
3. Aggregate: θ_global = Σᵢ qᵢ × θᵢ' / Σᵢ qᵢ
```

**Key Properties**:
- Robustness: Up to 50% Byzantine clients (with β=0.5)
- Quality-weighted: Better clients have more influence
- Projection: Limits impact of extreme outliers

**Implementation**: `lvp_aggregate()` in `aggregators.py`

---

## 5. Experimental Protocol

### 5.1 Training Configuration

| Parameter              | Value          | Justification                          |
|-----------------------|----------------|----------------------------------------|
| Number of clients     | 5              | Typical small federated scenario       |
| Training rounds       | 8              | Sufficient for convergence             |
| Local epochs per round| 1              | Standard federated learning practice   |
| Test/train split      | 20% test       | Standard validation split              |
| Batch training        | Full batch     | Deterministic for reproducibility      |

### 5.2 Random Seeds

Five seeds used for statistical robustness: `[42, 123, 456, 789, 2024]`

**Purpose**: 
- Compute mean and standard deviation across runs
- Provide confidence intervals
- Test statistical significance

### 5.3 Single Experiment Flow

```
FOR each model in [ARMAX, DLM, Kalman, Structural, MarkovReg]:
  FOR each malicious_frac in [0.0, 0.2, 0.4]:
    FOR each seed in [42, 123, 456, 789, 2024]:
      
      1. Load data and split into 20 clusters (K-means)
      2. Randomly select 5 clients
      3. Randomly select malicious clients (based on malicious_frac)
      
      4. FOR round in 0..7:
         a. Each client trains model locally on their data
         b. Malicious clients corrupt their parameters
         c. Server aggregates parameters (FedAvg OR LVP)
         d. Broadcast updated global model
         e. Evaluate global model on each client's test set
      
      5. Record loss trajectory for all rounds
```

---

## 6. Evaluation Metrics

### 6.1 Primary Metrics

**Final Loss** (lower is better):
```
Loss_final = mean(MSE_client_i) at round 8
```

**Robustness Score**:
```
Robustness = (Loss_FedAvg_40% - Loss_LVP_40%) / Loss_FedAvg_40% × 100%
```
Higher percentage = LVP more robust

### 6.2 Secondary Metrics

**Convergence Speed**:
- Rounds to reach 90% of final loss
- Lower = faster convergence

**Loss Growth Rate**:
```
Growth = (Loss_40% - Loss_0%) / Loss_0% × 100%
```
Measure of vulnerability to attacks

**Cross-Client Variance**:
```
Variance = std(MSE across clients)
```
Measure of fairness/heterogeneity handling

---

## 7. Statistical Analysis

### 7.1 Descriptive Statistics

For each configuration (model + malicious_frac + aggregator):
- **Mean** across 5 seeds
- **Standard deviation** (error bars)
- **Min/Max** range

### 7.2 Hypothesis Testing

**Paired t-test**: FedAvg vs LVP at 40% malicious
```
H₀: μ_FedAvg = μ_LVP
H₁: μ_FedAvg > μ_LVP (one-tailed)
```

**Significance level**: α = 0.05

**Effect size**: Cohen's d
```
d = (mean_FedAvg - mean_LVP) / pooled_std
```

### 7.3 Multiple Comparisons

**Bonferroni correction** applied when testing all 5 models:
```
α_corrected = 0.05 / 5 = 0.01
```

---

## 8. Reproducibility Checklist

### 8.1 Code Artifacts

- [x] `test_5models.py` - Main experiment script
- [x] `reproduce_experiments.py` - Master reproduction script
- [x] `publication_plots.py` - Figure generation
- [x] `latex_tables.py` - Table generation
- [x] `aggregators.py` - FedAvg and LVP implementations
- [x] `requirements_publication.txt` - Dependencies

### 8.2 Data Availability

- [x] Transaction data path documented
- [x] Synthetic data fallback implemented
- [x] Data preprocessing code included

### 8.3 Experiment Configuration

- [x] All hyperparameters documented
- [x] Random seeds specified
- [x] Attack parameters defined
- [x] Model architectures described

### 8.4 Results Validation

- [x] Raw results saved as JSON
- [x] Aggregated statistics computed
- [x] Figures in vector format (PDF)
- [x] LaTeX tables generated

---

## 9. Expected Results

### 9.1 Hypothesized Outcomes

**H1**: FedAvg loss increases significantly with malicious fraction
- Expected growth: 150-400%

**H2**: LVP loss remains stable across malicious fractions
- Expected growth: <50%

**H3**: LVP achieves 20-70% lower final loss than FedAvg at 40% malicious

**H4**: All 5 models show consistent pattern (LVP outperforms FedAvg)

### 9.2 Visualization Patterns

**Learning Curves**:
- FedAvg: Diverging lines (higher malicious = higher loss)
- LVP: Converging lines (stable regardless of attacks)

**Bar Charts**:
- Clear separation between red (FedAvg) and green (LVP) bars
- Gap widens at 40% malicious

---

## 10. Limitations and Future Work

### 10.1 Known Limitations

1. **Client sample size**: Only 5 clients (small-scale federation)
2. **Attack diversity**: Only label flipping tested
3. **Model scope**: Five models (limited to time-series)
4. **Data domain**: Financial data only

### 10.2 Future Extensions

1. Test with 20-100 clients (large-scale)
2. Multiple attack strategies (noise, backdoor, model poisoning)
3. Adaptive attacks (adversary aware of LVP)
4. Cross-domain validation (healthcare, IoT, text)

---

## 11. Ethical Considerations

### 11.1 Data Privacy

- No personally identifiable information (PII) in transaction data
- Aggregated MCC categories only
- Federated learning preserves data locality

### 11.2 Responsible Disclosure

- Byzantine attack methods disclosed for research purposes
- Defense mechanisms (LVP) published simultaneously
- No real-world harm from simulated attacks

---

## 12. References

**Federated Learning**:
- McMahan et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data.

**Byzantine Robustness**:
- Blanchard et al. (2017). Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent.
- Yin et al. (2018). Byzantine-Robust Distributed Learning.

**Time-Series Models**:
- Box & Jenkins (1970). Time Series Analysis: Forecasting and Control.
- Durbin & Koopman (2012). Time Series Analysis by State Space Methods.

---

**Document Version**: 1.0  
**Last Updated**: December 10, 2025  
**Authors**: [Your research team]  
**Contact**: [your email]
