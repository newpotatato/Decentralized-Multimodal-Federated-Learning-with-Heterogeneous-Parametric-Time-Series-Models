# Reproducing Byzantine Resilience Experiments

This document provides complete instructions for reproducing the Byzantine resilience experiments for our scientific publication.

## Overview

Our experiments demonstrate that **LVP (Limited Vector Projection)** aggregation is significantly more robust to Byzantine attacks than standard **FedAvg** when applied to federated learning on financial time-series data.

### Key Results
- **5 time-series models** tested: ARMAX, DynamicLinear, KalmanFilter, StructuralTS, MarkovReg
- **Byzantine attack scenarios**: 0%, 20%, 40% malicious clients
- **Attack strategy**: Label flipping with scale factor 2.5
- **LVP improvement**: 20-70% lower final loss compared to FedAvg under attack
- **Statistical validation**: 5 random seeds for confidence intervals

---

## Prerequisites

### System Requirements
- Python 3.8+
- 8GB RAM minimum (16GB recommended)
- 2-4 hours for full experiments (5 seeds)
- 10-15 minutes for quick smoke test

### Data Requirements

The experiments use real financial transaction data:

```
01_data_transactions/
├── dat_mcc.csv              # MCC transaction data (required)
├── moex_*.csv               # MOEX stock data (optional)
└── ...

02_data_fontanka/
└── fontanka_news_result.csv # News sentiment (optional, for exogenous factors)
```

**Note**: If data files are missing, synthetic data will be generated automatically.

---

## Installation

### 1. Install Python Dependencies

```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install requirements
pip install -r requirements_publication.txt
```

### 2. Verify Installation

```powershell
python -c "import numpy, pandas, matplotlib, sklearn, scipy, statsmodels; print('✓ All dependencies installed')"
```

---

## Quick Start (5 minutes)

Run a fast smoke test to verify everything works:

```powershell
python reproduce_experiments.py --mode quick
```

This runs a minimal configuration:
- 2 fast models (DynamicLinear, MarkovReg)
- 2 malicious fractions (0%, 20%)
- 5 rounds per experiment
- 1 random seed

**Expected output**:
```
[2025-12-10 10:30:00] Starting experiment reproduction - Mode: quick
[2025-12-10 10:30:05] Running experiment with seed=42
[2025-12-10 10:35:12] ✓ Completed seed=42
[2025-12-10 10:35:15] ✓ Quick test completed successfully
Results saved to: publication_results
```

---

## Full Reproduction (Paper Figures)

### Option 1: Generate All Results from Scratch

Run complete experiments with 5 seeds for statistical significance:

```powershell
# Full experiments + publication figures + LaTeX tables
python reproduce_experiments.py --mode all
```

**What this does**:
1. Runs 5 experiments with different random seeds (42, 123, 456, 789, 2024)
2. Tests all 5 models × 3 malicious fractions × 2 aggregators = 30 configurations per seed
3. Aggregates results (mean, std, min, max across seeds)
4. Generates publication-quality figures (PNG + PDF)
5. Creates LaTeX tables with statistics

**Time estimate**: 2-4 hours

**Outputs**:
```
publication_results/
├── results_seed_42.json
├── results_seed_123.json
├── results_seed_456.json
├── results_seed_789.json
├── results_seed_2024.json
├── aggregated_results.json       # Combined statistics
├── figures/
│   ├── fig_byzantine_armaX_combined.pdf
│   ├── fig_byzantine_armaX_combined.png
│   ├── fig_byzantine_statespace_combined.pdf
│   ├── ... (15 figures total)
└── tables/
    ├── table_final_loss.tex
    ├── table_robustness.tex
    ├── table_convergence.tex
    └── all_tables.tex
```

### Option 2: Use Pre-Computed Results

If you already have `aggregated_results.json`:

```powershell
# Generate figures only
python reproduce_experiments.py --mode paper
```

This regenerates all figures and tables from existing data.

---

## Manual Step-by-Step

### Step 1: Run Individual Experiments

For maximum control, run experiments manually:

```powershell
cd 08_federated_learning

# Run the main experiment script
python test_5models.py
```

**Outputs**:
- `byzantine_model_armaX.png`
- `byzantine_model_statespace.png`
- `byzantine_model_kalman.png`
- `byzantine_model_structural.png`
- `byzantine_model_markov_reg.png`
- `byzantine_5models_results.json`

### Step 2: Generate Publication Figures

```powershell
# From repository root
python publication_plots.py 08_federated_learning/byzantine_5models_results.json
```

### Step 3: Generate LaTeX Tables

```powershell
python latex_tables.py publication_results/aggregated_results.json
```

---

## Experiment Configuration

### Modifying Parameters

Edit `reproduce_experiments.py` to change experimental setup:

```python
self.paper_config = {
    "n_clients": 5,                    # Number of federated clients
    "rounds": 8,                       # Training rounds
    "malicious_fracs": [0.0, 0.2, 0.4], # Attack intensities
    "models": [...],                   # Models to test
    "seeds": [42, 123, 456, 789, 2024], # Random seeds
}
```

### Attack Strategies

The default attack is **label flipping** (gradient inversion). To test other attacks, modify `08_federated_learning/v2/aggregators.py`:

```python
# Available attack strategies:
- "label_flip": Inverts gradients (multiplies by -scale)
- "noise": Adds Gaussian noise
- "random": Completely random parameters
```

---

## Interpreting Results

### Learning Curves (Left Panel)

- **Dashed lines** = FedAvg
- **Solid lines** = LVP
- **Colors**: Blue (0%), Orange (20%), Purple (40% malicious)

**Key observation**: FedAvg lines go UP with more malicious clients (worse performance), while LVP lines stay flat (robust).

### Final Loss Bars (Right Panel)

- **Red bars** = FedAvg (higher = worse)
- **Green bars** = LVP (lower = better)
- **Yellow labels** = Exact loss values

**Key metric**: LVP should have 20-70% lower loss than FedAvg at 40% malicious.

### LaTeX Tables

#### Table 1: Final Loss Comparison
Shows mean ± std of final loss for all configurations. Lower is better.

#### Table 2: Robustness Metrics
Growth rate = (Loss_40% - Loss_0%) / Loss_0%

**Interpretation**:
- FedAvg growth: 200-400% (very vulnerable)
- LVP growth: 0-50% (robust)
- Ratio: 5-20× (LVP much more stable)

#### Table 3: Convergence Speed
Number of rounds to reach 90% of final loss. Lower = faster convergence.

---

## Troubleshooting

### Issue: ImportError for models

**Solution**: Ensure you're running from the repository root:
```powershell
cd c:\Users\Кирилл\Downloads\RNF
python reproduce_experiments.py --mode quick
```

### Issue: Data files not found

**Solution**: The scripts will use synthetic data automatically. For real data, ensure:
```powershell
Test-Path "01_data_transactions\dat_mcc.csv"  # Should return True
```

### Issue: Experiments too slow

**Solution**: Use quick mode or reduce seeds:
```powershell
# Edit reproduce_experiments.py
self.paper_config["seeds"] = [42, 123]  # Only 2 seeds instead of 5
```

### Issue: Out of memory

**Solution**: Reduce number of clients or use smaller models:
```python
self.paper_config["models"] = ["statespace", "markov_reg"]  # Fast models only
```

---

## Validation Checklist

Before submitting your paper, verify:

- [ ] All 5 models completed successfully
- [ ] 5 seeds run for each configuration
- [ ] Figures saved as PDF (vector graphics for LaTeX)
- [ ] Tables include error bars (mean ± std)
- [ ] LVP consistently outperforms FedAvg at 40% malicious
- [ ] Convergence curves show clear separation
- [ ] Data sources documented in paper

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourpaper2025,
  title={Byzantine-Resilient Federated Learning for Financial Time Series},
  author={Your Name},
  journal={Your Journal},
  year={2025},
  note={Code: github.com/yourrepo}
}
```

---

## File Structure

```
RNF/
├── reproduce_experiments.py      # Master reproduction script
├── publication_plots.py          # Figure generation
├── latex_tables.py               # Table generation
├── requirements_publication.txt  # Python dependencies
├── README_REPRODUCTION.md        # This file
│
├── 08_federated_learning/
│   ├── test_5models.py          # Main experiment script
│   ├── test_byzantine_sliding_window.py  # Helper functions
│   │
│   └── v2/                       # Advanced experiments
│       ├── run_real_experiments.py
│       ├── analyze_and_plot.py
│       ├── aggregators.py       # FedAvg, LVP implementations
│       └── data_utils.py
│
└── publication_results/          # Generated by reproduction scripts
    ├── aggregated_results.json
    ├── figures/
    └── tables/
```

---

## Contact

For questions about reproduction:
1. Check Issues on GitHub
2. Review experiment logs in `publication_results/`
3. Contact: [your email]

---

**Last Updated**: December 10, 2025  
**Version**: 1.0  
**Tested on**: Windows 10/11, Python 3.9-3.11
