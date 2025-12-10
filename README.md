# Byzantine-Resilient Federated Learning for Financial Time Series

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)]()

> **International Research Publication Package**  
> Complete reproducibility system for Byzantine-resilient federated learning experiments

---

## Publication Package

**To reproduce our scientific experiments**, start here:

### [START_HERE.md](START_HERE.md) ← **Begin Here**

This repository contains a complete publication-ready package for reproducing Byzantine resilience experiments in federated learning on financial time-series data.

---

##  Quick Overview

This research demonstrates that **LVP (Limited Vector Projection)** aggregation significantly outperforms standard **FedAvg** under Byzantine attacks:

| Metric | FedAvg (40% attack) | LVP (40% attack) | Improvement |
|--------|---------------------|------------------|-------------|
| Average Loss | 23.9 | 7.2 | **70%** better |
| Robustness | Vulnerable | Robust | 5-20× more stable |
| Convergence | Degrades | Maintains | Consistent |

**Key Finding**: LVP maintains performance even with 40% malicious clients, while FedAvg performance degrades severely.

---

## Quick Start

### Option 1: Quick Test (10 minutes)

```powershell
# Install dependencies
pip install -r requirements_publication.txt

# Run quick verification
.\run_quick_test.ps1
```

### Option 2: Full Reproduction (2-4 hours)

```powershell
# Run complete experiments with 5 random seeds
python reproduce_experiments.py --mode all
```

**Output**:
- 15+ publication-quality figures (PDF + PNG)
- LaTeX tables with statistics
- Aggregated results with confidence intervals

---

## 📁 Repository Structure

```
RNF/
├── 📍 START_HERE.md                      ⭐ Navigation & package index
├── 📖 README_REPRODUCTION.md              Complete step-by-step guide
├── 🔬 EXPERIMENT_METHODOLOGY.md           Scientific methodology
├── 📝 LATEX_INTEGRATION_GUIDE.tex         LaTeX integration examples
│
├── ⚙️  reproduce_experiments.py           Master reproduction script
├── 📊 publication_plots.py                Generate publication figures
├── 📋 latex_tables.py                     Generate LaTeX tables
├── ✅ run_quick_test.ps1                  Quick verification
│
├── federated_learning/                   Experiment implementations
│   ├── experiments_main.py               Main experiments
│   ├── core/
│   │   ├── aggregators.py                FedAvg & LVP implementations
│   │   └── evaluators.py
│   └── models/                           Time-series models
│
├── data/                                 Financial datasets
│   ├── transactions/                     MCC transaction data
│   └── news/                             News sentiment data
│
└── prediction_models/                    Base model implementations
```

---

## 🔬 Experimental Design

### Data
- **Primary**: MCC transaction data (merchant category codes)
- **Auxiliary**: News sentiment scores (exogenous factors)
- **Distribution**: K-means clustering (k=20) for non-IID partitions

### Models Tested (5)
1. **ARMAX** - Autoregressive with exogenous factors
2. **DynamicLinear** - State-space with Kalman filtering
3. **KalmanFilter** - Classical optimal filtering
4. **StructuralTS** - Decomposable time-series
5. **MarkovReg** - Regime-switching regression

### Byzantine Attack
- **Strategy**: Label flipping (gradient inversion)
- **Scale**: 2.5× amplification factor
- **Intensities**: 0%, 20%, 40% malicious clients

### Aggregation Methods
- **FedAvg**: Standard weighted averaging (baseline, vulnerable)
- **LVP**: Limited Vector Projection (proposed, robust)

---

## 📈 Key Results

At 40% malicious clients, LVP achieves:
- **ARMAX**: 69% improvement (19.9 → 6.1 MSE)
- **DynamicLinear**: 71% improvement (26.2 → 7.7 MSE)
- **KalmanFilter**: 65% improvement (16.4 → 5.7 MSE)
- **StructuralTS**: 72% improvement (30.8 → 8.6 MSE)
- **MarkovReg**: 71% improvement (26.8 → 7.8 MSE)

---

## 📋 Documentation

| Document | Purpose |
|----------|---------|
| [START_HERE.md](START_HERE.md) | **Start here** - Navigation & quick links |
| [README_REPRODUCTION.md](README_REPRODUCTION.md) | Complete step-by-step reproduction guide |
| [EXPERIMENT_METHODOLOGY.md](EXPERIMENT_METHODOLOGY.md) | Scientific methodology for paper |
| [LATEX_INTEGRATION_GUIDE.tex](LATEX_INTEGRATION_GUIDE.tex) | LaTeX paper integration examples |

---

## 🎯 Use Cases

### For Researchers
```powershell
# Reproduce experiments
python reproduce_experiments.py --mode all

# Results in: publication_results/
#   ├── figures/  (15+ PDFs)
#   └── tables/   (LaTeX .tex)
```

### For Paper Authors
```latex
% Include figures
\includegraphics{figures/fig_byzantine_armaX_combined.pdf}

% Include tables
\input{tables/table_final_loss.tex}
```

### For Reviewers
```powershell
# Quick verification (10 minutes)
.\run_quick_test.ps1
```

---

## 📦 Requirements

- **Python**: 3.8 or higher
- **Libraries**: numpy, pandas, matplotlib, scikit-learn, scipy, statsmodels
- **Install**: `pip install -r requirements_publication.txt`

---

## 📝 Citation

```bibtex
@article{yourpaper2025,
  title={Byzantine-Resilient Federated Learning for Financial Time Series},
  author={Your Name and Collaborators},
  journal={Your Journal},
  year={2025},
  url={https://github.com/yourrepo},
  note={Reproduction package available}
}
```

---

## 🌟 Highlights

✅ **Fully Reproducible** - Fixed random seeds, documented parameters  
✅ **Publication Ready** - PDF figures, LaTeX tables included  
✅ **Statistically Valid** - 5 seeds with confidence intervals  
✅ **Well Documented** - Comprehensive guides for all use cases  
✅ **Quick Testing** - 10-minute verification available  
✅ **Real Data** - Uses actual financial transaction data  

---

## 📧 Contact

**Research Questions**: [your-email@domain.com]  
**Technical Issues**: Open a GitHub issue  
**Methodology Details**: See [EXPERIMENT_METHODOLOGY.md](EXPERIMENT_METHODOLOGY.md)

---

## 🚀 Getting Started Now

```powershell
# 1. Clone repository
git clone [your-repo-url]
cd RNF

# 2. Install dependencies
pip install -r requirements_publication.txt

# 3. Quick test (10 min)
.\run_quick_test.ps1

# 4. Full reproduction (2-4 hours)
python reproduce_experiments.py --mode all
```

**Ready?** → Read [START_HERE.md](START_HERE.md)

---

*Last Updated: December 10, 2025*  
*Version: 1.0*  
*Status: ✅ Production Ready*
