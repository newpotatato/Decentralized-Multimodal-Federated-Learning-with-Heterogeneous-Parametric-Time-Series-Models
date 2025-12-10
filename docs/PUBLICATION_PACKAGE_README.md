# Byzantine Resilience Experiments - Publication Package

This directory contains all scripts and documentation needed to reproduce the Byzantine resilience experiments for scientific publication.

## 📁 Package Contents

### Core Scripts
- `reproduce_experiments.py` - Master script for running all experiments
- `publication_plots.py` - Generate publication-quality figures (PDF + PNG)
- `latex_tables.py` - Generate LaTeX tables with statistics
- `run_quick_test.ps1` - Quick verification script (5-10 minutes)

### Documentation
- `README_REPRODUCTION.md` - Complete reproduction guide
- `EXPERIMENT_METHODOLOGY.md` - Scientific methodology documentation
- `requirements_publication.txt` - Python dependencies

### Experiment Code
- `08_federated_learning/test_5models.py` - Main experiment implementation
- `08_federated_learning/v2/aggregators.py` - FedAvg and LVP implementations
- `08_federated_learning/v2/run_real_experiments.py` - Advanced experiments

## 🚀 Quick Start (5 minutes)

### Windows PowerShell

```powershell
# 1. Install dependencies
pip install -r requirements_publication.txt

# 2. Run quick test
.\run_quick_test.ps1

# OR manually:
python reproduce_experiments.py --mode quick
```

### Expected Output
```
[2025-12-10 10:00:00] Starting experiment reproduction - Mode: quick
[2025-12-10 10:05:30] ✓ Completed seed=42
[2025-12-10 10:05:35] ✓ Quick test completed successfully
Results saved to: publication_results
```

## 📊 Full Reproduction (2-4 hours)

Generate all paper figures with statistical validation:

```powershell
python reproduce_experiments.py --mode all
```

**This creates**:
- 5 experiment runs (different random seeds)
- 15+ publication figures (PNG + PDF)
- LaTeX tables with confidence intervals
- Aggregated statistical results

**Output structure**:
```
publication_results/
├── aggregated_results.json       # Combined statistics
├── figures/
│   ├── fig_byzantine_armaX_combined.pdf
│   ├── fig_byzantine_statespace_combined.pdf
│   └── ... (15 figures total)
└── tables/
    ├── table_final_loss.tex
    ├── table_robustness.tex
    ├── table_convergence.tex
    └── all_tables.tex
```

## 📈 Key Results

Your reproduced figures should show:

### Learning Curves (Left Panel)
- **Dashed lines** = FedAvg (vulnerable)
- **Solid lines** = LVP (robust)
- **Pattern**: FedAvg loss increases with more malicious clients, LVP stays stable

### Final Loss Comparison (Right Panel)
- **Red bars** = FedAvg (higher loss = worse)
- **Green bars** = LVP (lower loss = better)
- **Expected**: LVP 20-70% better than FedAvg at 40% malicious

## 🔬 Experimental Design

### Data
- **Primary**: MCC transaction data (`01_data_transactions/dat_mcc.csv`)
- **Auxiliary**: News sentiment (`02_data_fontanka/fontanka_news_result.csv`)
- **Fallback**: Synthetic data if real data unavailable

### Models Tested (5)
1. ARMAX - Autoregressive with exogenous factors
2. DynamicLinear - State-space model
3. KalmanFilter - Classical filtering
4. StructuralTS - Decomposable time series
5. MarkovReg - Regime-switching regression

### Byzantine Attack
- **Strategy**: Label flipping (gradient inversion)
- **Scale**: 2.5× amplification
- **Intensities**: 0%, 20%, 40% malicious clients

### Aggregation Methods
- **FedAvg**: Standard weighted averaging (baseline)
- **LVP**: Limited Vector Projection (proposed robust method)

## 📋 Reproducibility Checklist

Before submitting your paper:

- [ ] Ran `.\run_quick_test.ps1` successfully
- [ ] Generated full results with 5 seeds
- [ ] Verified all figures saved as PDF (vector graphics)
- [ ] Checked tables include mean ± std
- [ ] Confirmed LVP outperforms FedAvg at 40% malicious
- [ ] Documented random seeds: [42, 123, 456, 789, 2024]

## 📝 Citation

```bibtex
@article{yourpaper2025,
  title={Byzantine-Resilient Federated Learning for Financial Time Series},
  author={Your Name and Collaborators},
  journal={Your Journal},
  year={2025},
  note={Reproduction package: github.com/yourrepo}
}
```

## 📖 Documentation

- **Quick start**: See this file
- **Full guide**: Read `README_REPRODUCTION.md`
- **Methodology**: Read `EXPERIMENT_METHODOLOGY.md`
- **Troubleshooting**: See `README_REPRODUCTION.md` section 7

## 🔧 Common Issues

### "Data file not found"
→ Scripts will use synthetic data automatically. For real data, ensure `01_data_transactions/dat_mcc.csv` exists.

### "Import error: No module named..."
→ Run `pip install -r requirements_publication.txt`

### "Experiments too slow"
→ Use `--mode quick` for fast testing, or edit `reproduce_experiments.py` to reduce seeds

### "Out of memory"
→ Reduce number of clients or test fewer models (edit configuration in scripts)

## 📧 Support

- Check experiment logs in `publication_results/`
- Review `README_REPRODUCTION.md` troubleshooting section
- Open issue on GitHub

## 🎯 Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│                  REPRODUCTION WORKFLOW                       │
└─────────────────────────────────────────────────────────────┘

1. SETUP (5 minutes)
   ├─ Install Python 3.8+
   ├─ pip install -r requirements_publication.txt
   └─ Verify data files

2. QUICK TEST (10 minutes)
   ├─ .\run_quick_test.ps1
   └─ Verify output in quick_test_results/

3. FULL EXPERIMENTS (2-4 hours)
   ├─ python reproduce_experiments.py --mode all
   └─ Monitor progress in publication_results/

4. VALIDATE RESULTS
   ├─ Check figures/: 15+ PDFs generated
   ├─ Check tables/: LaTeX .tex files
   └─ Verify LVP outperforms FedAvg

5. USE IN PAPER
   ├─ Copy figures/*.pdf to your LaTeX paper
   ├─ \input{tables/table_final_loss.tex}
   └─ Cite methodology in paper
```

## 🌟 What Makes This Reproducible?

✅ **Complete code** - All experiment scripts included  
✅ **Fixed seeds** - [42, 123, 456, 789, 2024] for reproducibility  
✅ **Documented parameters** - Every hyperparameter specified  
✅ **Synthetic fallback** - Works without proprietary data  
✅ **Multiple formats** - Figures in PNG + PDF  
✅ **Statistical validation** - 5 seeds with confidence intervals  
✅ **Version control** - requirements.txt with exact versions  
✅ **Clear methodology** - Step-by-step documentation  

---

**Ready to reproduce?** Start with: `.\run_quick_test.ps1`

**Questions?** Read `README_REPRODUCTION.md` or `EXPERIMENT_METHODOLOGY.md`

**Version**: 1.0 | **Date**: December 2025 | **License**: [Your License]
