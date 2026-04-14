# 📋 Byzantine Resilience - Publication Package Index

## 🎯 Start Here

**New to this package?** → This is your navigation guide

**Want to run experiments?** → Run `.\run_quick_test.ps1`

**For paper writing?** → See `LATEX_INTEGRATION_GUIDE.tex`

---

## 📁 Complete File Index

### 🚀 Executable Scripts

| File | Purpose | Usage |
|------|---------|-------|
| **`reproduce_experiments.py`** | Master reproduction script | `python reproduce_experiments.py --mode quick` |
| **`publication_plots.py`** | Generate publication figures | `python publication_plots.py results.json` |
| **`latex_tables.py`** | Generate LaTeX tables | `python latex_tables.py results.json` |
| **`run_quick_test.ps1`** | Quick verification (10 min) | `.\run_quick_test.ps1` |

### 📚 Documentation Files

| File | Content | Read When |
|------|---------|-----------|
| **`README.md`** | Repository overview | First visit |
| **`README_REPRODUCTION.md`** | Complete step-by-step guide | Running experiments |
| **`EXPERIMENT_METHODOLOGY.md`** | Scientific methodology | Writing methods section |
| **`LATEX_INTEGRATION_GUIDE.tex`** | LaTeX examples | Writing paper |
| **`requirements_publication.txt`** | Python dependencies | Installation |

### 🔬 Experiment Code

| Location | Description |
|----------|-------------|
| `federated_learning/experiments_main.py` | Main experiment runner |
| `federated_learning/core/aggregators.py` | FedAvg & LVP implementations |
| `federated_learning/models/` | Time-series models |
| `prediction_models/` | Base model implementations |

### 📊 Generated Outputs (After Running)

```
publication_results/
├── aggregated_results.json          # Combined statistics
├── results_seed_42.json             # Individual seed results
├── results_seed_123.json
├── results_seed_456.json
├── results_seed_789.json
├── results_seed_2024.json
├── figures/                         # Publication-ready plots
│   ├── fig_byzantine_armaX_combined.pdf
│   ├── fig_byzantine_armaX_combined.png
│   └── ... (15+ figures for all models)
└── tables/                          # LaTeX tables
    ├── table_final_loss.tex
    ├── table_robustness.tex
    ├── table_convergence.tex
    └── all_tables.tex
```

---

## 🎓 Workflow Paths

### Path 1: Quick Verification (10 minutes)

```powershell
1. Install dependencies:
   pip install -r requirements_publication.txt

2. Run quick test:
   .\run_quick_test.ps1

3. Check results:
   quick_test_results/
```

### Path 2: Full Reproduction (2-4 hours)

```powershell
1. Read: README.md
2. Run: python reproduce_experiments.py --mode all
3. Get: publication_results/ with all figures and tables
4. Use: Copy PDFs to your paper
```

### Path 3: Paper Integration

```latex
1. Read: LATEX_INTEGRATION_GUIDE.tex
2. Copy: figures/*.pdf to your paper directory
3. Include: \input{tables/table_final_loss.tex}
4. Write: Methods section using EXPERIMENT_METHODOLOGY.md
```

---

## 🗺️ Navigation Guide

### "I want to..."

| Goal | Go To |
|------|-------|
| Understand the package | `README.md` |
| Run experiments | `README_REPRODUCTION.md` |
| Understand methodology | `EXPERIMENT_METHODOLOGY.md` |
| Write paper methods | `EXPERIMENT_METHODOLOGY.md` sections 2-5 |
| Include figures in LaTeX | `LATEX_INTEGRATION_GUIDE.tex` |
| Troubleshoot errors | `README_REPRODUCTION.md` section 7 |
| Check dependencies | `requirements_publication.txt` |

---

## 📊 How Your Figures Were Generated

The 5 figures in your attachments were created by:

```python
# Main experiment script
python federated_learning/experiments_main.py

# Output:
byzantine_model_armaX.png
byzantine_model_statespace.png  
byzantine_model_kalman.png
byzantine_model_structural.png
byzantine_model_markov_reg.png
```

**Process**:
1. Load MCC transaction data + news sentiment
2. Create 20 heterogeneous clients via K-means
3. Select 5 random clients per experiment
4. Run 8 rounds of federated learning
5. Test Byzantine attack intensities: 0%, 20%, 40%
6. Compare FedAvg (vulnerable) vs LVP (robust)
7. Visualize learning curves + final loss comparison

---

## 🔗 File Dependencies

```
reproduce_experiments.py
    ├── calls → federated_learning/experiments_main.py
    ├── calls → publication_plots.py
    └── calls → latex_tables.py

publication_plots.py
    ├── reads → publication_results/aggregated_results.json
    └── creates → publication_results/figures/*.pdf

latex_tables.py
    ├── reads → publication_results/aggregated_results.json
    └── creates → publication_results/tables/*.tex
```

---

## ✅ Pre-Submission Checklist

- [ ] Ran `.\run_quick_test.ps1` successfully
- [ ] Ran `python reproduce_experiments.py --mode all`
- [ ] Verified all 15+ figures generated (PDF format)
- [ ] Checked tables have mean ± std values
- [ ] Confirmed LVP outperforms FedAvg at 40% malicious
- [ ] Added figures to paper LaTeX
- [ ] Added tables to paper LaTeX
- [ ] Wrote methods section using EXPERIMENT_METHODOLOGY.md
- [ ] Added code availability statement
- [ ] Cited random seeds: [42, 123, 456, 789, 2024]

---

## 📦 Package Summary

| Category | Count | Description |
|----------|-------|-------------|
| **Executable Scripts** | 4 | reproduce_experiments.py, publication_plots.py, latex_tables.py, run_quick_test.ps1 |
| **Documentation** | 5 | README, reproduction guide, methodology, LaTeX guide, requirements |
| **Experiment Code** | 5+ | Main experiments, aggregators, models |
| **Generated Figures** | 15+ | PDF + PNG for all models |
| **Generated Tables** | 4 | LaTeX .tex files with statistics |

---

## 🆘 Getting Help

1. **Quick issues**: Check `README_REPRODUCTION.md` Troubleshooting
2. **Methodology questions**: Read `EXPERIMENT_METHODOLOGY.md`
3. **LaTeX issues**: See `LATEX_INTEGRATION_GUIDE.tex`
4. **Still stuck**: Check logs in `publication_results/`

---

## 🎯 Key Workflow

```
┌─────────────────────────────────────┐
│  1. Setup (5 min)                   │
│  pip install -r requirements.txt    │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  2. Quick Test (10 min)             │
│  .\run_quick_test.ps1               │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  3. Full Reproduction (2-4 hrs)     │
│  python reproduce_experiments.py    │
│         --mode all                  │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  4. Get Results                     │
│  publication_results/               │
│  ├── figures/ (PDFs)                │
│  └── tables/ (.tex)                 │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  5. Integrate in Paper              │
│  \includegraphics{fig_*.pdf}        │
│  \input{table_*.tex}                │
└─────────────────────────────────────┘
```

---

## 🌟 What Makes This Publication-Ready?

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

**Questions?** Read `README_REPRODUCTION.md`

**Version**: 1.0 | **Date**: December 2025 | **License**: MIT
