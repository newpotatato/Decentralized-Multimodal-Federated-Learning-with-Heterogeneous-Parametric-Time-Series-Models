# Publication Release (2026-04-14)

This folder is a curated, publication-oriented snapshot of the repository for decentralized federated learning experiments under adversarial settings.

The primary target of this release is the prepared experiment package:
- `federated_learning/artifacts/article_package_current_run_20260410`

## Scope

This release includes the code and artifacts required to:
1. Validate that key publication outputs are present and consistent.
2. Rebuild selected figures used in the article package.
3. Run smoke tests for loaders/core vectorization and artifact presence.

## Package Contents

- `federated_learning/core/` - consensus and model aggregation primitives.
- `federated_learning/data_loaders/` - data loading and preprocessing helpers.
- `federated_learning/experiments/` - experiment runners and figure generators.
- `federated_learning/artifacts/article_package_current_run_20260410/` - prepared publication artifacts.
- `01_data_transactions/` - dataset files required by loader-related tests.
- `tests/` - smoke and artifact tests for this release.
- `scripts/verify_publication_package.py` - required-file validation.
- `scripts/reproduce_key_figures.ps1` - rebuild selected publication figures.
- `scripts/run_publication_smoke.ps1` - one-command smoke validation.
- `docs/` - publication documentation/checklists.
- `requirements.txt` - dependency ranges.
- `requirements-lock.txt` - exact pinned environment snapshot.
- `LICENSE` - license for this package.
- `CITATION.cff` - citation metadata.

## Environment

- Python: 3.10+ (recommended)
- OS: validated on Windows/PowerShell

Create and activate a virtual environment, then install dependencies:

```powershell
pip install -r requirements.txt
```

For strict reproducibility use pinned versions:

```powershell
pip install -r requirements-lock.txt
```

## Quick Validation (Recommended First)

```powershell
python scripts/verify_publication_package.py
python -m pytest tests/test_publication_artifacts.py tests/test_param_vector.py tests/test_mcc_loader.py -q
```

Or run the bundled smoke script:

```powershell
./scripts/run_publication_smoke.ps1
```

## Rebuild Key Figures

```powershell
./scripts/reproduce_key_figures.ps1
```

This script rebuilds selected boxplots and ablation figures, then synchronizes the ablation image into the article package path:
- `federated_learning/artifacts/article_package_current_run_20260410/plots/ablation/tau_ablation_rounds10_article.png`

## Important Artifact Paths

- Main package root:
	- `federated_learning/artifacts/article_package_current_run_20260410/`
- Boxplots:
	- `federated_learning/artifacts/article_package_current_run_20260410/plots/boxplots/lvpfl_fedavg_only/fig3_boxplot_single_multiseed.png`
	- `federated_learning/artifacts/article_package_current_run_20260410/plots/boxplots/all_methods_no_pushsum/fig3_boxplot_single_multiseed.png`
	- `federated_learning/artifacts/article_package_current_run_20260410/plots/boxplots/lvpfl_pushsum_only/fig3_boxplot_single_multiseed.png`
- Ablation:
	- `federated_learning/artifacts/article_package_current_run_20260410/plots/ablation/tau_ablation_rounds10_article.png`
- Topology comparison summary:
	- `federated_learning/artifacts/article_package_current_run_20260410/optimum_topologi/topology_policy_comparison_summary.json`

## Reproducibility Notes

- This release intentionally keeps generated artifacts in-repo for direct publication usage.
- `requirements-lock.txt` captures the current resolved environment; use it for stable reruns.
- The parent repository may contain exploratory/non-publication files; this directory is the curated subset.

## Troubleshooting

- If `pytest` fails on loader tests, ensure `01_data_transactions/` is present in this release root.
- If a key figure is missing, run:
	- `./scripts/reproduce_key_figures.ps1`
- If an import error occurs, reinstall from lock file:
	- `pip install -r requirements-lock.txt`

## Citation and License

- Citation metadata: `CITATION.cff`
- License text: `LICENSE`
