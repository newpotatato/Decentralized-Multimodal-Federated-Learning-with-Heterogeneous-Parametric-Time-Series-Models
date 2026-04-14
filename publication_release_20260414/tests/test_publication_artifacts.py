from __future__ import annotations

from pathlib import Path


def test_key_publication_artifacts_exist() -> None:
    required = [
        Path("federated_learning/artifacts/article_package_current_run_20260410/plots/ablation/tau_ablation_rounds10_article.png"),
        Path("federated_learning/artifacts/article_package_current_run_20260410/plots/boxplots/lvpfl_fedavg_only/fig3_boxplot_single_multiseed.png"),
        Path("federated_learning/artifacts/article_package_current_run_20260410/plots/boxplots/all_methods_no_pushsum/fig3_boxplot_single_multiseed.png"),
        Path("federated_learning/artifacts/article_package_current_run_20260410/plots/boxplots/lvpfl_pushsum_only/fig3_boxplot_single_multiseed.png"),
        Path("federated_learning/artifacts/article_package_current_run_20260410/optimum_topologi/topology_policy_comparison_summary.json"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    assert not missing, f"Missing artifacts: {missing}"
