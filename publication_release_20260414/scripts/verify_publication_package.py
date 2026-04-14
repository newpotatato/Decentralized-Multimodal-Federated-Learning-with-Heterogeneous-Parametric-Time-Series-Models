from __future__ import annotations

from pathlib import Path
import sys

REQUIRED_PATHS = [
    Path("federated_learning/core/decentralized_consensus.py"),
    Path("federated_learning/data_loaders/data_utils.py"),
    Path("federated_learning/experiments/compare_method_topologies_multiseed.py"),
    Path("federated_learning/experiments/fig3_boxplot_with_without_fedavg_multiseed.py"),
    Path("federated_learning/experiments/ablate_tau_alpha_article_rounds10.py"),
    Path("federated_learning/artifacts/article_package_current_run_20260410/plots/ablation/tau_ablation_rounds10_article.png"),
    Path("federated_learning/artifacts/article_package_current_run_20260410/plots/boxplots/lvpfl_fedavg_only/fig3_boxplot_single_multiseed.png"),
    Path("federated_learning/artifacts/article_package_current_run_20260410/plots/boxplots/all_methods_no_pushsum/fig3_boxplot_single_multiseed.png"),
    Path("federated_learning/artifacts/article_package_current_run_20260410/plots/boxplots/lvpfl_pushsum_only/fig3_boxplot_single_multiseed.png"),
    Path("federated_learning/artifacts/article_package_current_run_20260410/optimum_topologi/topology_policy_comparison_summary.json"),
]


def main() -> int:
    missing = [p for p in REQUIRED_PATHS if not p.exists()]
    if missing:
        print("Missing required publication files:")
        for p in missing:
            print(f"- {p}")
        return 1

    print("Publication package verification passed.")
    print(f"Checked {len(REQUIRED_PATHS)} required paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
