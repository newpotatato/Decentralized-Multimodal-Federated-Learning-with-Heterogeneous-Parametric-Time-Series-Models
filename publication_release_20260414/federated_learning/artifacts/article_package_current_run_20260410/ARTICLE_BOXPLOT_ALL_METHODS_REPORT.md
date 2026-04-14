# Boxplot With vs Without FedAvg (Multiseed)

Seeds used: 3 runs
Excluded aggregators: (none)
Single panel: True

## Final MAE mean +- std
- LVP: 383907.285866 +- 121.014504
- Decentralized FedAvg: 384041.387497 +- 254.326712
- DeFTA: 385560.878102 +- 1131.602999
- BALANCE: 387425.758338 +- 4672.879431
- Push-Sum: 383875.083257 +- 0.000000

## Topology Policy Comparison

This short benchmark compares LVP on its unique hybrid topology against baselines on distinct structural graphs.

Model: DynamicLinearModel
Seeds used: 1 run (seed 42)
Rounds: 3

Topology map:
- LVP: hybrid
- Decentralized FedAvg: ring
- DeFTA: star
- BALANCE: line
- Push-Sum: ring

## Final MAE mean +- std
- LVP: 374451.627088 +- 0.000000
- Decentralized FedAvg: 428294.094092 +- 0.000000
- DeFTA: 386319.028385 +- 0.000000
- BALANCE: 379243.474485 +- 0.000000
- Push-Sum: 428293.384603 +- 0.000000

Artifacts:
- topology policy summary: federated_learning/artifacts/article_package_current_run_20260410/topology_policy_comparison/topology_policy_comparison_summary.json
- topology policy report: federated_learning/artifacts/article_package_current_run_20260410/topology_policy_comparison/topology_policy_comparison_report.md
- boxplot: federated_learning/artifacts/article_package_current_run_20260410/topology_policy_comparison/plots/boxplots/fig3_boxplot_single_multiseed.png

## All Comparison Graphs (built from collected topology_policy_comparison data)

- Dynamics (MAE vs rounds): federated_learning/artifacts/article_package_current_run_20260410/topology_policy_comparison/plots/multiseed/fig3_decentralized_methods_multiseed_network_mae.png
- Dynamics report: federated_learning/artifacts/article_package_current_run_20260410/topology_policy_comparison/plots/multiseed/fig3_decentralized_methods_multiseed_report.md
- Coherence (DeltaL2 vs rounds): federated_learning/artifacts/article_package_current_run_20260410/topology_policy_comparison/plots/coherence/fig3_decentralized_coherence_multiseed.png
- Coherence report: federated_learning/artifacts/article_package_current_run_20260410/topology_policy_comparison/plots/coherence/fig3_decentralized_coherence_multiseed_report.md
- Boxplot pair (with/without FedAvg): federated_learning/artifacts/article_package_current_run_20260410/topology_policy_comparison/plots/boxplots_pair/fig3_boxplot_with_vs_without_fedavg_multiseed.png
- Boxplot pair report: federated_learning/artifacts/article_package_current_run_20260410/topology_policy_comparison/plots/boxplots_pair/fig3_boxplot_with_vs_without_fedavg_multiseed_report.md

## Component-wise Alpha Graphs (LVP on optimal tau=0.79 topology)

Alpha grid: 0.15, 0.25, 0.35, 0.45, 0.53, 0.60

Variant 1 (title includes node count):
- federated_learning/artifacts/article_package_current_run_20260410/experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/with_size_in_title/component_01.png
- federated_learning/artifacts/article_package_current_run_20260410/experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/with_size_in_title/component_02.png
- federated_learning/artifacts/article_package_current_run_20260410/experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/with_size_in_title/component_03.png
- federated_learning/artifacts/article_package_current_run_20260410/experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/with_size_in_title/component_04.png

Variant 2 (title without node count):
- federated_learning/artifacts/article_package_current_run_20260410/experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/without_size_in_title/component_01.png
- federated_learning/artifacts/article_package_current_run_20260410/experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/without_size_in_title/component_02.png
- federated_learning/artifacts/article_package_current_run_20260410/experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/without_size_in_title/component_03.png
- federated_learning/artifacts/article_package_current_run_20260410/experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/without_size_in_title/component_04.png

Metadata:
- federated_learning/artifacts/article_package_current_run_20260410/experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/component_alpha_curves_summary.json
- federated_learning/artifacts/article_package_current_run_20260410/experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/component_alpha_curves_report.md

Reference index:
- federated_learning/artifacts/article_package_current_run_20260410/topology_policy_comparison/ALL_GRAPHS_REPORT.md