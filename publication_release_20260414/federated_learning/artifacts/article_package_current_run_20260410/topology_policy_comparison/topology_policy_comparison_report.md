# Method Topology Policy Comparison

Model: DynamicLinearModel
Seeds: 42
LVP tau: 0.79

## Topology map
- lvp: hybrid
- decentralized_fedavg: ring
- defta: star
- balance: line
- push_sum: ring

## Final MAE mean +- std
- lvp: 374451.627088 +- 0.000000
- decentralized_fedavg: 428294.094092 +- 0.000000
- defta: 386319.028385 +- 0.000000
- balance: 379243.474485 +- 0.000000
- push_sum: 428293.384603 +- 0.000000

Boxplot: federated_learning\artifacts\article_package_current_run_20260410\topology_policy_comparison\plots\boxplots\fig3_boxplot_single_multiseed.png
Boxplot report: federated_learning\artifacts\article_package_current_run_20260410\topology_policy_comparison\plots\boxplots\fig3_boxplot_with_vs_without_fedavg_multiseed_report.md