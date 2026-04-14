# Best Topology Per Method - Multiseed

Model: DynamicLinearModel
Methods: lvp, decentralized_fedavg, defta, balance, push_sum
Selection seed: 42
Eval seeds: 42, 52, 62

## Selected topology per method

| Method | Mode | Tau | Selection MAE |
|---|---:|---:|---:|
| lvp | jaccard_cosine_hybrid | 0.7900 | 375247.978493 |
| decentralized_fedavg | jaccard | 0.5700 | 377887.009684 |
| defta | jaccard | 0.7900 | 383814.917189 |
| balance | jaccard | 0.5700 | 381422.687991 |
| push_sum | jaccard | 0.5700 | 382834.149885 |

Boxplot: federated_learning\artifacts\article_package_current_run_20260410\best_topology_per_method\plots\boxplots\fig3_boxplot_single_multiseed.png
Boxplot report: federated_learning\artifacts\article_package_current_run_20260410\best_topology_per_method\plots\boxplots\fig3_boxplot_with_vs_without_fedavg_multiseed_report.md