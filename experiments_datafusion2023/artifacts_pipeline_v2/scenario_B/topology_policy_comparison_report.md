# Method Topology Policy Comparison

Model: DynamicLinearModel
Seeds: 42, 52, 62, 72, 82
LVP tau: 0.3

## Topology map
- lvp: hybrid
- decentralized_fedavg: hybrid
- defta: hybrid
- balance: hybrid
- push_sum: hybrid
- local: hybrid

## Final MAE mean +- std
- lvp: 1.021347 +- 0.379191
- decentralized_fedavg: 2.753638 +- 0.319338
- defta: 2.797137 +- 0.296925
- balance: 0.866659 +- 0.353182
- push_sum: 2.753638 +- 0.319338
- local: 0.842032 +- 0.343307

Boxplot: C:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_v2\scenario_B\plots\boxplots\fig3_boxplot_single_multiseed.png
Boxplot report: C:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_v2\scenario_B\plots\boxplots\fig3_boxplot_with_vs_without_fedavg_multiseed_report.md