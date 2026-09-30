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
- lvp: 1.278908 +- 0.645170
- decentralized_fedavg: 0.891938 +- 0.269002
- defta: 1.087708 +- 0.405813
- balance: 1.463612 +- 0.523731
- push_sum: 0.871636 +- 0.266251
- local: 0.842032 +- 0.343307

Boxplot: C:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_v2\scenario_A\plots\boxplots\fig3_boxplot_single_multiseed.png
Boxplot report: C:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_v2\scenario_A\plots\boxplots\fig3_boxplot_with_vs_without_fedavg_multiseed_report.md