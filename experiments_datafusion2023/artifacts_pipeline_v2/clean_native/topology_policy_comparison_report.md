# Method Topology Policy Comparison

Model: DynamicLinearModel
Seeds: 42, 52, 62, 72, 82
LVP tau: 0.3

## Topology map
- lvp: hybrid
- decentralized_fedavg: ring
- defta: star
- balance: line
- push_sum: ring
- local: ring

## Final MAE mean +- std
- lvp: 0.862886 +- 0.352095
- decentralized_fedavg: 0.947450 +- 0.385012
- defta: 0.867176 +- 0.353248
- balance: 0.884165 +- 0.366530
- push_sum: 0.947450 +- 0.385012
- local: 0.842032 +- 0.343307

Boxplot: C:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_v2\clean_native\plots\boxplots\fig3_boxplot_single_multiseed.png
Boxplot report: C:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_v2\clean_native\plots\boxplots\fig3_boxplot_with_vs_without_fedavg_multiseed_report.md