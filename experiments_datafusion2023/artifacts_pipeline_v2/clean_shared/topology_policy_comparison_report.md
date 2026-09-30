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
- lvp: 0.862886 +- 0.352095
- decentralized_fedavg: 0.960401 +- 0.348347
- defta: 0.946194 +- 0.356718
- balance: 0.882227 +- 0.364322
- push_sum: 0.960401 +- 0.348347
- local: 0.842032 +- 0.343307

Boxplot: C:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_v2\clean_shared\plots\boxplots\fig3_boxplot_single_multiseed.png
Boxplot report: C:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_v2\clean_shared\plots\boxplots\fig3_boxplot_with_vs_without_fedavg_multiseed_report.md