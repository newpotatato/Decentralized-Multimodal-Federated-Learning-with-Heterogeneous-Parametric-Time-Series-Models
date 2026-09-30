# Method Topology Policy Comparison

Model: DynamicLinearModel
Seeds: 42, 52, 62
LVP tau: 0.7

## Topology map
- lvp: hybrid
- decentralized_fedavg: hybrid
- defta: hybrid
- balance: hybrid
- push_sum: hybrid

## Final MAE mean +- std
- lvp: 1.123600 +- 0.214306
- decentralized_fedavg: 1.393499 +- 0.380363
- defta: 1.174657 +- 0.263490
- balance: 1.103530 +- 0.234273
- push_sum: 1.586197 +- 0.225292

Boxplot: c:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_exog\scenario_B\plots\boxplots\fig3_boxplot_single_multiseed.png
Boxplot report: c:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_exog\scenario_B\plots\boxplots\fig3_boxplot_with_vs_without_fedavg_multiseed_report.md