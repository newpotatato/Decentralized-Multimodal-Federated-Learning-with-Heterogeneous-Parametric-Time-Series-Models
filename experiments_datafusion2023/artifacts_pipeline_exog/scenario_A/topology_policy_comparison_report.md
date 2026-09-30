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
- lvp: 1.110076 +- 0.231129
- decentralized_fedavg: 1.132593 +- 0.245967
- defta: 1.120921 +- 0.239561
- balance: 1.110499 +- 0.233729
- push_sum: 1.133205 +- 0.245236

Boxplot: c:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_exog\scenario_A\plots\boxplots\fig3_boxplot_single_multiseed.png
Boxplot report: c:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_exog\scenario_A\plots\boxplots\fig3_boxplot_with_vs_without_fedavg_multiseed_report.md