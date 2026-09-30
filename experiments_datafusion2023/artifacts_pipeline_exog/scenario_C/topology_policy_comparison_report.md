# Method Topology Policy Comparison

Model: DynamicLinearModel
Seeds: 42, 52, 62
LVP tau: 0.7

## Topology map
- lvp: hybrid
- decentralized_fedavg: ring
- defta: star
- balance: line
- push_sum: ring

## Final MAE mean +- std
- lvp: 1.123600 +- 0.214306
- decentralized_fedavg: 1.468962 +- 0.192803
- defta: 1.397468 +- 0.109385
- balance: 1.117954 +- 0.241966
- push_sum: 1.468962 +- 0.192803

Boxplot: c:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_exog\scenario_C\plots\boxplots\fig3_boxplot_single_multiseed.png
Boxplot report: c:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_exog\scenario_C\plots\boxplots\fig3_boxplot_with_vs_without_fedavg_multiseed_report.md