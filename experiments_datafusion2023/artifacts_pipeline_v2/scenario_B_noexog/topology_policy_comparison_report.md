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
- lvp: 0.904236 +- 0.229335
- decentralized_fedavg: 2.754433 +- 0.318893
- defta: 2.582313 +- 0.230522
- balance: 0.858498 +- 0.348503
- push_sum: 2.754433 +- 0.318893
- local: 0.835523 +- 0.341361

Boxplot: C:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_v2\scenario_B_noexog\plots\boxplots\fig3_boxplot_single_multiseed.png
Boxplot report: C:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_v2\scenario_B_noexog\plots\boxplots\fig3_boxplot_with_vs_without_fedavg_multiseed_report.md