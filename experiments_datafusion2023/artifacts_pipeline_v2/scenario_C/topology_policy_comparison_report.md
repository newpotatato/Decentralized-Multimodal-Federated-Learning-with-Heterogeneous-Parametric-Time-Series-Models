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
- lvp: 1.021347 +- 0.379191
- decentralized_fedavg: 350469549422787.625000 +- 592602167887574.375000
- defta: 6990392.360658 +- 13980782.422596
- balance: 0.867346 +- 0.360708
- push_sum: 350469549422787.625000 +- 592602167887574.375000
- local: 0.842032 +- 0.343307

Boxplot: C:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_v2\scenario_C\plots\boxplots\fig3_boxplot_single_multiseed.png
Boxplot report: C:\tarasova\RNF\experiments_datafusion2023\artifacts_pipeline_v2\scenario_C\plots\boxplots\fig3_boxplot_with_vs_without_fedavg_multiseed_report.md