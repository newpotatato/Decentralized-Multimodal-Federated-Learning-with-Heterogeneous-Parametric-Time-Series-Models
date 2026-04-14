# Method Topology Policy Comparison

Model: DynamicLinearModel
Seeds: 42, 52, 62
LVP tau: 0.79

## Topology map
- lvp: hybrid
- decentralized_fedavg: ring
- defta: star
- balance: line
- push_sum: ring

## Final MAE mean +- std
- lvp: 385718.367070 +- 668.775514
- decentralized_fedavg: 119921633903609919245189358966343204864.000000 +- 97092091867788779315533042337536016384.000000
- defta: 510886230498640740521300019532486103207954612224.000000 +- 722502235954181570726661406465497404555170676736.000000
- balance: 4267462264339650523135725069049831378097339145223022423933827523889663832956361173536604160.000000 +- 3782587517463450922672742309951907330183905110087933325460274640736335623680484353142423552.000000
- push_sum: 29534933147503878625667587132988304916480.000000 +- 0.000000

Boxplot: federated_learning\artifacts\article_package_current_run_20260410\optimum_topologi\plots\boxplots\fig3_boxplot_single_multiseed.png
Boxplot report: federated_learning\artifacts\article_package_current_run_20260410\optimum_topologi\plots\boxplots\fig3_boxplot_with_vs_without_fedavg_multiseed_report.md