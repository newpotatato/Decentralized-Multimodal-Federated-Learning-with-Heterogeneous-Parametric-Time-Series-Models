# Topology Policy Comparison - All Graphs

## Source data
- Existing comparison data: `raw/seeds/seed42/fig3_decentralized_methods.json`
- LVP optimal-topology package for component alpha curves: `../experiment_with_more_rounds_hybrid_topology_selected_tau079/`

## Method comparison graphs (LVP hybrid vs baselines on structural graphs)

1. Dynamics curve (network MAE vs round)
- `plots/multiseed/fig3_decentralized_methods_multiseed_network_mae.png`
- Report: `plots/multiseed/fig3_decentralized_methods_multiseed_report.md`

2. Coherence curve (component DeltaL2 vs round)
- `plots/coherence/fig3_decentralized_coherence_multiseed.png`
- Report: `plots/coherence/fig3_decentralized_coherence_multiseed_report.md`

3. Boxplot pair (with and without FedAvg panel)
- `plots/boxplots_pair/fig3_boxplot_with_vs_without_fedavg_multiseed.png`
- Report: `plots/boxplots_pair/fig3_boxplot_with_vs_without_fedavg_multiseed_report.md`

4. Single-panel boxplot (previously generated)
- `plots/boxplots/fig3_boxplot_single_multiseed.png`
- Report: `plots/boxplots/fig3_boxplot_with_vs_without_fedavg_multiseed_report.md`

## LVP component-wise alpha curves (two title variants)

Generated for optimal LVP topology package (`tau=0.79`) with alpha grid:
- 0.15, 0.25, 0.35, 0.45, 0.53, 0.60

Component plot sets:
1. With node-count in title
- `../experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/with_size_in_title/component_01.png`
- `../experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/with_size_in_title/component_02.png`
- `../experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/with_size_in_title/component_03.png`
- `../experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/with_size_in_title/component_04.png`

2. Without node-count in title
- `../experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/without_size_in_title/component_01.png`
- `../experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/without_size_in_title/component_02.png`
- `../experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/without_size_in_title/component_03.png`
- `../experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/without_size_in_title/component_04.png`

Metadata:
- `../experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/component_alpha_curves_summary.json`
- `../experiment_with_more_rounds_hybrid_topology_selected_tau079/plots/coherence/components/component_alpha_curves_report.md`
