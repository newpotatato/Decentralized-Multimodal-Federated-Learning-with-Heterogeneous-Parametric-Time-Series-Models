# Optimum Topologi - All Graphs

This package is rebuilt as the method-specific topology experiment.

## Topology policy used
- LVP: hybrid (tau=0.79)
- decentralized_fedavg: ring
- defta: star
- balance: line
- push_sum: ring

Topology metadata:
- topology/method_topology_map.json
- topology/lvp_reference_round1_hybrid_topology_tau079.json
- topology/lvp_reference_round1_hybrid_topology_tau079.png
- topology/lvp_reference_round1_hybrid_similarity_heatmap_tau079.png

## Raw run outputs
- raw/seeds/seed42/fig3_decentralized_methods.json
- raw/seeds/seed42/fig3_decentralized_methods_report.md

## Method comparison plots
- plots/multiseed/fig3_decentralized_methods_multiseed_network_mae.png
- plots/multiseed/fig3_decentralized_methods_multiseed_report.md
- plots/multiseed/fig3_decentralized_methods_multiseed.json

## Boxplots
- plots/boxplots/fig3_boxplot_single_multiseed.png
- plots/boxplots/fig3_boxplot_with_vs_without_fedavg_multiseed_report.md
- plots/boxplots_pair/fig3_boxplot_with_vs_without_fedavg_multiseed.png
- plots/boxplots_pair/fig3_boxplot_with_vs_without_fedavg_multiseed_report.md

## Coherence plots
- plots/coherence/multiseed/fig3_within_component_shift_multiseed.png
- plots/coherence/multiseed/fig3_within_component_shift_multiseed_report.md

## Component-wise alpha plots (two variants)
- with node count in title: plots/coherence/components/with_size_in_title/component_01.png ... component_04.png
- without node count in title: plots/coherence/components/without_size_in_title/component_01.png ... component_04.png
- metadata: plots/coherence/components/component_alpha_curves_summary.json
- report: plots/coherence/components/component_alpha_curves_report.md

Source for component-wise alpha plots:
- rebuilt from the selected final LVP topology package (`tau=0.79`) and copied into this package for final reporting
