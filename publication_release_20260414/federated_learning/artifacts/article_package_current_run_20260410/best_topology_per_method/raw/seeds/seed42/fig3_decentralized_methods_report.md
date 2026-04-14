# Best Topology Per Method (Single Seed Run)

Model: DynamicLinearModel
Seed: 42

## Best topology map
- lvp: mode=jaccard_cosine_hybrid, tau=0.7900, selection_mae=375247.978493
- decentralized_fedavg: mode=jaccard, tau=0.5700, selection_mae=377887.009684
- defta: mode=jaccard, tau=0.7900, selection_mae=383814.917189
- balance: mode=jaccard, tau=0.5700, selection_mae=381422.687991
- push_sum: mode=jaccard, tau=0.5700, selection_mae=382834.149885

## Final MAE by method
- defta: 383865.662497
- lvp: 383866.620127
- balance: 383880.101989
- push_sum: 383883.023778
- decentralized_fedavg: 383900.853620