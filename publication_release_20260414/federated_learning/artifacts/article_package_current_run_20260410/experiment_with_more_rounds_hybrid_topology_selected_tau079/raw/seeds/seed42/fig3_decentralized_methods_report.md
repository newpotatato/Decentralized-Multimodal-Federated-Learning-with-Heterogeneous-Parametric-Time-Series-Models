# Figure 3 Decentralized Comparison

Best model: DynamicLinearModel
Scenario: contiguous, mal=0.25, attack=noise_colluded, scale=5.0

## Final MAE by method
- defta: 383865.662497
- balance: 383865.838834
- push_sum: 383866.387232
- decentralized_fedavg: 383866.388818
- lvp: 383866.620127

## Notes
- LVP is the reference decentralized method.
- Decentralized FedAvg, DeFTA, BALANCE, and Push-Sum use the same client graph as LVP.