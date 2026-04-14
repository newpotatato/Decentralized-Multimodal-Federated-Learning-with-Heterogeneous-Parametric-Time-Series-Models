# Figure 3 Decentralized Comparison

Best model: DynamicLinearModel
Scenario: contiguous, mal=0.25, attack=noise_colluded, scale=5.0

## Final MAE by method
- defta: 383865.590424
- balance: 383865.869174
- decentralized_fedavg: 383866.379881
- push_sum: 383866.387232
- lvp: 383875.084290

## Notes
- LVP is the reference decentralized method.
- Decentralized FedAvg, DeFTA, BALANCE, and Push-Sum use the same client graph as LVP.