# Figure 3 Decentralized Comparison

Best model: DynamicLinearModel
Scenario: contiguous, mal=0.25, attack=noise_colluded, scale=5.0

## Final MAE by method
- balance: 383861.898269
- lvp: 383871.766142
- push_sum: 383875.083257
- decentralized_fedavg: 383881.531002
- defta: 383971.732319

## Notes
- LVP is the reference decentralized method.
- Decentralized FedAvg, DeFTA, BALANCE, and Push-Sum use the same client graph as LVP.