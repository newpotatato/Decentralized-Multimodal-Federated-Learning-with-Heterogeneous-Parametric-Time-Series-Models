# Figure 3 Decentralized Comparison

Best model: DynamicLinearModel
Scenario: contiguous, mal=0.25, attack=noise_colluded, scale=5.0

## Final MAE by method
- defta: 383865.656432
- push_sum: 383866.387232
- decentralized_fedavg: 383866.389478
- lvp: 383873.029740
- balance: 385508.609762

## Notes
- LVP is the reference decentralized method.
- Decentralized FedAvg, DeFTA, BALANCE, and Push-Sum use the same client graph as LVP.