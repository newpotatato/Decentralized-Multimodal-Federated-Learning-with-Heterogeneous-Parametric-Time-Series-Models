# Figure 3 Decentralized Comparison

Best model: DynamicLinearModel
Scenario: contiguous, mal=0.25, attack=noise_colluded, scale=5.0

## Final MAE by method
- decentralized_fedavg: 383794.790460
- lvp: 383879.756459
- push_sum: 383884.708681
- balance: 395434.406046
- defta: 397903.131306

## Notes
- LVP is the reference decentralized method.
- Decentralized FedAvg, DeFTA, BALANCE, and Push-Sum use the same client graph as LVP.