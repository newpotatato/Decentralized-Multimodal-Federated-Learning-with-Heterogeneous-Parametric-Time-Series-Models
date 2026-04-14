# Figure 3 Decentralized Comparison

Best model: DynamicLinearModel
Scenario: contiguous, mal=0.25, attack=noise_colluded, scale=5.0

## Final MAE by method
- defta: 383741.859670
- lvp: 383868.567816
- push_sum: 383884.708681
- decentralized_fedavg: 384253.148938
- balance: 388320.466835

## Notes
- LVP is the reference decentralized method.
- Decentralized FedAvg, DeFTA, BALANCE, and Push-Sum use the same client graph as LVP.