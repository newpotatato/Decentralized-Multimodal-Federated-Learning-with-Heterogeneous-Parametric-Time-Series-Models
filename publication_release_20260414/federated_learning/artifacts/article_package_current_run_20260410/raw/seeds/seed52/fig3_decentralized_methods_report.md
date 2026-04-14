# Figure 3 Decentralized Comparison

Best model: DynamicLinearModel
Scenario: contiguous, mal=0.25, attack=noise_colluded, scale=5.0

## Final MAE by method
- decentralized_fedavg: 383842.286105
- push_sum: 383875.083257
- lvp: 384070.030305
- balance: 384388.157345
- defta: 386191.897398

## Notes
- LVP is the reference decentralized method.
- Decentralized FedAvg, DeFTA, BALANCE, and Push-Sum use the same client graph as LVP.