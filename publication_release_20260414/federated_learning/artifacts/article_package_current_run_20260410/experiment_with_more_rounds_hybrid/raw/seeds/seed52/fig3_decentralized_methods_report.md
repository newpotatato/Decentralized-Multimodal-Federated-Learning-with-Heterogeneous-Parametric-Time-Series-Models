# Figure 3 Decentralized Comparison

Best model: DynamicLinearModel
Scenario: contiguous, mal=0.25, attack=noise_colluded, scale=5.0

## Final MAE by method
- decentralized_fedavg: 383876.057007
- defta: 383879.378049
- push_sum: 383883.023778
- lvp: 383884.721790
- balance: 389674.229505

## Notes
- LVP is the reference decentralized method.
- Decentralized FedAvg, DeFTA, BALANCE, and Push-Sum use the same client graph as LVP.