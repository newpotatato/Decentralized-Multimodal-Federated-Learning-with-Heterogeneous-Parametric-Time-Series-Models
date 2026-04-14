# Figure 3 Decentralized Comparison

Best model: DynamicLinearModel
Scenario: contiguous, mal=0.25, attack=noise_colluded, scale=5.0

## Final MAE by method
- lvp: 383433.917491
- defta: 383879.370246
- balance: 383880.101989
- push_sum: 383883.023778
- decentralized_fedavg: 383900.853620

## Notes
- LVP is the reference decentralized method.
- Decentralized FedAvg, DeFTA, BALANCE, and Push-Sum use the same client graph as LVP.