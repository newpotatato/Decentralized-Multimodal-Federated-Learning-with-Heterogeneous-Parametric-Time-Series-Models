# Figure 3 Decentralized Comparison

Best model: DynamicLinearModel
Scenario: contiguous, mal=0.25, attack=noise_colluded, scale=5.0

## Final MAE by method
- lvp: 383864.588512
- defta: 383879.377381
- balance: 383880.037749
- push_sum: 383883.023778
- decentralized_fedavg: 383886.652238

## Notes
- LVP is the reference decentralized method.
- Decentralized FedAvg, DeFTA, BALANCE, and Push-Sum use the same client graph as LVP.