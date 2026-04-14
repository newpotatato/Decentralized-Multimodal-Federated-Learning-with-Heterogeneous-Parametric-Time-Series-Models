# Figure 3 Decentralized Comparison

Best model: DynamicLinearModel
Scenario: contiguous, mal=0.25, attack=noise_colluded, scale=5.0

## Final MAE by method
- decentralized_fedavg: 383868.838160
- push_sum: 383884.708681
- lvp: 383903.514998
- balance: 391767.713973
- defta: 398171.772998

## Notes
- LVP is the reference decentralized method.
- Decentralized FedAvg, DeFTA, BALANCE, and Push-Sum use the same client graph as LVP.