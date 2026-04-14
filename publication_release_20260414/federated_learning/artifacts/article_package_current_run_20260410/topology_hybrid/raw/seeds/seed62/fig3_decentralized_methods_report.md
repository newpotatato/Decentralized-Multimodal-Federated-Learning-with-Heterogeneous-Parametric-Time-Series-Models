# Figure 3 Decentralized Comparison

Best model: DynamicLinearModel
Scenario: contiguous, mal=0.25, attack=noise_colluded, scale=5.0

## Final MAE by method
- push_sum: 383875.083257
- decentralized_fedavg: 384400.345385
- defta: 386519.004590
- lvp: 388545.710790
- balance: 394027.219399

## Notes
- LVP is the reference decentralized method.
- Decentralized FedAvg, DeFTA, BALANCE, and Push-Sum use the same client graph as LVP.