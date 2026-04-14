#!/usr/bin/env python3
"""
Deep diagnostic of LVP performance issues.
Goals:
1. Compare all aggregators on HONEST agents (0% malicious)
2. Check if tau sweep has ANY impact on neighbor graph structure
3. Diagnose signal propagation vs data mismatch
"""

import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "data_loaders"))
sys.path.insert(0, str(_FL / "core"))

from data_utils import (
    build_client_information_profiles,
    build_clients_from_mcc,
    load_mcc_series,
    train_test_split_series,
)
from run_real_experiments import MODEL_REGISTRY, _build_exogenous, evaluate_model, run_one_model
from decentralized_lvp import build_neighbor_graph

import numpy as np

print("=" * 100)
print("DIAGNOSTIC 1: NEIGHBOR GRAPH STRUCTURE BY tau")
print("=" * 100)

base_path = Path(__file__).resolve().parent
base_path = Path(__file__).resolve().parent.parent.parent  # Go to c:\tarasova\RNF
mcc_df = load_mcc_series(base_path)
exog = _build_exogenous(base_path, mcc_df, use_reuters=True)

# Build ONE scenario
clients = build_clients_from_mcc(mcc_df, exog, n_clients=20, column_partition="contiguous")
profiles = build_client_information_profiles(clients, "mcc")

tau_values = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.72]
print(f"\n{len(clients)} clients, {len(profiles)} profiles")
print(f"Profile sizes (MCC categories per client):\n  {[len(p) for p in profiles]}\n")

for tau in tau_values:
    neighbors, sim = build_neighbor_graph(profiles, tau=tau)
    degrees = [len(n) for n in neighbors]
    
    # Connectivity: % of connected clients
    connected_pairs = sum(degrees) // 2  # Each edge counted twice
    total_possible = (len(clients) * (len(clients) - 1)) // 2
    connectivity = 100 * connected_pairs / total_possible if total_possible else 0
    
    print(f"tau={tau:5.2f} | Degrees: {np.mean(degrees):5.1f}+/-{np.std(degrees):4.1f} "
          f"| Connected: {connectivity:6.1f}% ({connected_pairs:2d}/{total_possible:2d} edges) "
          f"| Isolated: {sum(1 for d in degrees if d == 0)}")

# Print actual similarity matrix for tau=0.35 (default) and tau=0.05 (permissive)
print("\n" + "-" * 100)
print("Jaccard similarity matrix (subset: clients 0-4 vs 0-4):")
neighbors_default, sim_default = build_neighbor_graph(profiles, tau=0.35)
print("\nAt tau=0.35 (default):")
print(sim_default[:5, :5].round(3))
print(f"Thresholded neighbors for client 0: {neighbors_default[0]}")

print("\nAt tau=0.05 (permissive):")
neighbors_perm, sim_perm = build_neighbor_graph(profiles, tau=0.05)
print(sim_perm[:5, :5].round(3))
print(f"Thresholded neighbors for client 0: {neighbors_perm[0]}")

# Statistics
print(f"\nSimilarity statistics (Jaccard across all pairs, diagonal excluded):")
tri = np.triu_indices(len(profiles), k=1)
jacc_all = sim_default[tri]
print(f"  Min: {jacc_all.min():.4f}, Max: {jacc_all.max():.4f}, Mean: {jacc_all.mean():.4f}, Median: {np.median(jacc_all):.4f}")
print(f"  % >= 0.35: {100 * (jacc_all >= 0.35).mean():.1f}%")
print(f"  % >= 0.15: {100 * (jacc_all >= 0.15).mean():.1f}%")

print("\n" + "=" * 100) 
print("DIAGNOSTIC 2: HONEST-ONLY BASELINE (0% MALICIOUS)")
print("=" * 100)
print("\nRunning single seed (seed=42) with 0% malicious to check aggregator diff on clean data...")
print("This tells us: is LVP inherently broken, or just poor at Byzantine resilience?\n")

ModelClass = MODEL_REGISTRY["DynamicLinearModel"]
results_honest = {}

for agg_mode in ["fedavg", "lvp", "krum", "weighted_median", "scaffold"]:
    print(f"Testing {agg_mode:15} ... ", end="", flush=True)
    try:
        exp = run_one_model(
            "DynamicLinearModel",
            ModelClass,
            clients,
            profiles,
            aggregator=agg_mode,
            rounds=10,
            local_epochs=1,
            malicious_frac=0.0,  # KEY: 0% = all honest
            seed=42,
            attack_strategy="label_flip",
            attack_scale=3.0,
            similarity_tau=0.35,
            similarity_mode="jaccard",
            lambda_jaccard=0.5,
            tau_cos_min=0.0,
            lvp_alpha=0.53,
            krum_f=-1,
            local_fit_maxiter=10,
            eval_fit_maxiter=0,
        )
        
        final_mae = float(exp["history"][-1]["network_mae"])
        auc = float(sum(r["network_mae"] for r in exp["history"]) / len(exp["history"]))
        results_honest[agg_mode] = {"final_mae": final_mae, "auc": auc}
        print(f"OK Final MAE: {final_mae:12.2f}, AUC: {auc:12.2f}")
    except Exception as e:
        print(f"ERROR: {e}")

print("\nComparison (0% malicious):")
print("  Aggregator       | Final MAE  | AUC        | vs FedAvg")
print("  ---|---|---|---|---")
if "fedavg" in results_honest:
    fedavg_mae = results_honest["fedavg"]["final_mae"]
    for agg in ["fedavg", "lvp", "krum", "weighted_median", "scaffold"]:
        if agg in results_honest:
            mae = results_honest[agg]["final_mae"]
            auc = results_honest[agg]["auc"]
            delta = ((mae - fedavg_mae) / fedavg_mae * 100) if fedavg_mae != 0 else 0
            print(f"  {agg:15} | {mae:10.2f} | {auc:10.2f} | {delta:+6.1f}%")

print("\nINTERPRETATION:")
print("  - If LVP is similar to FedAvg on 0% malicious: problem is Byzantine resilience, not basic aggregation")
print("  - If LVP is much worse on 0% malicious: problem is fundamental (initialization, learning rate, etc.)")

print("\n" + "=" * 100)
print("DIAGNOSTIC 3: SOLO TRAINING (Each client alone)")
print("=" * 100)
print("\nRunning single-epoch local training on ONE representative client to get solo MAE baseline...\n")

# Train one representative client alone to compare against FL aggregators
try:
    solo_df = clients[0]
    train_df, test_df = train_test_split_series(solo_df, test_ratio=0.2)
    solo_model = ModelClass()
    fit_kwargs = {"target_col": "amt", "use_transform": True}
    exog_cols = [c for c in train_df.columns if c.startswith("exog")]
    if exog_cols and ModelClass.__name__ == "ARMAXModel":
        fit_kwargs["exog_cols"] = exog_cols

    best_mae = float('inf')
    for _ in range(10):
        solo_model.fit(train_df, **fit_kwargs)
        mae = evaluate_model(solo_model, test_df)
        best_mae = min(best_mae, mae)

    print(f"Single client solo MAE (10 epochs): {best_mae:.2f}")
    print(f"Aggregated MAE at round 1, 0% mal:  {results_honest.get('fedavg', {}).get('final_mae', 'N/A'):.2f}")
    print("\nIf fedavg MAE >> solo MAE: data is too heterogeneous and aggregation hurts.")

except Exception as e:
    print(f"Could not compute solo baseline: {e}")

print("\n" + "=" * 100)
print("HYPOTHESIS CANDIDATES")
print("=" * 100)

print("""
H1: Data too heterogeneous (non-IID) -> LVP local consensus cannot converge
    Evidence to check: fedavg on 0% mal much worse than solo

H2: Neighbor graph too sparse (tau too high) -> isolated clients, no propagation
    Evidence to check: neighbor degrees very low, connectivity < 50%

H3: Neighbor graph too dense (tau too low) -> Byzantine nodes pollute all neighbors
    Evidence to check: even tau=0.05 should have few connections, but LVP still fails

H4: Learning rate (alpha) or step size wrong -> slow convergence
    Evidence to check: LVP MAE decreases/stable across rounds, or immediately plateaus

H5: Byzantine attack is poorly matched to model structure
    Evidence to check: label_flip attack doesn't align with parameter scaling

H6: Profile mismatch (Jaccard on MCC categories) does not capture model relevance
    Evidence to check: manual inspection of which clients connect / high Jaccard pairs
""")

print("\nNEXT STEPS:")
print("  1. Use DIAGNOSTIC 2 to separate Byzantine issue from baseline aggregation issue")
print("  2. If LVP ~= FedAvg on 0% mal, investigate attack robustness")
print("  3. If LVP >> FedAvg on 0% mal, investigate graph sparsity and learning-rate settings")
