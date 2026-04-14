#!/usr/bin/env python3
"""Run all models with current scenario parameters and save to article package."""
from __future__ import annotations

import json
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "core"))
sys.path.insert(0, str(_FL / "data_loaders"))

from data_utils import build_client_information_profiles, build_clients_from_mcc, load_mcc_series
from run_real_experiments import MODEL_REGISTRY, _build_exogenous, run_one_model


def _topic_profiles(clients: list, n_groups: int = 4) -> list:
    profiles = build_client_information_profiles(clients, "mcc")
    return [frozenset(p) | {f"sync_topic_{idx % n_groups}"} for idx, p in enumerate(profiles)]


def main() -> None:
    # Load scenario parameters from existing run
    pkg = Path(_FL / "artifacts" / "article_package_current_run_20260410")
    seed42_json = pkg / "raw" / "seeds" / "seed42" / "fig3_decentralized_methods.json"
    
    if not seed42_json.exists():
        raise FileNotFoundError(f"Missing {seed42_json}")
    
    config = json.loads(seed42_json.read_text(encoding="utf-8"))
    scenario = config.get("scenario", {})
    
    # Extract parameters
    base_path = _FL.parent
    mcc_df = load_mcc_series(base_path)
    exog = _build_exogenous(base_path, mcc_df, use_reuters=True)
    clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=20,
        column_partition=scenario.get("column_partition", "contiguous"),
    )
    profiles = _topic_profiles(clients)
    
    # Run all models with LVP aggregator (skip Markov for speed)
    models = ["DynamicLinearModel", "ARMAXModel", "KalmanFilterModel", "StructuralTimeSeriesModel"]  # Markov is too slow
    all_results = []
    
    out_path = pkg / "raw" / "all_models_lvp_seed42.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Running {model_name}...")
        print(f"{'='*60}")
        
        if model_name not in MODEL_REGISTRY:
            print(f"  Skipping {model_name} (not in registry)")
            continue
        
        ModelClass = MODEL_REGISTRY[model_name]
        
        try:
            exp = run_one_model(
                model_name,
                ModelClass,
                clients,
                profiles,
                aggregator="lvp",
                rounds=scenario.get("rounds", 10),
                local_epochs=scenario.get("local_epochs", 1),
                malicious_frac=scenario.get("malicious_frac", 0.25),
                seed=42,
                attack_strategy=scenario.get("attack_strategy", "noise_colluded"),
                attack_scale=scenario.get("attack_scale", 5.0),
                similarity_tau=config.get("similarity_tau", 0.35),
                similarity_mode=config.get("similarity_mode", "jaccard"),
                lambda_jaccard=config.get("lambda_jaccard", 0.5),
                tau_cos_min=config.get("tau_cos_min", -1.0),
                lvp_alpha=config.get("lvp_alpha", 0.6),
                lvp_self_weight=config.get("lvp_self_weight", 0.0),
                strict_errors=True,
                network_eval_mode=scenario.get("network_eval_mode", "proxy"),
                num_workers=1,
            )
            all_results.append(exp)
            print(f"  ✓ {model_name} completed")
            
            # Save incrementally
            summary = {
                "models_run": models,
                "scenario": scenario,
                "results": all_results,
            }
            out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"  → Saved intermediate results to {out_path}")
        except Exception as e:
            print(f"  ✗ {model_name} failed: {e}")
            continue
    
    print(f"\n✓ Final results saved to {out_path}")


if __name__ == "__main__":
    main()
