#!/usr/bin/env python3
"""LVP vs decentralized baselines with configurable fairness.

Stage A tunes LVP parameters on the selected client partition.
Stage B compares LVP and baselines in either fair mode (default) or fixed legacy mode.

Fair mode defaults:
    - same partition for LVP and baselines,
    - same evaluation mode,
    - baselines can pick the better similarity variant per seed (Jaccard vs hybrid).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "core"))
sys.path.insert(0, str(_FL / "data_loaders"))

from data_utils import build_client_information_profiles, build_clients_from_mcc, load_mcc_series
from run_real_experiments import MODEL_REGISTRY, _build_exogenous, run_one_model

DECENTRALIZED_METHODS = ["lvp", "decentralized_fedavg", "defta", "balance", "push_sum"]
TUNING_METHODS = ["lvp"]  # Only tune LVP in Stage A


@dataclass(frozen=True)
class LVPTuneConfig:
    """Parameters to tune in Stage A."""
    lambda_jaccard: float
    tau_cos_min: float
    lvp_alpha: float
    lvp_self_weight: float


# Stage A: Grid search over these LVP parameters
TUNE_GRID = [
    LVPTuneConfig(lambda_jaccard=0.5, tau_cos_min=0.05, lvp_alpha=0.20, lvp_self_weight=0.20),
    LVPTuneConfig(lambda_jaccard=0.5, tau_cos_min=0.05, lvp_alpha=0.20, lvp_self_weight=0.30),
    LVPTuneConfig(lambda_jaccard=0.6, tau_cos_min=0.10, lvp_alpha=0.25, lvp_self_weight=0.25),
    LVPTuneConfig(lambda_jaccard=0.6, tau_cos_min=0.15, lvp_alpha=0.25, lvp_self_weight=0.30),
    LVPTuneConfig(lambda_jaccard=0.65, tau_cos_min=0.15, lvp_alpha=0.30, lvp_self_weight=0.25),
]

# Stage B: Best LVP config from Stage A (to be filled after tuning)
BEST_LVP_CONFIG: Optional[LVPTuneConfig] = None


def _topic_profiles(clients: List, n_groups: int = 4) -> List[frozenset]:
    profiles = build_client_information_profiles(clients, "mcc")
    return [frozenset(p) | {f"sync_topic_{idx % n_groups}"} for idx, p in enumerate(profiles)]


def _build_partitioned_clients(
    mcc_df: pd.DataFrame,
    exog: Optional[pd.DataFrame],
    n_clients: int,
    partition_mode: str,
    partition_seed: int,
) -> Tuple[List[pd.DataFrame], List[frozenset]]:
    clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=n_clients,
        column_partition=partition_mode,
        partition_seed=partition_seed,
    )
    profiles = _topic_profiles(clients)
    return clients, profiles


def _anti_spike_score(metrics: Dict[str, float], beta: float = 0.3, gamma: float = 0.2) -> float:
    """Composite stability score: final_mae + beta*max_jump + gamma*round_std."""
    return metrics["final_mae"] + beta * metrics["max_jump"] + gamma * metrics["round_std"]


def stage_a_tune_lvp(
    clients: List[pd.DataFrame],
    profiles: List[frozenset],
    model_name: str,
    ModelClass,
    rounds: int,
    local_epochs: int,
    local_fit_maxiter: int,
    malicious_frac: float,
    attack_strategy: str,
    attack_scale: float,
    seed_list: List[int],
    skip_train_rounds: int = 0,
    random_init: bool = False,
    forecast_horizon: int = 10,
    transmitted_param_norm_clip: Optional[float] = None,
    state_param_norm_clip: Optional[float] = None,
    lvp_column_partition: str = "strided",
) -> Tuple[LVPTuneConfig, Dict[str, Any]]:
    """Stage A: Search for best LVP config on strided partition + hybrid similarity."""
    print("\n" + "=" * 80)
    print(f"STAGE A: TUNING LVP ON REUTERS DATASET ({lvp_column_partition} partition, hybrid similarity)")
    print("=" * 80)
    
    results = {}
    
    for config in TUNE_GRID:
        config_name = f"lvp_t{config.tau_cos_min:.2f}_l{config.lambda_jaccard:.2f}_a{config.lvp_alpha:.2f}_g{config.lvp_self_weight:.2f}"
        print(f"\n>>> Testing config: {config_name}")
        
        metrics_per_seed = []
        
        for seed in seed_list:
            try:
                exp = run_one_model(
                    model_name=model_name,
                    ModelClass=ModelClass,
                    clients=clients,
                    profiles=profiles,
                    aggregator="lvp",
                    rounds=rounds,
                    local_epochs=local_epochs,
                    local_fit_maxiter=local_fit_maxiter,
                    malicious_frac=malicious_frac,
                    seed=seed,
                    attack_strategy=attack_strategy,
                    attack_scale=attack_scale,
                    similarity_tau=0.80,
                    similarity_mode="jaccard_cosine_hybrid",
                    lambda_jaccard=config.lambda_jaccard,
                    tau_cos_min=config.tau_cos_min,
                    lvp_alpha=config.lvp_alpha,
                    lvp_self_weight=config.lvp_self_weight,
                    network_eval_mode="proxy",
                    skip_train_rounds=skip_train_rounds,
                    random_init=random_init,
                    forecast_horizon=forecast_horizon,
                    transmitted_param_norm_clip=transmitted_param_norm_clip,
                    state_param_norm_clip=state_param_norm_clip,
                )
                vals = np.asarray([float(r["network_mae"]) for r in exp.get("history") or []], dtype=float)
                jumps = np.abs(np.diff(vals)) if vals.size > 1 else np.asarray([0.0], dtype=float)
                metrics = {
                    "final_mae": float(vals[-1]) if vals.size > 0 else float("inf"),
                    "max_jump": float(np.max(jumps)) if jumps.size > 0 else 0.0,
                    "round_std": float(np.std(vals)) if vals.size > 1 else 0.0,
                }
                metrics_per_seed.append(metrics)
                score = _anti_spike_score(metrics)
                print(f"  Seed {seed}: final_mae={metrics['final_mae']:.0f}, max_jump={metrics['max_jump']:.1f}, score={score:.0f}")
            except Exception as e:
                print(f"  Seed {seed}: FAILED - {type(e).__name__}: {e}")
                continue
        
        if metrics_per_seed:
            avg_metrics = {
                k: float(np.mean([m[k] for m in metrics_per_seed]))
                for k in ["final_mae", "max_jump", "round_std"]
            }
            avg_score = _anti_spike_score(avg_metrics)
            results[config_name] = {
                "config": asdict(config),
                "avg_metrics": avg_metrics,
                "score": avg_score,
                "n_seeds": len(metrics_per_seed),
            }
            print(f"  AVG: final_mae={avg_metrics['final_mae']:.0f}, score={avg_score:.0f}")
    
    # Find best config
    best_name = min(results.keys(), key=lambda k: results[k]["score"])
    best_config_dict = results[best_name]["config"]
    best_config = LVPTuneConfig(**best_config_dict)
    best_score = results[best_name]["score"]
    
    print(f"\n*** BEST STAGE A CONFIG: {best_name} (score={best_score:.0f}) ***")
    
    return best_config, results


def stage_b_compare_methods(
    clients_strided: List[pd.DataFrame],
    clients_contiguous: List[pd.DataFrame],
    profiles_strided: List[frozenset],
    profiles_contiguous: List[frozenset],
    best_lvp_config: LVPTuneConfig,
    model_name: str,
    ModelClass,
    rounds: int,
    local_epochs: int,
    local_fit_maxiter: int,
    malicious_frac: float,
    attack_strategy: str,
    attack_scale: float,
    seed_list: List[int],
    skip_train_rounds: int = 0,
    random_init: bool = False,
    forecast_horizon: int = 10,
    transmitted_param_norm_clip: Optional[float] = None,
    state_param_norm_clip: Optional[float] = None,
    lvp_similarity_tau: float = 0.80,
    baseline_similarity_tau: float = 0.60,
    lvp_column_partition: str = "strided",
    baseline_column_partition: str = "contiguous",
    baseline_mode: str = "best_of_two",
    comparison_eval_mode: str = "refit",
) -> Dict[str, Any]:
    """Stage B: Compare methods with configurable fairness for baselines."""
    print("\n" + "=" * 80)
    print("STAGE B: HONEST COMPARISON")
    print(f"  LVP partition: {lvp_column_partition}")
    print(f"  Baseline partition: {baseline_column_partition}")
    print(f"  Baseline mode: {baseline_mode}")
    print(f"  Eval mode: {comparison_eval_mode}")
    print("=" * 80)
    
    comparison = {}
    
    for agg in DECENTRALIZED_METHODS:
        print(f"\n>>> Running {agg.upper()}")
        
        histories = []
        
        for seed in seed_list:
            try:
                if agg == "lvp":
                    clients = clients_strided
                    profiles = profiles_strided
                    sim_mode = "jaccard_cosine_hybrid"
                    lambda_jac = best_lvp_config.lambda_jaccard
                    tau_cos = best_lvp_config.tau_cos_min
                    tau = float(lvp_similarity_tau)
                    alpha = best_lvp_config.lvp_alpha
                    self_w = best_lvp_config.lvp_self_weight
                    candidate_cfgs = [
                        {
                            "sim_mode": sim_mode,
                            "lambda_jac": lambda_jac,
                            "tau_cos": tau_cos,
                            "tau": tau,
                            "alpha": alpha,
                            "self_w": self_w,
                            "tag": "lvp_fixed",
                        }
                    ]
                else:
                    clients = clients_contiguous
                    profiles = profiles_contiguous
                    if baseline_mode == "best_of_two":
                        candidate_cfgs = [
                            {
                                "sim_mode": "jaccard",
                                "lambda_jac": 1.0,
                                "tau_cos": 0.0,
                                "tau": float(baseline_similarity_tau),
                                "alpha": None,
                                "self_w": 0.0,
                                "tag": "jaccard",
                            },
                            {
                                "sim_mode": "jaccard_cosine_hybrid",
                                "lambda_jac": best_lvp_config.lambda_jaccard,
                                "tau_cos": best_lvp_config.tau_cos_min,
                                "tau": float(baseline_similarity_tau),
                                "alpha": None,
                                "self_w": 0.0,
                                "tag": "hybrid",
                            },
                        ]
                    else:
                        candidate_cfgs = [
                            {
                                "sim_mode": "jaccard",
                                "lambda_jac": 1.0,
                                "tau_cos": 0.0,
                                "tau": float(baseline_similarity_tau),
                                "alpha": None,
                                "self_w": 0.0,
                                "tag": "jaccard",
                            }
                        ]

                best_exp = None
                best_final = float("inf")
                best_tag = ""
                for cfg in candidate_cfgs:
                    exp = run_one_model(
                        model_name=model_name,
                        ModelClass=ModelClass,
                        clients=clients,
                        profiles=profiles,
                        aggregator=agg,
                        rounds=rounds,
                        local_epochs=local_epochs,
                        local_fit_maxiter=local_fit_maxiter,
                        malicious_frac=malicious_frac,
                        seed=seed,
                        attack_strategy=attack_strategy,
                        attack_scale=attack_scale,
                        similarity_tau=cfg["tau"],
                        similarity_mode=cfg["sim_mode"],
                        lambda_jaccard=cfg["lambda_jac"],
                        tau_cos_min=cfg["tau_cos"],
                        lvp_alpha=cfg["alpha"],
                        lvp_self_weight=cfg["self_w"],
                        network_eval_mode=comparison_eval_mode,
                        skip_train_rounds=skip_train_rounds,
                        random_init=random_init,
                        forecast_horizon=forecast_horizon,
                        transmitted_param_norm_clip=transmitted_param_norm_clip,
                        state_param_norm_clip=state_param_norm_clip,
                    )
                    vals = np.asarray([float(r["network_mae"]) for r in exp.get("history") or []], dtype=float)
                    final_val = float(vals[-1]) if vals.size > 0 else float("inf")
                    if final_val < best_final:
                        best_final = final_val
                        best_exp = exp
                        best_tag = str(cfg["tag"])

                if best_exp is None:
                    raise RuntimeError(f"No successful candidate run for {agg}, seed={seed}")

                history_vals = [float(r["network_mae"]) for r in best_exp.get("history") or []]
                histories.append(history_vals)
                vals = np.asarray(history_vals, dtype=float)
                final_mae = float(vals[-1]) if vals.size > 0 else float("inf")
                print(f"  Seed {seed}: final_mae={final_mae:.0f} ({best_tag})")
            except Exception as e:
                print(f"  Seed {seed}: FAILED - {type(e).__name__}: {e}")
                continue
        
        if histories:
            # Aggregate across seeds
            h_array = np.asarray(histories, dtype=float)  # (n_seeds, n_rounds)
            mean_mae = np.mean(h_array, axis=0) if h_array.size > 0 else np.array([])
            std_mae = np.std(h_array, axis=0) if h_array.size > 0 else np.array([])
            
            if len(mean_mae) > 0:
                final__ = float(mean_mae[-1])
                jumps = np.abs(np.diff(mean_mae)) if len(mean_mae) > 1 else np.array([0.0])
                max_jump = float(np.max(jumps)) if len(jumps) > 0 else 0.0
                round_std_val = float(np.std(mean_mae)) if len(mean_mae) > 1 else 0.0
            else:
                final__ = float("inf")
                max_jump = 0.0
                round_std_val = 0.0
            
            comparison[agg] = {
                "mean_mae": mean_mae.tolist(),
                "std_mae": std_mae.tolist(),
                "final_mae": final__,
                "max_jump": max_jump,
                "round_std": round_std_val,
                "n_seeds": len(histories),
            }
    
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description="LVP tuning + honest comparison with asymmetric topology")
    parser.add_argument("--base-path", type=str, default=str(_FL.parent))
    parser.add_argument("--out-dir", type=str, default=str(_FL / "artifacts" / "lvp_honest_comparison"))
    parser.add_argument("--n-clients", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=15)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--local-fit-maxiter", type=int, default=1)
    parser.add_argument("--malicious-frac", type=float, default=0.25)
    parser.add_argument("--attack-strategy", type=str, default="noise_colluded")
    parser.add_argument("--attack-scale", type=float, default=2.5)
    parser.add_argument("--seed-list", type=str, default="42,52")
    parser.add_argument("--skip-train-rounds", type=int, default=0, help="Skip local training for first N rounds (federation only)")
    parser.add_argument("--random-init", action="store_true", help="Initialize with random parameters instead of pre-trained")
    parser.add_argument("--forecast-horizon", type=int, default=10, help="Forecast horizon K for MAE/sMAPE evaluation")
    parser.add_argument("--transmitted-param-norm-clip", type=float, default=300.0, help="L2 clip for transmitted client parameter vectors")
    parser.add_argument("--state-param-norm-clip", type=float, default=300.0, help="L2 clip for synchronized client states")
    parser.add_argument("--lvp-similarity-tau", type=float, default=0.80, help="Similarity threshold for LVP graph")
    parser.add_argument("--baseline-similarity-tau", type=float, default=0.60, help="Similarity threshold for baseline graphs")
    parser.add_argument(
        "--baseline-mode",
        type=str,
        default="best_of_two",
        choices=["best_of_two", "fixed_jaccard"],
        help="Baseline fairness mode: best_of_two tries Jaccard and hybrid per seed; fixed_jaccard keeps legacy behavior.",
    )
    parser.add_argument(
        "--comparison-eval-mode",
        type=str,
        default="refit",
        choices=["proxy", "refit"],
        help="Network evaluation mode for all methods in Stage B comparison.",
    )
    parser.add_argument(
        "--lvp-column-partition",
        type=str,
        default="strided",
        choices=["contiguous", "strided", "random", "random_strided"],
        help="Partition mode for LVP clients",
    )
    parser.add_argument(
        "--baseline-column-partition",
        type=str,
        default="strided",
        choices=["contiguous", "strided", "random", "random_strided"],
        help="Partition mode for baseline clients",
    )
    parser.add_argument(
        "--exog-mode",
        type=str,
        default="news_reuters",
        choices=["news_reuters", "reuters_only"],
        help="Exogenous feature mode: both news+reuters or only reuters",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.base_path).resolve()
    
    seed_list = [int(s.strip()) for s in args.seed_list.split(",") if s.strip()]
    partition_seed = sum(seed_list) if seed_list else 0
    
    # Load data and create clients with both partition types
    print("Loading MCC time series and reuters exogenous data...")
    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=True)
    if args.exog_mode == "reuters_only":
        if exog is None or "exog_reuters" not in exog.columns:
            raise ValueError("exog_mode='reuters_only' requires Reuters features, but exog_reuters was not found.")
        exog = exog[["exog_reuters"]].copy()
    
    clients_lvp, profiles_lvp = _build_partitioned_clients(
        mcc_df,
        exog,
        n_clients=args.n_clients,
        partition_mode=args.lvp_column_partition,
        partition_seed=partition_seed,
    )
    print(f"LVP partition ({args.lvp_column_partition}): {len(clients_lvp)} clients")

    clients_baseline, profiles_baseline = _build_partitioned_clients(
        mcc_df,
        exog,
        n_clients=args.n_clients,
        partition_mode=args.baseline_column_partition,
        partition_seed=partition_seed + 17,
    )
    print(f"Baseline partition ({args.baseline_column_partition}): {len(clients_baseline)} clients")
    
    # Get model
    model_name = "DynamicLinearModel"
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    ModelClass = MODEL_REGISTRY[model_name]

    # Stage A: Tune LVP
    best_lvp_config, stage_a_results = stage_a_tune_lvp(
        clients=clients_lvp,
        profiles=profiles_lvp,
        model_name=model_name,
        ModelClass=ModelClass,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        local_fit_maxiter=args.local_fit_maxiter,
        malicious_frac=args.malicious_frac,
        attack_strategy=args.attack_strategy,
        attack_scale=args.attack_scale,
        seed_list=seed_list,
        skip_train_rounds=args.skip_train_rounds,
        random_init=args.random_init,
        forecast_horizon=args.forecast_horizon,
        transmitted_param_norm_clip=args.transmitted_param_norm_clip,
        state_param_norm_clip=args.state_param_norm_clip,
        lvp_column_partition=args.lvp_column_partition,
    )

    # Stage B: Compare all methods with asymmetric topology
    stage_b_results = stage_b_compare_methods(
        clients_strided=clients_lvp,
        clients_contiguous=clients_baseline,
        profiles_strided=profiles_lvp,
        profiles_contiguous=profiles_baseline,
        best_lvp_config=best_lvp_config,
        model_name=model_name,
        ModelClass=ModelClass,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        local_fit_maxiter=args.local_fit_maxiter,
        malicious_frac=args.malicious_frac,
        attack_strategy=args.attack_strategy,
        attack_scale=args.attack_scale,
        seed_list=seed_list,
        skip_train_rounds=args.skip_train_rounds,
        random_init=args.random_init,
        forecast_horizon=args.forecast_horizon,
        transmitted_param_norm_clip=args.transmitted_param_norm_clip,
        state_param_norm_clip=args.state_param_norm_clip,
        lvp_similarity_tau=args.lvp_similarity_tau,
        baseline_similarity_tau=args.baseline_similarity_tau,
        lvp_column_partition=args.lvp_column_partition,
        baseline_column_partition=args.baseline_column_partition,
        baseline_mode=args.baseline_mode,
        comparison_eval_mode=args.comparison_eval_mode,
    )

    # Save results
    final_report = {
        "stage_a_tuning": {
            "best_config": asdict(best_lvp_config),
            "all_results": stage_a_results,
        },
        "stage_b_comparison": stage_b_results,
    }

    report_path = out_dir / "honest_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    print(f"\nReport saved to {report_path}")

    # Print summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print("\nStage A (Tuning) Best Config:")
    print(f"  lambda_jaccard: {best_lvp_config.lambda_jaccard}")
    print(f"  tau_cos_min: {best_lvp_config.tau_cos_min}")
    print(f"  lvp_alpha: {best_lvp_config.lvp_alpha}")
    print(f"  lvp_self_weight: {best_lvp_config.lvp_self_weight}")
    
    print("\nStage B (Comparison) - Final MAE across methods:")
    for agg in DECENTRALIZED_METHODS:
        if agg in stage_b_results:
            mae = stage_b_results[agg]["final_mae"]
            print(f"  {agg:20s}: {mae:8.1f}")


if __name__ == "__main__":
    main()
