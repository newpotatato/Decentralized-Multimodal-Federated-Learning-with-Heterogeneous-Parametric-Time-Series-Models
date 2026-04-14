#!/usr/bin/env python3
"""
Targeted LVP sweep on three nonstandard scenarios.

Workflow:
1) Coarse stage: run only LVP over a compact hyperparameter grid.
2) Fine stage: for top-K LVP configs per scenario, run full method comparison.

This keeps compute tractable while directly checking whether LVP can become top-1.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from federated_learning.data_loaders.data_utils import (  # noqa: E402
    build_client_information_profiles,
    build_clients_from_mcc,
    load_mcc_series,
)
from run_real_experiments import (  # noqa: E402
    MODEL_REGISTRY,
    REPO_ROOT_DEFAULT,
    _build_exogenous,
    run_one_model,
)


def _final_mae(exp: Dict) -> float:
    hist = exp.get("history", [])
    if not hist:
        return float("inf")
    return float(hist[-1].get("network_mae", 1e12))


def _stable_float(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    return float(x)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Targeted nonstandard LVP sweep")
    p.add_argument("--base-path", type=str, default=str(REPO_ROOT_DEFAULT))
    p.add_argument(
        "--output",
        type=str,
        default="federated_learning/experiments/artifacts/nonstandard_lvp_targeted_sweep.json",
    )
    p.add_argument("--model", type=str, default="DynamicLinearModel", choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--forecast-horizon", type=int, default=10)
    p.add_argument("--partition", type=str, default="strided", choices=["contiguous", "strided", "random", "random_strided"])
    p.add_argument("--partition-seed", type=int, default=123)
    p.add_argument("--network-eval-mode", type=str, default="proxy", choices=["proxy", "refit"])
    p.add_argument("--topk", type=int, default=6, help="Top LVP configs per scenario for full comparison")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["lvp", "decentralized_fedavg", "defta", "balance", "push_sum"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_path).resolve()

    mcc_df = load_mcc_series(base_path)
    exog = _build_exogenous(base_path, mcc_df, use_reuters=True)
    clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=int(args.n_clients),
        column_partition=args.partition,
        partition_seed=int(args.partition_seed),
    )
    profiles = build_client_information_profiles(clients, data_source="mcc")

    if not clients:
        raise RuntimeError("No clients were built from MCC data")

    ModelClass = MODEL_REGISTRY[args.model]

    common = {
        "model_name": args.model,
        "ModelClass": ModelClass,
        "clients": clients,
        "profiles": profiles,
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "seed": int(args.seed),
        "similarity_mode": "jaccard_cosine_hybrid",
        "tau_cos_min": -0.25,
        "local_fit_maxiter": 10,
        "eval_fit_maxiter": 0 if args.network_eval_mode == "proxy" else 10,
        "forecast_horizon": int(args.forecast_horizon),
        "transmitted_param_norm_clip": 40.0,
        "state_param_norm_clip": 40.0,
        "strict_errors": False,
        "network_eval_mode": args.network_eval_mode,
        "skip_train_rounds": 0,
        "random_init": False,
    }

    scenarios: List[Dict] = [
        {
            "name": "regime_clients_cascading_drift",
            "base": {
                "malicious_frac": 0.20,
                "attack_strategy": "noise_colluded",
                "attack_scale": 2.5,
                "malicious_selection": "random",
                "regime_drift_scale": 0.40,
                "async_stale_frac": 0.0,
                "block_missing_frac": 0.0,
            },
            "async_grid": [0.0, 0.1],
        },
        {
            "name": "hub_targeted_attack_heavy_tails",
            "base": {
                "malicious_frac": 0.30,
                "attack_strategy": "noise_colluded_heavy_tail",
                "attack_scale": 4.0,
                "malicious_selection": "hub_targeted",
                "regime_drift_scale": 0.0,
                "async_stale_frac": 0.0,
                "block_missing_frac": 0.0,
            },
            "async_grid": [0.0, 0.1],
        },
        {
            "name": "asynchrony_block_missing",
            "base": {
                "malicious_frac": 0.15,
                "attack_strategy": "noise",
                "attack_scale": 2.0,
                "malicious_selection": "random",
                "regime_drift_scale": 0.0,
                "async_stale_frac": 0.35,
                "block_missing_frac": 0.20,
            },
            "async_grid": [0.25, 0.35, 0.45],
        },
    ]

    tau_grid = [0.08, 0.12, 0.16]
    lambda_grid = [0.5, 0.6, 0.7]
    self_weight_grid = [0.0, 0.1, 0.2]
    alpha_grid: List[Optional[float]] = [None, 0.2, 0.3]

    summary: Dict[str, Dict] = {
        "config": {
            "model": args.model,
            "methods": args.methods,
            "seed": int(args.seed),
            "n_clients": int(args.n_clients),
            "rounds": int(args.rounds),
            "local_epochs": int(args.local_epochs),
            "partition": args.partition,
            "network_eval_mode": args.network_eval_mode,
            "topk": int(args.topk),
        },
        "scenarios": {},
    }

    any_lvp_top1 = False

    for scenario in scenarios:
        s_name = scenario["name"]
        s_base = dict(scenario["base"])
        async_grid = scenario["async_grid"]

        print(f"\n=== Coarse LVP stage: {s_name} ===")
        lvp_candidates: List[Dict] = []

        for tau, lam, sw, alpha, async_frac in product(
            tau_grid,
            lambda_grid,
            self_weight_grid,
            alpha_grid,
            async_grid,
        ):
            kwargs = dict(s_base)
            kwargs["async_stale_frac"] = float(async_frac)
            kwargs["similarity_tau"] = float(tau)
            kwargs["lambda_jaccard"] = float(lam)
            kwargs["lvp_self_weight"] = float(sw)
            kwargs["lvp_alpha"] = _stable_float(alpha)

            exp_lvp = run_one_model(
                aggregator="lvp",
                **common,
                **kwargs,
            )
            mae = _final_mae(exp_lvp)
            row = {
                "final_network_mae": mae,
                "similarity_tau": float(tau),
                "lambda_jaccard": float(lam),
                "lvp_self_weight": float(sw),
                "lvp_alpha": _stable_float(alpha),
                "async_stale_frac": float(async_frac),
                "scenario_kwargs": dict(s_base),
            }
            lvp_candidates.append(row)

        lvp_candidates_sorted = sorted(lvp_candidates, key=lambda x: x["final_network_mae"])
        top_candidates = lvp_candidates_sorted[: max(1, int(args.topk))]

        print(f"Top-{len(top_candidates)} LVP configs selected for full comparison")

        full_results: List[Dict] = []
        for i, cfg in enumerate(top_candidates, start=1):
            print(
                "  candidate #{}: mae={:.6f}, tau={}, lambda={}, self_w={}, alpha={}, async={}"
                .format(
                    i,
                    cfg["final_network_mae"],
                    cfg["similarity_tau"],
                    cfg["lambda_jaccard"],
                    cfg["lvp_self_weight"],
                    cfg["lvp_alpha"],
                    cfg["async_stale_frac"],
                )
            )

            per_method: List[Dict] = []
            for method in args.methods:
                kwargs = dict(s_base)
                kwargs["async_stale_frac"] = cfg["async_stale_frac"]
                kwargs["similarity_tau"] = cfg["similarity_tau"]
                kwargs["lambda_jaccard"] = cfg["lambda_jaccard"]
                kwargs["lvp_self_weight"] = cfg["lvp_self_weight"]
                kwargs["lvp_alpha"] = cfg["lvp_alpha"]

                exp = run_one_model(
                    aggregator=method,
                    **common,
                    **kwargs,
                )
                per_method.append(
                    {
                        "method": method,
                        "final_network_mae": _final_mae(exp),
                    }
                )

            ranking = sorted(per_method, key=lambda x: x["final_network_mae"])
            winner = ranking[0]["method"] if ranking else None
            if winner == "lvp":
                any_lvp_top1 = True
            full_results.append(
                {
                    "candidate": cfg,
                    "winner": winner,
                    "ranking": ranking,
                }
            )

        best_overall = min(full_results, key=lambda x: x["ranking"][0]["final_network_mae"]) if full_results else None
        best_lvp_top1 = next((x for x in full_results if x["winner"] == "lvp"), None)

        summary["scenarios"][s_name] = {
            "coarse_grid_size": len(lvp_candidates),
            "top_lvp_candidates": top_candidates,
            "full_results": full_results,
            "best_overall": best_overall,
            "best_lvp_top1": best_lvp_top1,
        }

    summary["any_lvp_top1"] = any_lvp_top1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nSweep finished")
    print(f"Saved summary: {out}")
    print(f"Any LVP top-1 found: {any_lvp_top1}")


if __name__ == "__main__":
    main()
