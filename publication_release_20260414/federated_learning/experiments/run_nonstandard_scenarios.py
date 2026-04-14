#!/usr/bin/env python3
"""
Run three nonstandard federated scenarios requested by user:
1) Regime clients + cascading drift
2) Hub-targeted attack + heavy tails
3) Asynchrony + block-missing

The script evaluates multiple aggregation methods under identical settings
and prints final MAE rankings per scenario.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

from run_real_experiments import (
    MODEL_REGISTRY,
    REPO_ROOT_DEFAULT,
    _build_exogenous,
    run_one_model,
)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from federated_learning.data_loaders.data_utils import (
    build_client_information_profiles,
    build_clients_from_mcc,
    load_mcc_series,
)


def _final_mae(exp: Dict) -> float:
    hist = exp.get("history", [])
    if not hist:
        return float("inf")
    return float(hist[-1].get("network_mae", 1e12))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run nonstandard LVP stress scenarios")
    p.add_argument("--base-path", type=str, default=str(REPO_ROOT_DEFAULT))
    p.add_argument("--output", type=str, default="federated_learning/experiments/artifacts/nonstandard_scenarios_summary.json")
    p.add_argument("--model", type=str, default="ARMAXModel", choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--methods", nargs="+", default=["lvp", "decentralized_fedavg", "defta", "balance", "push_sum"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--forecast-horizon", type=int, default=10)
    p.add_argument("--partition", type=str, default="strided", choices=["contiguous", "strided", "random", "random_strided"])
    p.add_argument("--partition-seed", type=int, default=123)
    p.add_argument("--network-eval-mode", type=str, default="proxy", choices=["proxy", "refit"])
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
        "similarity_tau": 0.12,
        "similarity_mode": "jaccard_cosine_hybrid",
        "lambda_jaccard": 0.6,
        "tau_cos_min": -0.25,
        "lvp_self_weight": 0.05,
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
            "kwargs": {
                "malicious_frac": 0.20,
                "attack_strategy": "noise_colluded",
                "attack_scale": 2.5,
                "malicious_selection": "random",
                "regime_drift_scale": 0.40,
                "async_stale_frac": 0.0,
                "block_missing_frac": 0.0,
            },
        },
        {
            "name": "hub_targeted_attack_heavy_tails",
            "kwargs": {
                "malicious_frac": 0.30,
                "attack_strategy": "noise_colluded_heavy_tail",
                "attack_scale": 4.0,
                "malicious_selection": "hub_targeted",
                "regime_drift_scale": 0.0,
                "async_stale_frac": 0.0,
                "block_missing_frac": 0.0,
            },
        },
        {
            "name": "asynchrony_block_missing",
            "kwargs": {
                "malicious_frac": 0.15,
                "attack_strategy": "noise",
                "attack_scale": 2.0,
                "malicious_selection": "random",
                "regime_drift_scale": 0.0,
                "async_stale_frac": 0.35,
                "block_missing_frac": 0.20,
            },
        },
    ]

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
        },
        "scenarios": {},
    }

    for scenario in scenarios:
        name = scenario["name"]
        kwargs = scenario["kwargs"]
        rows: List[Dict] = []
        print(f"\\n=== Scenario: {name} ===")
        for method in args.methods:
            exp = run_one_model(
                aggregator=method,
                **common,
                **kwargs,
            )
            value = _final_mae(exp)
            rows.append(
                {
                    "method": method,
                    "final_network_mae": value,
                    "history": exp.get("history", []),
                    "scenario_kwargs": kwargs,
                }
            )
            print(f"{method:24s} final_mae={value:.6f}")

        rows_sorted = sorted(rows, key=lambda x: x["final_network_mae"])
        winner = rows_sorted[0]["method"] if rows_sorted else None
        print(f"winner: {winner}")

        summary["scenarios"][name] = {
            "winner": winner,
            "ranking": rows_sorted,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\\nSaved summary: {out}")


if __name__ == "__main__":
    main()
