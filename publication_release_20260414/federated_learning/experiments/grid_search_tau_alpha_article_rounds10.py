#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "core"))
sys.path.insert(0, str(_FL / "data_loaders"))

from data_utils import build_client_information_profiles, build_clients_from_mcc, load_mcc_series
from run_real_experiments import MODEL_REGISTRY, _build_exogenous, run_one_model


def _topic_profiles(clients: List, n_groups: int = 4) -> List:
    profiles = build_client_information_profiles(clients, "mcc")
    return [frozenset(p) | {f"sync_topic_{idx % n_groups}"} for idx, p in enumerate(profiles)]


def _final_mae(exp: Dict) -> float:
    hist = exp.get("history") or []
    if not hist:
        return float("inf")
    return float(hist[-1].get("network_mae", float("inf")))


def _evaluate_combo(task: Dict) -> Dict[str, float]:
    base = Path(task["base_path"]).resolve()
    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=True)
    clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=int(task["n_clients"]),
        column_partition=str(task["column_partition"]),
    )
    profiles = _topic_profiles(clients)

    ModelClass = MODEL_REGISTRY["DynamicLinearModel"]
    exp = run_one_model(
        "DynamicLinearModel",
        ModelClass,
        clients,
        profiles,
        aggregator="lvp",
        rounds=int(task["rounds"]),
        local_epochs=int(task["local_epochs"]),
        malicious_frac=float(task["malicious_frac"]),
        seed=int(task["seed"]),
        attack_strategy=str(task["attack_strategy"]),
        attack_scale=float(task["attack_scale"]),
        similarity_tau=float(task["tau"]),
        similarity_mode=str(task["similarity_mode"]),
        lambda_jaccard=float(task["lambda_jaccard"]),
        tau_cos_min=float(task["tau_cos_min"]),
        lvp_alpha=float(task["alpha"]),
        strict_errors=True,
        network_eval_mode=str(task["network_eval_mode"]),
        num_workers=int(task["num_workers"]),
    )
    return {
        "tau": float(task["tau"]),
        "alpha": float(task["alpha"]),
        "seed": int(task["seed"]),
        "final_mae": _final_mae(exp),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Full tau-alpha grid search for one-seed article scenario")
    p.add_argument("--base-path", type=str, default=str(_FL.parent))
    p.add_argument("--out-dir", type=str, default=str(_FL / "artifacts" / "grid_search_tau_alpha_article_rounds10"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--column-partition", type=str, default="contiguous", choices=["contiguous", "strided"])
    p.add_argument("--attack-strategy", type=str, default="noise_colluded")
    p.add_argument("--attack-scale", type=float, default=5.0)
    p.add_argument("--malicious-frac", type=float, default=0.25)
    p.add_argument("--network-eval-mode", type=str, default="proxy", choices=["proxy", "refit"])
    p.add_argument("--similarity-mode", type=str, default="jaccard", choices=["jaccard", "jaccard_cosine_hybrid"])
    p.add_argument("--lambda-jaccard", type=float, default=0.5)
    p.add_argument("--tau-cos-min", type=float, default=-1.0)
    p.add_argument("--tau-grid", type=str, default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--alpha-grid", type=str, default="0.15,0.20,0.25,0.30,0.35,0.40")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--grid-workers", type=int, default=4)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.base_path).resolve()

    tau_grid = [float(x.strip()) for x in args.tau_grid.split(",") if x.strip()]
    alpha_grid = [float(x.strip()) for x in args.alpha_grid.split(",") if x.strip()]

    tasks: List[Dict] = []
    for tau in tau_grid:
        for alpha in alpha_grid:
            tasks.append(
                {
                    "base_path": str(base),
                    "seed": int(args.seed),
                    "rounds": int(args.rounds),
                    "local_epochs": int(args.local_epochs),
                    "n_clients": int(args.n_clients),
                    "column_partition": str(args.column_partition),
                    "attack_strategy": str(args.attack_strategy),
                    "attack_scale": float(args.attack_scale),
                    "malicious_frac": float(args.malicious_frac),
                    "network_eval_mode": str(args.network_eval_mode),
                    "similarity_mode": str(args.similarity_mode),
                    "lambda_jaccard": float(args.lambda_jaccard),
                    "tau_cos_min": float(args.tau_cos_min),
                    "tau": float(tau),
                    "alpha": float(alpha),
                    "num_workers": int(args.num_workers),
                }
            )

    grid_workers = max(1, int(args.grid_workers))
    if grid_workers == 1:
        rows = [_evaluate_combo(t) for t in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=grid_workers, mp_context=ctx) as ex:
            rows = list(ex.map(_evaluate_combo, tasks))

    rows.sort(key=lambda r: (r["final_mae"], r["tau"], r["alpha"]))
    best = rows[0] if rows else {"tau": None, "alpha": None, "final_mae": float("inf")}

    summary = {
        "config": {
            "seed": args.seed,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "n_clients": args.n_clients,
            "column_partition": args.column_partition,
            "attack_strategy": args.attack_strategy,
            "attack_scale": args.attack_scale,
            "malicious_frac": args.malicious_frac,
            "network_eval_mode": args.network_eval_mode,
            "similarity_mode": args.similarity_mode,
            "lambda_jaccard": args.lambda_jaccard,
            "tau_cos_min": args.tau_cos_min,
            "num_workers": args.num_workers,
            "grid_workers": args.grid_workers,
            "tau_grid": tau_grid,
            "alpha_grid": alpha_grid,
            "n_combinations": len(tasks),
        },
        "best": best,
        "results": rows,
    }

    summary_json = out_dir / "tau_alpha_grid_search_one_seed_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    csv_path = out_dir / "tau_alpha_grid_search_one_seed_results.csv"
    lines = ["tau,alpha,seed,final_mae"]
    for r in rows:
        lines.append(f"{r['tau']:.2f},{r['alpha']:.2f},{r['seed']},{r['final_mae']:.12f}")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Completed combinations: {len(tasks)}")
    print(f"Best tau={best['tau']:.2f}, alpha={best['alpha']:.2f}, final_mae={best['final_mae']:.6f}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
