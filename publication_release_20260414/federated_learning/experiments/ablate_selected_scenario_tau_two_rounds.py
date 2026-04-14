#!/usr/bin/env python3
"""Tau ablation for two selected scenarios (small and large rounds).

Scenarios:
- Small rounds: rounds=10
- Large rounds: rounds=30

Common setup (can be overridden by args):
- model: DynamicLinearModel
- partition: contiguous
- malicious_frac: 0.25
- attack_strategy: noise_colluded
- attack_scale: 5.0
- local_epochs: 1
- hybrid similarity by default (lambda_jaccard=0.25)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "core"))
sys.path.insert(0, str(_FL / "data_loaders"))

from data_utils import build_client_information_profiles, build_clients_from_mcc, load_mcc_series  # noqa: E402
from run_real_experiments import MODEL_REGISTRY, _build_exogenous, run_one_model  # noqa: E402


def _parse_csv_ints(text: str) -> List[int]:
    vals: List[int] = []
    for x in (text or "").split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    return vals


def _parse_csv_floats(text: str) -> List[float]:
    vals: List[float] = []
    for x in (text or "").split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return vals


def _final_mae(exp: Dict[str, Any]) -> float:
    hist = exp.get("history", [])
    if not hist:
        return float("inf")
    return float(hist[-1].get("network_mae", float("inf")))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tau ablation for selected scenario at two round budgets")
    p.add_argument("--base-path", type=str, default=str(_FL.parent))
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_FL / "artifacts" / "selected_scenario_tau_ablation_two_rounds"),
    )
    p.add_argument("--model", type=str, default="DynamicLinearModel", choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--seed-list", type=str, default="42,52,62")
    p.add_argument("--rounds-small", type=int, default=10)
    p.add_argument("--rounds-large", type=int, default=30)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--similarity-mode", type=str, default="jaccard_cosine_hybrid", choices=["jaccard", "jaccard_cosine_hybrid"])
    p.add_argument("--lambda-jaccard", type=float, default=0.25)
    p.add_argument("--tau-cos-min", type=float, default=0.0)
    p.add_argument("--lvp-alpha", type=float, default=0.53)
    p.add_argument("--tau-grid", type=str, default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--attack-scale", type=float, default=5.0)
    p.add_argument("--network-eval-mode", type=str, default="proxy", choices=["proxy", "refit"])
    p.add_argument("--local-fit-maxiter", type=int, default=10)
    p.add_argument("--eval-fit-maxiter", type=int, default=0)
    p.add_argument("--no-reuters", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_csv_ints(args.seed_list)
    taus = _parse_csv_floats(args.tau_grid)

    base_path = Path(args.base_path).resolve()
    mcc_df = load_mcc_series(base_path)
    exog = _build_exogenous(base_path, mcc_df, use_reuters=not args.no_reuters)
    clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=args.n_clients,
        column_partition="contiguous",
    )
    profiles = build_client_information_profiles(clients, "mcc")

    ModelClass = MODEL_REGISTRY[args.model]
    rounds_set = [int(args.rounds_small), int(args.rounds_large)]

    rows: List[Dict[str, Any]] = []

    for rounds in rounds_set:
        for seed in seeds:
            for tau in taus:
                exp = run_one_model(
                    args.model,
                    ModelClass,
                    clients,
                    profiles,
                    aggregator="lvp",
                    rounds=rounds,
                    local_epochs=args.local_epochs,
                    malicious_frac=0.25,
                    seed=seed,
                    attack_strategy="noise_colluded",
                    attack_scale=float(args.attack_scale),
                    similarity_tau=float(tau),
                    similarity_mode=args.similarity_mode,
                    lambda_jaccard=float(args.lambda_jaccard),
                    tau_cos_min=float(args.tau_cos_min),
                    lvp_alpha=float(args.lvp_alpha),
                    lvp_self_weight=0.0,
                    krum_f=-1,
                    local_fit_maxiter=args.local_fit_maxiter,
                    eval_fit_maxiter=args.eval_fit_maxiter,
                    strict_errors=True,
                    network_eval_mode=args.network_eval_mode,
                )
                value = _final_mae(exp)
                rows.append(
                    {
                        "rounds": rounds,
                        "seed": seed,
                        "similarity_tau": float(tau),
                        "final_network_mae": value,
                    }
                )
                print(f"rounds={rounds} seed={seed} tau={tau:.3f} final_mae={value:.6f}")

    summary_rows: List[Dict[str, Any]] = []
    for rounds in rounds_set:
        for tau in taus:
            vals = [
                float(r["final_network_mae"])
                for r in rows
                if int(r["rounds"]) == rounds and abs(float(r["similarity_tau"]) - float(tau)) < 1e-12
            ]
            arr = np.asarray(vals, dtype=float)
            summary_rows.append(
                {
                    "rounds": rounds,
                    "similarity_tau": float(tau),
                    "n": int(arr.size),
                    "mean_final_mae": float(np.mean(arr)) if arr.size else float("inf"),
                    "std_final_mae": float(np.std(arr, ddof=0)) if arr.size else float("inf"),
                    "median_final_mae": float(np.median(arr)) if arr.size else float("inf"),
                }
            )

    payload = {
        "config": {
            "model": args.model,
            "n_clients": args.n_clients,
            "seeds": seeds,
            "rounds_small": args.rounds_small,
            "rounds_large": args.rounds_large,
            "local_epochs": args.local_epochs,
            "similarity_mode": args.similarity_mode,
            "lambda_jaccard": args.lambda_jaccard,
            "tau_cos_min": args.tau_cos_min,
            "lvp_alpha": args.lvp_alpha,
            "attack_strategy": "noise_colluded",
            "attack_scale": args.attack_scale,
            "network_eval_mode": args.network_eval_mode,
            "tau_grid": taus,
        },
        "rows": rows,
        "summary_rows": summary_rows,
    }

    json_path = out_dir / "selected_scenario_tau_ablation_two_rounds.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt

    for rounds in rounds_set:
        sub = [r for r in summary_rows if int(r["rounds"]) == rounds]
        sub = sorted(sub, key=lambda r: float(r["similarity_tau"]))
        xs = [float(r["similarity_tau"]) for r in sub]
        ys = [float(r["mean_final_mae"]) for r in sub]
        es = [float(r["std_final_mae"]) for r in sub]

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.errorbar(xs, ys, yerr=es, fmt="o-", linewidth=2.2, markersize=5, capsize=4)
        ax.set_xlabel("similarity_tau")
        ax.set_ylabel("Final network MAE")
        ax.set_title(f"Selected scenario: error vs similarity_tau (rounds={rounds})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = out_dir / f"selected_scenario_tau_curve_rounds_{rounds}.png"
        fig.savefig(fig_path, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {fig_path}")

    md_lines = [
        "# Selected Scenario Tau Ablation (Two Rounds)",
        "",
        f"Model: {args.model}",
        f"Similarity mode: {args.similarity_mode}",
        f"lambda_jaccard: {args.lambda_jaccard}",
        f"tau_cos_min: {args.tau_cos_min}",
        f"lvp_alpha: {args.lvp_alpha}",
        "",
        "## Best tau by rounds",
    ]

    for rounds in rounds_set:
        sub = [r for r in summary_rows if int(r["rounds"]) == rounds]
        best = min(sub, key=lambda r: float(r["mean_final_mae"]))
        md_lines.append(
            f"- rounds={rounds}: best_tau={best['similarity_tau']:.3f}, mean_final_mae={best['mean_final_mae']:.6f}, std={best['std_final_mae']:.6f}"
        )

    md_path = out_dir / "selected_scenario_tau_ablation_two_rounds_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
