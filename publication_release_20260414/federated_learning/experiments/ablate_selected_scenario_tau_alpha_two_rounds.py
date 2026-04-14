#!/usr/bin/env python3
"""Joint ablation for similarity_tau and lvp_alpha on two round budgets.

Produces 4 plots:
1) tau curve for rounds_small
2) tau curve for rounds_large
3) alpha curve for rounds_small
4) alpha curve for rounds_large
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
    out: List[int] = []
    for x in (text or "").split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def _parse_csv_floats(text: str) -> List[float]:
    out: List[float] = []
    for x in (text or "").split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    return out


def _final_mae(exp: Dict[str, Any]) -> float:
    hist = exp.get("history", [])
    if not hist:
        return float("inf")
    return float(hist[-1].get("network_mae", float("inf")))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tau+alpha ablation for two selected scenarios")
    p.add_argument("--base-path", type=str, default=str(_FL.parent))
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_FL / "artifacts" / "selected_scenario_tau_alpha_ablation_two_rounds"),
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
    p.add_argument("--fixed-alpha-for-tau", type=float, default=0.53)
    p.add_argument("--fixed-tau-small-for-alpha", type=float, default=0.60)
    p.add_argument("--fixed-tau-large-for-alpha", type=float, default=0.50)
    p.add_argument("--tau-grid", type=str, default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--alpha-grid", type=str, default="0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.53")
    p.add_argument("--attack-scale", type=float, default=5.0)
    p.add_argument("--network-eval-mode", type=str, default="proxy", choices=["proxy", "refit"])
    p.add_argument("--local-fit-maxiter", type=int, default=10)
    p.add_argument("--eval-fit-maxiter", type=int, default=0)
    p.add_argument("--no-reuters", action="store_true")
    return p.parse_args()


def _aggregate_mean_std(rows: List[Dict[str, Any]], key_name: str, rounds: int, values: List[float]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for val in values:
        sub = [
            float(r["final_network_mae"])
            for r in rows
            if int(r["rounds"]) == rounds and abs(float(r[key_name]) - float(val)) < 1e-12
        ]
        arr = np.asarray(sub, dtype=float)
        out.append(
            {
                "rounds": int(rounds),
                key_name: float(val),
                "n": int(arr.size),
                "mean_final_mae": float(np.mean(arr)) if arr.size else float("inf"),
                "std_final_mae": float(np.std(arr, ddof=0)) if arr.size else float("inf"),
                "median_final_mae": float(np.median(arr)) if arr.size else float("inf"),
            }
        )
    return out


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = _parse_csv_ints(args.seed_list)
    tau_grid = _parse_csv_floats(args.tau_grid)
    alpha_grid = _parse_csv_floats(args.alpha_grid)
    rounds_set = [int(args.rounds_small), int(args.rounds_large)]

    base = Path(args.base_path).resolve()
    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=not args.no_reuters)
    clients = build_clients_from_mcc(mcc_df, exog, n_clients=args.n_clients, column_partition="contiguous")
    profiles = build_client_information_profiles(clients, "mcc")
    ModelClass = MODEL_REGISTRY[args.model]

    tau_rows: List[Dict[str, Any]] = []
    alpha_rows: List[Dict[str, Any]] = []

    # Tau sweep with fixed alpha for each rounds setting.
    for rounds in rounds_set:
        for seed in seeds:
            for tau in tau_grid:
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
                    lvp_alpha=float(args.fixed_alpha_for_tau),
                    lvp_self_weight=0.0,
                    krum_f=-1,
                    local_fit_maxiter=args.local_fit_maxiter,
                    eval_fit_maxiter=args.eval_fit_maxiter,
                    strict_errors=True,
                    network_eval_mode=args.network_eval_mode,
                )
                mae = _final_mae(exp)
                tau_rows.append(
                    {
                        "rounds": rounds,
                        "seed": seed,
                        "similarity_tau": float(tau),
                        "final_network_mae": mae,
                    }
                )
                print(f"[tau] rounds={rounds} seed={seed} tau={tau:.3f} mae={mae:.6f}")

    # Alpha sweep with fixed tau per rounds setting.
    fixed_tau_map = {
        int(args.rounds_small): float(args.fixed_tau_small_for_alpha),
        int(args.rounds_large): float(args.fixed_tau_large_for_alpha),
    }
    for rounds in rounds_set:
        fixed_tau = fixed_tau_map[int(rounds)]
        for seed in seeds:
            for alpha in alpha_grid:
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
                    similarity_tau=float(fixed_tau),
                    similarity_mode=args.similarity_mode,
                    lambda_jaccard=float(args.lambda_jaccard),
                    tau_cos_min=float(args.tau_cos_min),
                    lvp_alpha=float(alpha),
                    lvp_self_weight=0.0,
                    krum_f=-1,
                    local_fit_maxiter=args.local_fit_maxiter,
                    eval_fit_maxiter=args.eval_fit_maxiter,
                    strict_errors=True,
                    network_eval_mode=args.network_eval_mode,
                )
                mae = _final_mae(exp)
                alpha_rows.append(
                    {
                        "rounds": rounds,
                        "seed": seed,
                        "lvp_alpha": float(alpha),
                        "fixed_tau": float(fixed_tau),
                        "final_network_mae": mae,
                    }
                )
                print(f"[alpha] rounds={rounds} seed={seed} alpha={alpha:.3f} tau={fixed_tau:.3f} mae={mae:.6f}")

    tau_summary = []
    alpha_summary = []
    for rounds in rounds_set:
        tau_summary.extend(_aggregate_mean_std(tau_rows, "similarity_tau", rounds, tau_grid))
        alpha_summary.extend(_aggregate_mean_std(alpha_rows, "lvp_alpha", rounds, alpha_grid))

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
            "fixed_alpha_for_tau": args.fixed_alpha_for_tau,
            "fixed_tau_small_for_alpha": args.fixed_tau_small_for_alpha,
            "fixed_tau_large_for_alpha": args.fixed_tau_large_for_alpha,
            "tau_grid": tau_grid,
            "alpha_grid": alpha_grid,
            "attack_strategy": "noise_colluded",
            "attack_scale": args.attack_scale,
            "network_eval_mode": args.network_eval_mode,
        },
        "tau_rows": tau_rows,
        "alpha_rows": alpha_rows,
        "tau_summary": tau_summary,
        "alpha_summary": alpha_summary,
    }

    json_path = out_dir / "selected_scenario_tau_alpha_ablation_two_rounds.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt

    # 4 plots
    for rounds in rounds_set:
        sub_tau = sorted([r for r in tau_summary if int(r["rounds"]) == rounds], key=lambda x: float(x["similarity_tau"]))
        xs = [float(r["similarity_tau"]) for r in sub_tau]
        ys = [float(r["mean_final_mae"]) for r in sub_tau]
        es = [float(r["std_final_mae"]) for r in sub_tau]

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.errorbar(xs, ys, yerr=es, fmt="o-", linewidth=2.2, markersize=5, capsize=4)
        ax.set_xlabel("similarity_tau")
        ax.set_ylabel("Final network MAE")
        ax.set_title(f"Tau ablation: error vs similarity_tau (rounds={rounds})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p_tau = out_dir / f"tau_ablation_rounds_{rounds}.png"
        fig.savefig(p_tau, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p_tau}")

        sub_alpha = sorted([r for r in alpha_summary if int(r["rounds"]) == rounds], key=lambda x: float(x["lvp_alpha"]))
        xa = [float(r["lvp_alpha"]) for r in sub_alpha]
        ya = [float(r["mean_final_mae"]) for r in sub_alpha]
        ea = [float(r["std_final_mae"]) for r in sub_alpha]

        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        ax.errorbar(xa, ya, yerr=ea, fmt="s-", linewidth=2.2, markersize=5, capsize=4)
        ax.set_xlabel("lvp_alpha")
        ax.set_ylabel("Final network MAE")
        ax.set_title(f"Alpha ablation: error vs lvp_alpha (rounds={rounds}, tau={fixed_tau_map[int(rounds)]:.2f})")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p_alpha = out_dir / f"alpha_ablation_rounds_{rounds}.png"
        fig.savefig(p_alpha, dpi=170, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {p_alpha}")

    md_lines = [
        "# Tau+Alpha Ablation (Two Rounds)",
        "",
        "## Best settings",
    ]
    for rounds in rounds_set:
        best_tau = min([r for r in tau_summary if int(r["rounds"]) == rounds], key=lambda x: float(x["mean_final_mae"]))
        best_alpha = min([r for r in alpha_summary if int(r["rounds"]) == rounds], key=lambda x: float(x["mean_final_mae"]))
        md_lines.append(
            f"- rounds={rounds}: best_tau={best_tau['similarity_tau']:.3f} (mean={best_tau['mean_final_mae']:.6f}), best_alpha={best_alpha['lvp_alpha']:.3f} (mean={best_alpha['mean_final_mae']:.6f})"
        )

    md_path = out_dir / "selected_scenario_tau_alpha_ablation_two_rounds_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
