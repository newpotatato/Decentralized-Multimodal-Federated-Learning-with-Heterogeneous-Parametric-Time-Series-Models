#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
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


def _mean_std(vals: List[float]) -> Dict[str, float]:
    arr = np.asarray(vals, dtype=float)
    return {
        "mean": float(np.mean(arr)) if arr.size else float("inf"),
        "std": float(np.std(arr, ddof=0)) if arr.size else float("inf"),
        "n": int(arr.size),
    }


def _evaluate_grid_point(task: Dict) -> Dict[str, float]:
    """Run one tau or alpha grid point across all seeds; process-safe for Windows spawn mode."""
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
    model_name = str(task.get("model_name", "DynamicLinearModel"))
    ModelClass = MODEL_REGISTRY[model_name]

    kind = str(task["kind"])
    value = float(task["value"])
    fixed_tau = float(task["fixed_tau"])
    fixed_alpha = float(task["fixed_alpha"])
    seeds = [int(s) for s in task["seeds"]]

    vals: List[float] = []
    for seed in seeds:
        exp = run_one_model(
            model_name,
            ModelClass,
            clients,
            profiles,
            aggregator="lvp",
            rounds=int(task["rounds"]),
            local_epochs=int(task["local_epochs"]),
            malicious_frac=float(task["malicious_frac"]),
            seed=seed,
            attack_strategy=str(task["attack_strategy"]),
            attack_scale=float(task["attack_scale"]),
            similarity_tau=(value if kind == "tau" else fixed_tau),
            similarity_mode=str(task["similarity_mode"]),
            lambda_jaccard=float(task["lambda_jaccard"]),
            tau_cos_min=float(task["tau_cos_min"]),
            lvp_alpha=(fixed_alpha if kind == "tau" else value),
            strict_errors=True,
            network_eval_mode=str(task["network_eval_mode"]),
            num_workers=int(task["num_workers"]),
            local_fit_maxiter=int(task.get("local_fit_maxiter", 10)),
        )
        vals.append(_final_mae(exp))

    st = _mean_std(vals)
    return {
        kind: value,
        "mean": float(st["mean"]),
        "std": float(st["std"]),
        "n": int(st["n"]),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Kappa_0/alpha ablation for selected 10-round article scenario")
    p.add_argument("--base-path", type=str, default=str(_FL.parent))
    p.add_argument("--out-dir", type=str, default=str(_FL / "artifacts" / "ablation_tau_alpha_article_rounds10"))
    p.add_argument("--seed-list", type=str, default="42,52,62")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--local-fit-maxiter", type=int, default=10)
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--model", type=str, default="DynamicLinearModel", choices=sorted(MODEL_REGISTRY.keys()))
    p.add_argument("--column-partition", type=str, default="contiguous", choices=["contiguous", "strided", "random", "random_strided"])
    p.add_argument("--attack-strategy", type=str, default="noise_colluded")
    p.add_argument("--attack-scale", type=float, default=5.0)
    p.add_argument("--malicious-frac", type=float, default=0.25)
    p.add_argument("--network-eval-mode", type=str, default="proxy", choices=["proxy", "refit"])
    p.add_argument("--similarity-mode", type=str, default="jaccard", choices=["jaccard", "jaccard_cosine_hybrid"])
    p.add_argument("--lambda-jaccard", type=float, default=0.5)
    p.add_argument("--tau-cos-min", type=float, default=-1.0)
    p.add_argument("--fixed-alpha", type=float, default=0.53)
    p.add_argument("--fixed-tau", type=float, default=0.35)
    p.add_argument("--tau-grid", type=str, default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70")
    p.add_argument("--alpha-grid", type=str, default="0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.53,0.60")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--grid-workers", type=int, default=1)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.base_path).resolve()

    seeds = [int(s.strip()) for s in args.seed_list.split(",") if s.strip()]
    tau_grid = [float(x.strip()) for x in args.tau_grid.split(",") if x.strip()]
    alpha_grid = [float(x.strip()) for x in args.alpha_grid.split(",") if x.strip()]

    base_task = {
        "base_path": str(base),
        "model_name": str(args.model),
        "rounds": int(args.rounds),
        "local_epochs": int(args.local_epochs),
        "local_fit_maxiter": int(args.local_fit_maxiter),
        "n_clients": int(args.n_clients),
        "column_partition": str(args.column_partition),
        "attack_strategy": str(args.attack_strategy),
        "attack_scale": float(args.attack_scale),
        "malicious_frac": float(args.malicious_frac),
        "network_eval_mode": str(args.network_eval_mode),
        "similarity_mode": str(args.similarity_mode),
        "lambda_jaccard": float(args.lambda_jaccard),
        "tau_cos_min": float(args.tau_cos_min),
        "fixed_alpha": float(args.fixed_alpha),
        "fixed_tau": float(args.fixed_tau),
        "seeds": seeds,
        "num_workers": int(args.num_workers),
    }

    tau_tasks = [{**base_task, "kind": "tau", "value": float(tau)} for tau in tau_grid]
    alpha_tasks = [{**base_task, "kind": "alpha", "value": float(alpha)} for alpha in alpha_grid]

    grid_workers = max(1, int(args.grid_workers))
    if grid_workers == 1:
        tau_rows = [_evaluate_grid_point(task) for task in tau_tasks]
        alpha_rows = [_evaluate_grid_point(task) for task in alpha_tasks]
    else:
        # Process-level parallel grid search gives the largest CPU speedup on independent points.
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=grid_workers, mp_context=ctx) as ex:
            tau_rows = list(ex.map(_evaluate_grid_point, tau_tasks))
            alpha_rows = list(ex.map(_evaluate_grid_point, alpha_tasks))

    tau_rows.sort(key=lambda r: r["tau"])
    alpha_rows.sort(key=lambda r: r["alpha"])

    best_tau_row = min(tau_rows, key=lambda r: r["mean"])
    best_alpha_row = min(alpha_rows, key=lambda r: r["mean"])

    _FONT = 12
    _FONT_SM = 10
    _COLOR_LINE = "#2171b5"
    _COLOR_BAND = "#9ecae1"
    _COLOR_BEST = "#d62728"

    # Plot kappa_0 (tau) ablation
    fig1, ax1 = plt.subplots(figsize=(7.5, 4.8))
    xs = np.asarray([r["tau"] for r in tau_rows])
    ys = np.asarray([r["mean"] for r in tau_rows])
    sd = np.asarray([r["std"] for r in tau_rows])
    ax1.fill_between(xs, ys - sd, ys + sd, color=_COLOR_BAND, alpha=0.40, label=r"$\pm 1\,\sigma$ (across seeds)")
    ax1.plot(xs, ys, "o-", color=_COLOR_LINE, linewidth=2.0, markersize=5, label="Mean final MAE")
    ax1.axvline(best_tau_row["tau"], color=_COLOR_BEST, linewidth=1.4, linestyle="--", alpha=0.8)
    ax1.scatter(
        [best_tau_row["tau"]], [best_tau_row["mean"]],
        s=110, marker="*", color=_COLOR_BEST, zorder=5,
        label=fr"Best $\kappa_0 = {best_tau_row['tau']:.2f}$",
    )
    ax1.set_xlabel(r"$\kappa_0$  (Jaccard similarity threshold)", fontsize=_FONT)
    ax1.set_ylabel("Final network MAE", fontsize=_FONT)
    ax1.set_title(r"Ablation study: $\kappa_0$ sweep (10 rounds)", fontsize=_FONT)
    ax1.tick_params(labelsize=_FONT_SM)
    ax1.grid(True, axis="y", alpha=0.35, linestyle=":")
    ax1.grid(True, axis="x", alpha=0.20, linestyle=":")
    ax1.legend(loc="best", fontsize=_FONT_SM, framealpha=0.85)
    fig1.tight_layout()
    tau_png = out_dir / "tau_ablation_rounds10_article.png"
    fig1.savefig(tau_png, dpi=200, bbox_inches="tight")
    plt.close(fig1)

    # Plot alpha ablation
    fig2, ax2 = plt.subplots(figsize=(7.5, 4.8))
    xs = np.asarray([r["alpha"] for r in alpha_rows])
    ys = np.asarray([r["mean"] for r in alpha_rows])
    sd = np.asarray([r["std"] for r in alpha_rows])
    ax2.fill_between(xs, ys - sd, ys + sd, color=_COLOR_BAND, alpha=0.40, label=r"$\pm 1\,\sigma$ (across seeds)")
    ax2.plot(xs, ys, "o-", color=_COLOR_LINE, linewidth=2.0, markersize=5, label="Mean final MAE")
    ax2.axvline(best_alpha_row["alpha"], color=_COLOR_BEST, linewidth=1.4, linestyle="--", alpha=0.8)
    ax2.scatter(
        [best_alpha_row["alpha"]], [best_alpha_row["mean"]],
        s=110, marker="*", color=_COLOR_BEST, zorder=5,
        label=fr"Best $\alpha = {best_alpha_row['alpha']:.2f}$",
    )
    ax2.set_xlabel(r"$\alpha$  (LVP synchronization step)", fontsize=_FONT)
    ax2.set_ylabel("Final network MAE", fontsize=_FONT)
    ax2.set_title(r"Ablation study: $\alpha$ sweep (10 rounds)", fontsize=_FONT)
    ax2.tick_params(labelsize=_FONT_SM)
    ax2.grid(True, axis="y", alpha=0.35, linestyle=":")
    ax2.grid(True, axis="x", alpha=0.20, linestyle=":")
    ax2.legend(loc="best", fontsize=_FONT_SM, framealpha=0.85)
    fig2.tight_layout()
    alpha_png = out_dir / "alpha_ablation_rounds10_article.png"
    fig2.savefig(alpha_png, dpi=200, bbox_inches="tight")
    plt.close(fig2)

    summary = {
        "config": {
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "seeds": seeds,
            "column_partition": args.column_partition,
            "attack_strategy": args.attack_strategy,
            "attack_scale": args.attack_scale,
            "malicious_frac": args.malicious_frac,
            "network_eval_mode": args.network_eval_mode,
            "similarity_mode": args.similarity_mode,
            "lambda_jaccard": args.lambda_jaccard,
            "tau_cos_min": args.tau_cos_min,
            "fixed_alpha_for_tau": args.fixed_alpha,
            "fixed_tau_for_alpha": args.fixed_tau,
            "num_workers": args.num_workers,
            "grid_workers": args.grid_workers,
        },
        "tau_rows": tau_rows,
        "alpha_rows": alpha_rows,
        "best_tau": best_tau_row,
        "best_alpha": best_alpha_row,
        "figures": {
            "tau": str(tau_png),
            "alpha": str(alpha_png),
        },
    }
    summary_json = out_dir / "tau_alpha_ablation_rounds10_article_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = out_dir / "tau_alpha_ablation_rounds10_article_report.md"
    report.write_text(
        "\n".join(
            [
                "# Kappa_0/Alpha Ablation (rounds=10 article scenario)",
                "",
                f"Best kappa_0: {best_tau_row['tau']:.6f} (mean={best_tau_row['mean']:.6f}, std={best_tau_row['std']:.6f})",
                f"Best alpha: {best_alpha_row['alpha']:.6f} (mean={best_alpha_row['mean']:.6f}, std={best_alpha_row['std']:.6f})",
                "",
                f"Kappa_0 figure: {tau_png}",
                f"Alpha figure: {alpha_png}",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved: {tau_png}")
    print(f"Saved: {alpha_png}")
    print(f"Saved: {summary_json}")
    print(f"Saved: {report}")


if __name__ == "__main__":
    main()
