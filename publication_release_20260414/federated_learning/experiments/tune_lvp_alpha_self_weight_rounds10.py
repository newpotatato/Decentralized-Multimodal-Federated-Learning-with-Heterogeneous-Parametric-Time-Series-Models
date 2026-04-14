#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

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


def _objective(mean: float, std: float, beta: float) -> float:
    return float(mean + beta * std)


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

    alpha = float(task["alpha"])
    self_weight = float(task["self_weight"])
    tau = float(task["tau"])
    beta = float(task["score_beta"])
    seeds = [int(s) for s in task["seeds"]]

    vals: List[float] = []
    for seed in seeds:
        exp = run_one_model(
            "DynamicLinearModel",
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
            similarity_tau=tau,
            similarity_mode=str(task["similarity_mode"]),
            lambda_jaccard=float(task["lambda_jaccard"]),
            tau_cos_min=float(task["tau_cos_min"]),
            lvp_alpha=alpha,
            lvp_self_weight=self_weight,
            strict_errors=True,
            network_eval_mode=str(task["network_eval_mode"]),
            num_workers=int(task["num_workers"]),
        )
        vals.append(_final_mae(exp))

    st = _mean_std(vals)
    obj = _objective(st["mean"], st["std"], beta)
    return {
        "alpha": alpha,
        "self_weight": self_weight,
        "mean": float(st["mean"]),
        "std": float(st["std"]),
        "objective": obj,
        "n": int(st["n"]),
    }


def _plot_heatmap(ax: Axes, rows: List[Dict], alpha_grid: List[float], self_grid: List[float], value_key: str, title: str, best_point: Tuple[float, float] | None = None) -> None:
    mat = np.full((len(self_grid), len(alpha_grid)), np.nan, dtype=float)
    for row in rows:
        ai = alpha_grid.index(row["alpha"])
        si = self_grid.index(row["self_weight"])
        mat[si, ai] = float(row[value_key])
    im = ax.imshow(mat, origin="lower", aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(alpha_grid)))
    ax.set_xticklabels([f"{x:.2f}" for x in alpha_grid], rotation=30)
    ax.set_yticks(range(len(self_grid)))
    ax.set_yticklabels([f"{x:.2f}" for x in self_grid])
    ax.set_xlabel("lvp_alpha")
    ax.set_ylabel("lvp_self_weight")
    ax.set_title(title)
    if best_point is not None:
        ai = alpha_grid.index(best_point[0])
        si = self_grid.index(best_point[1])
        ax.scatter([ai], [si], s=120, marker="*", color="red", edgecolors="white", linewidths=0.8)
    return im


def main() -> None:
    p = argparse.ArgumentParser(description="Tune LVP alpha and self_weight on the rounds=10 scenario")
    p.add_argument("--base-path", type=str, default=str(_FL.parent))
    p.add_argument("--out-dir", type=str, default=str(_FL / "artifacts" / "tune_lvp_alpha_self_weight_rounds10"))
    p.add_argument("--seed-list", type=str, default="42,52,62")
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
    p.add_argument("--tau", type=float, default=0.35)
    p.add_argument("--alpha-grid", type=str, default="0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60")
    p.add_argument("--self-weight-grid", type=str, default="0.00,0.05,0.10,0.15,0.20,0.25,0.30")
    p.add_argument("--score-beta", type=float, default=0.5)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--grid-workers", type=int, default=4)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.base_path).resolve()

    seeds = [int(s.strip()) for s in args.seed_list.split(",") if s.strip()]
    alpha_grid = [float(x.strip()) for x in args.alpha_grid.split(",") if x.strip()]
    self_grid = [float(x.strip()) for x in args.self_weight_grid.split(",") if x.strip()]

    tasks: List[Dict] = []
    for alpha in alpha_grid:
        for self_weight in self_grid:
            tasks.append({
                "base_path": str(base),
                "seeds": seeds,
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
                "tau": float(args.tau),
                "alpha": float(alpha),
                "self_weight": float(self_weight),
                "score_beta": float(args.score_beta),
                "num_workers": int(args.num_workers),
            })

    grid_workers = max(1, int(args.grid_workers))
    if grid_workers == 1:
        rows = [_evaluate_combo(task) for task in tasks]
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=grid_workers, mp_context=ctx) as ex:
            rows = list(ex.map(_evaluate_combo, tasks))

    rows.sort(key=lambda r: (r["objective"], r["mean"], r["std"]))
    best = rows[0] if rows else {}

    alpha_rows: Dict[float, List[Dict]] = {a: [] for a in alpha_grid}
    self_rows: Dict[float, List[Dict]] = {s: [] for s in self_grid}
    for row in rows:
        alpha_rows[row["alpha"]].append(row)
        self_rows[row["self_weight"]].append(row)

    best_alpha_rows = []
    for alpha in alpha_grid:
        subset = alpha_rows[alpha]
        if subset:
            best_alpha_rows.append(min(subset, key=lambda r: r["objective"]))
    best_self_rows = []
    for s in self_grid:
        subset = self_rows[s]
        if subset:
            best_self_rows.append(min(subset, key=lambda r: r["objective"]))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10.0))
    ax_heat = axes[0, 0]
    im = _plot_heatmap(ax_heat, rows, alpha_grid, self_grid, "objective", f"Objective = mean + {args.score_beta:g}*std", best_point=(best.get("alpha"), best.get("self_weight")))
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    ax1 = axes[0, 1]
    x = [r["alpha"] for r in best_alpha_rows]
    y = [r["objective"] for r in best_alpha_rows]
    ax1.plot(x, y, "o-", linewidth=2.1, markersize=5)
    if best_alpha_rows:
        best_r = min(best_alpha_rows, key=lambda r: r["objective"])
        ax1.scatter([best_r["alpha"]], [best_r["objective"]], s=90, marker="*", color="red")
    ax1.set_xlabel("lvp_alpha")
    ax1.set_ylabel("Objective")
    ax1.set_title("Best objective by alpha")
    ax1.grid(True, alpha=0.3)

    ax2 = axes[1, 0]
    x = [r["self_weight"] for r in best_self_rows]
    y = [r["objective"] for r in best_self_rows]
    ax2.plot(x, y, "o-", linewidth=2.1, markersize=5)
    if best_self_rows:
        best_r = min(best_self_rows, key=lambda r: r["objective"])
        ax2.scatter([best_r["self_weight"]], [best_r["objective"]], s=90, marker="*", color="red")
    ax2.set_xlabel("lvp_self_weight")
    ax2.set_ylabel("Objective")
    ax2.set_title("Best objective by self_weight")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 1]
    x = [r["self_weight"] for r in rows]
    y = [r["alpha"] for r in rows]
    c = [r["objective"] for r in rows]
    sc = ax3.scatter(x, y, c=c, cmap="viridis", s=55)
    ax3.scatter([best.get("self_weight")], [best.get("alpha")], s=120, marker="*", color="red", edgecolors="white", linewidths=0.8)
    ax3.set_xlabel("lvp_self_weight")
    ax3.set_ylabel("lvp_alpha")
    ax3.set_title("Search cloud")
    ax3.grid(True, alpha=0.25)
    fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04)

    fig.suptitle(f"LVP tuning on rounds={args.rounds} (seeds={len(seeds)})", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig_path = out_dir / "lvp_alpha_self_weight_tuning.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

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
            "fixed_tau": args.tau,
            "alpha_grid": alpha_grid,
            "self_weight_grid": self_grid,
            "score_beta": args.score_beta,
            "num_workers": args.num_workers,
            "grid_workers": args.grid_workers,
        },
        "rows": rows,
        "best": best,
        "figures": {
            "tuning": str(fig_path),
        },
    }
    summary_path = out_dir / "lvp_alpha_self_weight_tuning_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# LVP alpha/self_weight tuning",
        "",
        f"Best alpha: {best.get('alpha'):.6f}",
        f"Best self_weight: {best.get('self_weight'):.6f}",
        f"Mean final MAE: {best.get('mean'):.6f}",
        f"Std final MAE: {best.get('std'):.6f}",
        f"Objective: {best.get('objective'):.6f}",
        "",
        f"Figure: {fig_path}",
    ]
    report_path = out_dir / "lvp_alpha_self_weight_tuning_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved: {fig_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {report_path}")
    print(f"Best alpha={best.get('alpha'):.3f}, self_weight={best.get('self_weight'):.3f}, objective={best.get('objective'):.3f}")


if __name__ == "__main__":
    main()
