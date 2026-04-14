#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def _mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=0))


def _final_mae(exp: Dict) -> float:
    hist = exp.get("history") or []
    if not hist:
        return float("inf")
    return float(hist[-1].get("network_mae", float("inf")))


def _evaluate_tau_grid(
    *,
    tau_grid: List[float],
    seeds: List[int],
    rounds: int,
    local_epochs: int,
    n_clients: int,
    column_partition: str,
    malicious_frac: float,
    attack_strategy: str,
    attack_scale: float,
    similarity_mode: str,
    lambda_jaccard: float,
    tau_cos_min: float,
    lvp_alpha: float,
    network_eval_mode: str,
    base_path: Path,
    model_name: str = "DynamicLinearModel",
) -> List[Dict[str, float]]:
    mcc_df = load_mcc_series(base_path)
    exog = _build_exogenous(base_path, mcc_df, use_reuters=True)
    clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=n_clients,
        column_partition=column_partition,
    )
    profiles = _topic_profiles(clients)
    ModelClass = MODEL_REGISTRY[model_name]

    rows: List[Dict[str, float]] = []
    for tau in tau_grid:
        vals: List[float] = []
        for seed in seeds:
            exp = run_one_model(
                model_name,
                ModelClass,
                clients,
                profiles,
                aggregator="lvp",
                rounds=rounds,
                local_epochs=local_epochs,
                malicious_frac=malicious_frac,
                seed=seed,
                attack_strategy=attack_strategy,
                attack_scale=attack_scale,
                similarity_tau=float(tau),
                similarity_mode=similarity_mode,
                lambda_jaccard=lambda_jaccard,
                tau_cos_min=tau_cos_min,
                lvp_alpha=lvp_alpha,
                lvp_self_weight=0.0,
                strict_errors=True,
                network_eval_mode=network_eval_mode,
                num_workers=1,
            )
            vals.append(_final_mae(exp))
        mean_v, std_v = _mean_std(vals)
        rows.append({"tau": float(tau), "mean": mean_v, "std": std_v, "n": float(len(vals))})
    return rows


def _plot(rows: List[Dict[str, float]], out_path: Path, title: str) -> None:
    taus = [r["tau"] for r in rows]
    means = [r["mean"] for r in rows]
    stds = [r["std"] for r in rows]
    best_idx = int(np.argmin(means)) if rows else 0

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.plot(taus, means, "o-", linewidth=2.2, markersize=5, label="mean final MAE")
    ax.fill_between(taus, np.asarray(means) - np.asarray(stds), np.asarray(means) + np.asarray(stds), alpha=0.15, label="+-1 std")
    ax.scatter([taus[best_idx]], [means[best_idx]], s=110, marker="*", color="red", zorder=5, label=f"best tau={taus[best_idx]:.2f}")
    ax.set_xlabel("similarity_tau")
    ax.set_ylabel("Final network MAE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Smart tau search for hybrid LVP topology")
    p.add_argument("--base-path", type=str, default=str(_FL.parent))
    p.add_argument("--out-dir", type=str, default=str(_FL / "artifacts" / "article_package_current_run_20260410" / "experiment_with_more_rounds_hybrid" / "tau_ablation"))
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--column-partition", type=str, default="contiguous", choices=["contiguous", "strided"])
    p.add_argument("--malicious-frac", type=float, default=0.25)
    p.add_argument("--attack-strategy", type=str, default="noise_colluded")
    p.add_argument("--attack-scale", type=float, default=5.0)
    p.add_argument("--similarity-mode", type=str, default="jaccard_cosine_hybrid", choices=["jaccard", "jaccard_cosine_hybrid"])
    p.add_argument("--lambda-jaccard", type=float, default=0.5)
    p.add_argument("--tau-cos-min", type=float, default=-1.0)
    p.add_argument("--lvp-alpha", type=float, default=0.6)
    p.add_argument("--network-eval-mode", type=str, default="proxy", choices=["proxy", "refit"])
    p.add_argument("--seeds", type=str, default="42,52,62")
    p.add_argument("--coarse-grid", type=str, default="0.30,0.40,0.50,0.60,0.70,0.80")
    p.add_argument("--refine-width", type=float, default=0.08)
    p.add_argument("--refine-step", type=float, default=0.01)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    coarse_grid = [float(x.strip()) for x in args.coarse_grid.split(",") if x.strip()]

    base_path = Path(args.base_path).resolve()
    coarse_rows = _evaluate_tau_grid(
        tau_grid=coarse_grid,
        seeds=seeds,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        n_clients=args.n_clients,
        column_partition=args.column_partition,
        malicious_frac=args.malicious_frac,
        attack_strategy=args.attack_strategy,
        attack_scale=args.attack_scale,
        similarity_mode=args.similarity_mode,
        lambda_jaccard=args.lambda_jaccard,
        tau_cos_min=args.tau_cos_min,
        lvp_alpha=args.lvp_alpha,
        network_eval_mode=args.network_eval_mode,
        base_path=base_path,
    )
    coarse_best = min(coarse_rows, key=lambda r: (r["mean"], r["std"]))
    best_tau = float(coarse_best["tau"])

    fine_min = max(0.05, best_tau - float(args.refine_width))
    fine_max = min(0.95, best_tau + float(args.refine_width))
    fine_grid = list(np.round(np.arange(fine_min, fine_max + 1e-12, float(args.refine_step)), 4))
    fine_rows = _evaluate_tau_grid(
        tau_grid=fine_grid,
        seeds=seeds,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        n_clients=args.n_clients,
        column_partition=args.column_partition,
        malicious_frac=args.malicious_frac,
        attack_strategy=args.attack_strategy,
        attack_scale=args.attack_scale,
        similarity_mode=args.similarity_mode,
        lambda_jaccard=args.lambda_jaccard,
        tau_cos_min=args.tau_cos_min,
        lvp_alpha=args.lvp_alpha,
        network_eval_mode=args.network_eval_mode,
        base_path=base_path,
    )
    best_row = min(fine_rows, key=lambda r: (r["mean"], r["std"]))
    best_tau = float(best_row["tau"])

    _plot(fine_rows, out_dir / "tau_ablation_hybrid.png", "Hybrid LVP tau search (coarse-to-fine)")

    summary = {
        "config": {
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "seeds": seeds,
            "column_partition": args.column_partition,
            "malicious_frac": args.malicious_frac,
            "attack_strategy": args.attack_strategy,
            "attack_scale": args.attack_scale,
            "similarity_mode": args.similarity_mode,
            "lambda_jaccard": args.lambda_jaccard,
            "tau_cos_min": args.tau_cos_min,
            "lvp_alpha": args.lvp_alpha,
            "network_eval_mode": args.network_eval_mode,
        },
        "coarse_rows": coarse_rows,
        "fine_rows": fine_rows,
        "best_tau": best_tau,
        "best_row": best_row,
    }
    (out_dir / "tau_search_hybrid_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / "tau_search_hybrid_report.md").write_text(
        "\n".join([
            "# Hybrid tau search",
            "",
            f"Best tau: {best_tau:.6f}",
            f"Mean final MAE: {best_row['mean']:.6f}",
            f"Std final MAE: {best_row['std']:.6f}",
            f"Figure: {out_dir / 'tau_ablation_hybrid.png'}",
        ]),
        encoding="utf-8",
    )

    print(f"Saved: {out_dir / 'tau_ablation_hybrid.png'}")
    print(f"Saved: {out_dir / 'tau_search_hybrid_summary.json'}")
    print(f"Best tau={best_tau:.3f}")


if __name__ == "__main__":
    main()
