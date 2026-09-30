#!/usr/bin/env python3
"""Dedicated Figure 3 comparison for decentralized baselines.

Compares LVP with decentralized FedAvg, DeFTA, and BALANCE Push-Sum under a
single selected scenario on the best model from the ablation summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "core"))
sys.path.insert(0, str(_FL / "data_loaders"))

from data_utils import build_client_information_profiles, build_clients_from_mcc, load_mcc_series
from full_comparison_lvp import DEFAULT_ABLATION_SUMMARY, load_best_hparams
from run_real_experiments import MODEL_REGISTRY, _build_exogenous, run_one_model


DECENTRALIZED_AGGS = [
    "lvp",
    "decentralized_fedavg",
    "defta",
    "balance",
    "push_sum",
]


def _topic_profiles(clients: List, n_groups: int = 4) -> List:
    profiles = build_client_information_profiles(clients, "mcc")
    return [
        frozenset(p) | {f"sync_topic_{idx % n_groups}"}
        for idx, p in enumerate(profiles)
    ]


def _final_mae(exp: Dict[str, Any]) -> float:
    history = exp.get("history") or []
    if not history:
        return float("inf")
    return float(history[-1].get("network_mae", float("inf")))


def _plot_decentralized_comparison(
    results: List[Dict[str, Any]],
    model_name: str,
    out_path: Path,
    title_suffix: str,
    use_log_scale: bool = False,
    clip_outliers_sigma: float = 0.0,
) -> None:
    import matplotlib.pyplot as plt

    label_map = {
        "lvp": "LVP",
        "decentralized_fedavg": "Decentralized FedAvg",
        "defta": "DeFTA",
        "balance": "BALANCE",
        "push_sum": "Push-Sum",
    }
    order = {name: idx for idx, name in enumerate(DECENTRALIZED_AGGS)}

    subset = [r for r in results if r["model"] == model_name]
    subset.sort(key=lambda x: order.get(x["aggregator"], 99))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for exp in subset:
        agg = exp["aggregator"]
        xs = [int(r["round"]) for r in exp["history"]]
        ys = np.asarray([float(r["network_mae"]) for r in exp["history"]], dtype=float)
        
        # Optionally clip outliers
        if clip_outliers_sigma > 0:
            mean_y = np.nanmean(ys)
            std_y = np.nanstd(ys)
            threshold = mean_y + clip_outliers_sigma * std_y
            ys = np.minimum(ys, threshold)
        
        lw = 3.0 if agg == "lvp" else 1.8
        alpha = 1.0 if agg == "lvp" else 0.9
        ax.plot(xs, ys, "o-", linewidth=lw, markersize=5, alpha=alpha, label=label_map.get(agg, agg))

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Network MAE (absolute, amt units)")
    ax.set_title(f"Network MAE vs round — decentralized methods ({model_name}){title_suffix}")
    if use_log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Network MAE (log scale, amt units)")
    ax.grid(True, alpha=0.3, which="both" if use_log_scale else "major")
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare LVP with decentralized FL baselines.")
    parser.add_argument("--base-path", type=str, default=str(_FL.parent))
    parser.add_argument("--ablation-summary", type=str, default=str(DEFAULT_ABLATION_SUMMARY))
    parser.add_argument("--out-dir", type=str, default=str(_FL / "artifacts" / "decentralized_comparison"))
    parser.add_argument("--n-clients", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--malicious-frac", type=float, default=0.2)
    parser.add_argument("--attack-strategy", type=str, default="label_flip", choices=["label_flip", "noise", "noise_colluded", "random"])
    parser.add_argument("--attack-scale", type=float, default=2.5)
    parser.add_argument("--similarity-tau", type=float, default=None)
    parser.add_argument(
        "--similarity-mode",
        type=str,
        default="jaccard",
        choices=["jaccard", "jaccard_cosine_hybrid"],
    )
    parser.add_argument("--lambda-jaccard", type=float, default=0.5)
    parser.add_argument("--tau-cos-min", type=float, default=-1.0)
    parser.add_argument("--lvp-alpha", type=float, default=None)
    parser.add_argument("--lvp-self-weight", type=float, default=0.0)
    parser.add_argument("--network-eval-mode", type=str, default="proxy", choices=["proxy", "refit"])
    parser.add_argument("--no-reuters", action="store_true")
    parser.add_argument("--local-fit-maxiter", type=int, default=10)
    parser.add_argument("--eval-fit-maxiter", type=int, default=0)
    parser.add_argument("--column-partition", type=str, default="random", choices=["contiguous", "strided", "random", "random_strided"])
    parser.add_argument("--use-log-scale", action="store_true", help="Use log scale for network MAE y-axis")
    parser.add_argument("--clip-outliers-sigma", type=float, default=0.0, help="Clip outliers beyond mean+N*sigma (0=disabled)")
    args = parser.parse_args()

    base = Path(args.base_path).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_model, tau_star, alpha_star = load_best_hparams(Path(args.ablation_summary))
    if args.similarity_tau is not None:
        tau_star = float(args.similarity_tau)
    if args.lvp_alpha is not None:
        alpha_star = float(args.lvp_alpha)

    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=not args.no_reuters)
    clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=args.n_clients,
        column_partition=args.column_partition,
        partition_seed=args.seed,
    )
    if len(clients) < 2:
        raise RuntimeError("Need at least 2 clients.")
    profiles = _topic_profiles(clients)

    print(f"Best model: {best_model} | tau={tau_star:.3g} | alpha={alpha_star!s}")
    print(f"Scenario: partition={args.column_partition}, mal={args.malicious_frac:.2f}, attack={args.attack_strategy}, scale={args.attack_scale}")

    results: List[Dict[str, Any]] = []
    for idx, agg in enumerate(DECENTRALIZED_AGGS, start=1):
        ModelClass = MODEL_REGISTRY[best_model]
        print(f"[{idx}/{len(DECENTRALIZED_AGGS)}] aggregator={agg}")
        exp = run_one_model(
            best_model,
            ModelClass,
            clients,
            profiles,
            aggregator=agg,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            malicious_frac=args.malicious_frac,
            seed=args.seed,
            attack_strategy=args.attack_strategy,
            attack_scale=args.attack_scale,
            similarity_tau=tau_star,
            similarity_mode=args.similarity_mode,
            lambda_jaccard=args.lambda_jaccard,
            tau_cos_min=args.tau_cos_min,
            lvp_alpha=alpha_star if agg == "lvp" else None,
            lvp_self_weight=float(np.clip(args.lvp_self_weight, 0.0, 1.0)) if agg == "lvp" else 0.0,
            krum_f=-1,
            local_fit_maxiter=args.local_fit_maxiter,
            eval_fit_maxiter=args.eval_fit_maxiter,
            strict_errors=True,
            network_eval_mode=args.network_eval_mode,
        )
        results.append(exp)

    title_suffix = f" (partition={args.column_partition}, mal={args.malicious_frac:.0%}, attack={args.attack_strategy}, scale={args.attack_scale})"
    fig_path = out_dir / "fig3_decentralized_methods_network_mae.png"
    _plot_decentralized_comparison(results, best_model, fig_path, title_suffix, 
                                    use_log_scale=args.use_log_scale,
                                    clip_outliers_sigma=args.clip_outliers_sigma)

    summary = {
        "best_model": best_model,
        "similarity_tau": tau_star,
        "similarity_mode": args.similarity_mode,
        "lambda_jaccard": float(args.lambda_jaccard),
        "tau_cos_min": float(args.tau_cos_min),
        "lvp_alpha": alpha_star,
        "lvp_self_weight": float(np.clip(args.lvp_self_weight, 0.0, 1.0)),
        "scenario": {
            "column_partition": args.column_partition,
            "malicious_frac": args.malicious_frac,
            "attack_strategy": args.attack_strategy,
            "attack_scale": args.attack_scale,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "seed": args.seed,
            "network_eval_mode": args.network_eval_mode,
        },
        "final_network_mae_by_aggregator": {
            r["aggregator"]: _final_mae(r) for r in results
        },
        "results": [
            {
                "model": r["model"],
                "aggregator": r["aggregator"],
                "history": r["history"],
                "attack_strategy": r["attack_strategy"],
                "attack_scale": r["attack_scale"],
                "malicious_frac": r["malicious_frac"],
            }
            for r in results
        ],
        "figure": str(fig_path),
    }

    json_path = out_dir / "fig3_decentralized_methods.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    report_path = out_dir / "fig3_decentralized_methods_report.md"
    ordered = sorted(summary["final_network_mae_by_aggregator"].items(), key=lambda kv: kv[1])
    lines = [
        "# Figure 3 Decentralized Comparison",
        "",
        f"Best model: {best_model}",
        f"Scenario: {args.column_partition}, mal={args.malicious_frac:.2f}, attack={args.attack_strategy}, scale={args.attack_scale}",
        "",
        "## Final MAE by method",
    ]
    for name, value in ordered:
        lines.append(f"- {name}: {value:.6f}")
    lines.extend([
        "",
        "## Notes",
        "- LVP is the reference decentralized method.",
        "- Decentralized FedAvg, DeFTA, BALANCE, and Push-Sum use the same client graph as LVP.",
    ])
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {fig_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()