#!/usr/bin/env python3
"""
One-seed article pipeline in the exact requested order:

1) Run LVP only on all 5 main models.
2) Plot network MAE dynamics for these models.
3) Pick the best model by final network MAE (last round).
4) Plot local-agent error dynamics for the selected model.
5) Run aggregator comparison only for this model and plot it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "core"))
sys.path.insert(0, str(_FL / "data_loaders"))

from data_utils import build_client_information_profiles, build_clients_from_mcc, load_mcc_series
from full_comparison_lvp import (
    AGGREGATORS_COMPARE,
    DEFAULT_ABLATION_SUMMARY,
    load_best_hparams,
    plot_aggregators_network_mae,
    plot_local_mae_agents,
    plot_models_network_mae,
)
from run_real_experiments import MODEL_REGISTRY, _build_exogenous, run_one_model


MAIN_MODELS: List[str] = [
    "ARMAXModel",
    "DynamicLinearModel",
    "KalmanFilterModel",
    "StructuralTimeSeriesModel",
    "MarkovSwitchingRegressionModel",
]


def _topic_profiles(clients: List, n_groups: int = 4) -> List:
    profiles = build_client_information_profiles(clients, "mcc")
    return [
        frozenset(p) | {f"sync_topic_{idx % n_groups}"}
        for idx, p in enumerate(profiles)
    ]


def _final_network_mae(exp: Dict[str, Any]) -> float:
    hist = exp.get("history") or []
    if not hist:
        return float("inf")
    val = hist[-1].get("network_mae")
    try:
        f = float(val)
    except (TypeError, ValueError):
        return float("inf")
    if not math.isfinite(f):
        return float("inf")
    return f


def _run_suite(
    *,
    clients: List,
    profiles: List,
    model_names: List[str],
    aggregators: List[str],
    rounds: int,
    local_epochs: int,
    malicious_frac: float,
    seed: int,
    attack_strategy: str,
    attack_scale: float,
    similarity_tau: float,
    lvp_alpha: Optional[float],
    local_fit_maxiter: int,
    eval_fit_maxiter: int,
    network_eval_mode: str,
    random_init: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    total = len(model_names) * len(aggregators)
    tag = 0
    for model_name in model_names:
        ModelClass = MODEL_REGISTRY[model_name]
        for agg in aggregators:
            tag += 1
            print(
                f"[{tag}/{total}] model={model_name} agg={agg} "
                f"mal={malicious_frac} attack={attack_strategy} scale={attack_scale}"
            )
            exp = run_one_model(
                model_name,
                ModelClass,
                clients,
                profiles,
                aggregator=agg,
                rounds=rounds,
                local_epochs=local_epochs,
                malicious_frac=malicious_frac,
                seed=seed,
                attack_strategy=attack_strategy,
                attack_scale=attack_scale,
                similarity_tau=similarity_tau,
                lvp_alpha=lvp_alpha if agg == "lvp" else None,
                krum_f=-1,
                local_fit_maxiter=local_fit_maxiter,
                eval_fit_maxiter=eval_fit_maxiter,
                strict_errors=True,
                network_eval_mode=network_eval_mode,
                random_init=random_init,
            )
            out.append(exp)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="One-seed article pipeline (LVP -> best model -> aggregators).")
    p.add_argument("--base-path", type=str, default=str(_FL.parent))
    p.add_argument("--ablation-summary", type=str, default=str(DEFAULT_ABLATION_SUMMARY))
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_FL / "artifacts" / "article_pipeline_one_seed"),
    )
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--malicious-frac", type=float, default=0.2)
    p.add_argument("--attack-strategy", type=str, default="label_flip", choices=["label_flip", "noise", "noise_colluded", "random"])
    p.add_argument("--attack-scale", type=float, default=2.5)
    p.add_argument("--similarity-tau", type=float, default=None, help="Override LVP Jaccard threshold tau.")
    p.add_argument(
        "--similarity-mode",
        type=str,
        default=None,
        choices=["jaccard", "jaccard_cosine_hybrid", "hybrid"],
        help="Override LVP similarity mode.",
    )
    p.add_argument("--lambda-jaccard", type=float, default=None, help="Override hybrid mix coefficient lambda.")
    p.add_argument("--tau-cos-min", type=float, default=None, help="Override hybrid cosine gate.")
    p.add_argument("--lvp-alpha", type=float, default=None, help="Override LVP alpha.")
    p.add_argument("--network-eval-mode", type=str, default="proxy", choices=["proxy", "refit"])
    p.add_argument("--local-fit-maxiter", type=int, default=10)
    p.add_argument("--eval-fit-maxiter", type=int, default=0)
    p.add_argument("--no-reuters", action="store_true")
    p.add_argument("--random-init", action="store_true", help="Initialize all models with random parameters before training")
    args = p.parse_args()

    missing = [m for m in MAIN_MODELS if m not in MODEL_REGISTRY]
    if missing:
        raise RuntimeError(f"Missing models in MODEL_REGISTRY: {missing}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _, tau_star, alpha_star = load_best_hparams(Path(args.ablation_summary))
    if args.similarity_tau is not None:
        tau_star = float(args.similarity_tau)
    if args.lvp_alpha is not None:
        alpha_star = float(args.lvp_alpha)
    title_suffix = f" (tau={tau_star:.3g}, alpha={alpha_star!s}, eval={args.network_eval_mode})"

    base = Path(args.base_path).resolve()
    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=not args.no_reuters)
    clients = build_clients_from_mcc(mcc_df, exog, n_clients=args.n_clients)
    if len(clients) < 2:
        raise RuntimeError("Need at least 2 clients.")
    profiles = _topic_profiles(clients)

    print("=== Stage 1: LVP on 5 models ===")
    lvp_results = _run_suite(
        clients=clients,
        profiles=profiles,
        model_names=MAIN_MODELS,
        aggregators=["lvp"],
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        malicious_frac=args.malicious_frac,
        seed=args.seed,
        attack_strategy=args.attack_strategy,
        attack_scale=args.attack_scale,
        similarity_tau=tau_star,
        # Use the selected similarity mode and hybrid knobs if provided.
        lvp_alpha=alpha_star,
        local_fit_maxiter=args.local_fit_maxiter,
        eval_fit_maxiter=args.eval_fit_maxiter,
        network_eval_mode=args.network_eval_mode,
        random_init=args.random_init,
    )

    plot_models_network_mae(
        lvp_results,
        MAIN_MODELS,
        out_dir / "fig1_models_network_mae_lvp_only.png",
        title_suffix,
    )

    scored: List[Tuple[str, float]] = [
        (r["model"], _final_network_mae(r)) for r in lvp_results
    ]
    scored.sort(key=lambda x: x[1])
    best_model, best_final = scored[0]
    if not math.isfinite(best_final):
        raise RuntimeError("Could not determine best model: all final network MAE are non-finite.")
    print(f"Best model by final network MAE: {best_model} ({best_final:.6f})")

    best_lvp = next(
        r for r in lvp_results if r["model"] == best_model and r["aggregator"] == "lvp"
    )
    plot_local_mae_agents(
        best_lvp,
        out_dir / "fig2_local_mae_agents_best_lvp_model.png",
        title_suffix,
    )

    print("=== Stage 2: Aggregators on best model ===")
    agg_results = _run_suite(
        clients=clients,
        profiles=profiles,
        model_names=[best_model],
        aggregators=AGGREGATORS_COMPARE,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        malicious_frac=args.malicious_frac,
        seed=args.seed,
        attack_strategy=args.attack_strategy,
        attack_scale=args.attack_scale,
        similarity_tau=tau_star,
        lvp_alpha=alpha_star,
        local_fit_maxiter=args.local_fit_maxiter,
        eval_fit_maxiter=args.eval_fit_maxiter,
        network_eval_mode=args.network_eval_mode,
        random_init=args.random_init,
    )

    plot_aggregators_network_mae(
        agg_results,
        best_model,
        out_dir / "fig3_aggregators_network_mae_best_model.png",
        title_suffix,
    )

    payload = {
        "config": {
            "n_clients": args.n_clients,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "seed": args.seed,
            "malicious_frac": args.malicious_frac,
            "attack_strategy": args.attack_strategy,
            "attack_scale": args.attack_scale,
            "network_eval_mode": args.network_eval_mode,
            "use_reuters": not args.no_reuters,
            "random_init": bool(args.random_init),
            "similarity_tau": tau_star,
            "lvp_alpha": alpha_star,
        },
        "best_model_selection": {
            "criterion": "minimum final network_mae at last round among 5 LVP runs",
            "scores": [{"model": m, "final_network_mae": v} for (m, v) in scored],
            "best_model": best_model,
            "best_final_network_mae": best_final,
        },
        "lvp_results": [
            {
                "model": r["model"],
                "aggregator": r["aggregator"],
                "history": r["history"],
            }
            for r in lvp_results
        ],
        "aggregator_results": [
            {
                "model": r["model"],
                "aggregator": r["aggregator"],
                "history": r["history"],
            }
            for r in agg_results
        ],
        "figures": [
            str(out_dir / "fig1_models_network_mae_lvp_only.png"),
            str(out_dir / "fig2_local_mae_agents_best_lvp_model.png"),
            str(out_dir / "fig3_aggregators_network_mae_best_model.png"),
        ],
    }
    out_json = out_dir / "article_pipeline_one_seed_results.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Saved: {out_json}")
    print("Done.")


if __name__ == "__main__":
    main()
