#!/usr/bin/env python3
"""
Full FL comparison after LVP ablation:
  1) Network MAE vs round — all forecasting models (LVP + best tau/alpha from ablation).
  2) Mean local MAE vs round (+ std across clients) — best model, LVP.
  3) Network MAE vs round — all aggregators (same best model); LVP highlighted.

Outputs: federated_learning/artifacts/full_comparison/ by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "core"))
sys.path.insert(0, str(_FL / "data_loaders"))

from data_utils import (
    build_client_information_profiles,
    build_clients_from_mcc,
    load_mcc_series,
)
from run_real_experiments import (
    MODEL_REGISTRY,
    _build_exogenous,
    run_one_model,
)

DEFAULT_ABLATION_SUMMARY = _FL / "artifacts" / "ablation_lvp" / "ablation_summary.json"
AGGREGATORS_COMPARE = [
    "fedavg",
    "lvp",
    "krum",
    "weighted_median",
    "scaffold",
]

def load_best_hparams(
    summary_path: Path,
) -> Tuple[str, float, Optional[float]]:
    """
    Returns (model_name, tau, lvp_alpha).
    lvp_alpha None means auto heuristic (pass None to run_one_model).
    """
    model = "DynamicLinearModel"
    tau = 0.35
    alpha: Optional[float] = 0.53

    if summary_path.exists():
        try:
            meta = json.loads(summary_path.read_text(encoding="utf-8"))
            row = meta.get("best_network_mae_alpha_sweep") or meta.get(
                "best_network_mae_tau_sweep"
            )
            if isinstance(row, dict):
                model = str(row.get("model", model))
                tau = float(row.get("tau", row.get("fixed_tau", tau)))
                eff = row.get("lvp_alpha_effective")
                req = row.get("alpha_requested")
                if req is not None and not (isinstance(req, float) and np.isnan(req)):
                    try:
                        alpha = float(req)
                    except (TypeError, ValueError):
                        alpha = float(eff) if eff is not None else None
                elif eff is not None:
                    try:
                        alpha = float(eff)
                    except (TypeError, ValueError):
                        alpha = None
                else:
                    alpha = None
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError):
            pass

    return model, tau, alpha


def _mean_local_mae(round_row: Dict) -> Tuple[float, float]:
    vals = list((round_row.get("client_mae") or {}).values())
    if not vals:
        return float("nan"), float("nan")
    arr = np.asarray(vals, dtype=float)
    # Some clients can get inf/nan local MAE after unstable aggregation + refit
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def _network_mae_series(exp: Dict) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    for r in exp["history"]:
        xs.append(int(r["round"]))
        ys.append(float(r["network_mae"]))
    return xs, ys


def _local_mae_series(exp: Dict) -> Tuple[List[int], List[float], List[float]]:
    xs: List[int] = []
    means: List[float] = []
    stds: List[float] = []
    for r in exp["history"]:
        m, s = _mean_local_mae(r)
        xs.append(int(r["round"]))
        means.append(m)
        stds.append(s)
    return xs, means, stds


def run_suite(
    *,
    base_path: Path,
    clients: List,
    profiles: List,
    model_names: List[str],
    aggregators: List[str],
    rounds: int,
    local_epochs: int,
    malicious_frac: float,
    seed: int,
    similarity_tau: float,
    lvp_alpha: Optional[float],
    local_fit_maxiter: int,
    eval_fit_maxiter: int,
    attack_strategy: str = "label_flip",
    attack_scale: float = 2.5,
    network_eval_mode: str = "refit",
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    tag = 0
    for mname in model_names:
        for agg in aggregators:
            tag += 1
            ModelClass = MODEL_REGISTRY[mname]
            print(
                f"[{tag}] model={mname} aggregator={agg} "
                f"tau={similarity_tau:.4g} lvp_alpha={lvp_alpha!r}"
            )
            exp = run_one_model(
                mname,
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
                network_eval_mode=network_eval_mode,
            )
            exp["suite_tag"] = f"{mname}_{agg}"
            results.append(exp)
    return results


def plot_models_network_mae(
    results: List[Dict], model_names: List[str], out_path: Path, title_suffix: str
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    subset = [r for r in results if r["aggregator"] == "lvp" and r["model"] in model_names]
    for exp in sorted(subset, key=lambda x: x["model"]):
        xs, ys = _network_mae_series(exp)
        ax.plot(xs, ys, "o-", linewidth=2, markersize=5, label=exp["model"])
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Network MAE (absolute, amt units)")
    ax.set_title(f"Network MAE vs round — by model (LVP){title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_local_mae_agents(
    exp: Dict, out_path: Path, title_suffix: str
) -> None:
    import matplotlib.pyplot as plt

    xs, means, stds = _local_mae_series(exp)
    fig, ax1 = plt.subplots(figsize=(9, 5))
    means_a = np.asarray(means, dtype=float)
    stds_a = np.asarray(stds, dtype=float)
    ax1.plot(
        xs,
        means_a,
        "s-",
        color="C0",
        linewidth=2,
        markersize=6,
        label="Mean local MAE (nan-mean over clients)",
    )
    ax1.fill_between(
        xs,
        means_a - stds_a,
        means_a + stds_a,
        color="C0",
        alpha=0.2,
        label="±1 std (nan-std)",
    )
    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Local MAE (amt units)", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = None
    finite = np.isfinite(means_a)
    if np.any(finite):
        base = float(means_a[finite][0])
        if base and abs(base) > 1e-9:
            ax2 = ax1.twinx()
            delta_pct = (means_a / base - 1.0) * 100.0
            ax2.plot(
                xs,
                delta_pct,
                "^--",
                color="C3",
                linewidth=1.5,
                markersize=4,
                alpha=0.85,
                label="Change vs round 1 (%)",
            )
            ax2.set_ylabel("Relative to round 1 (%)", color="C3")
            ax2.tick_params(axis="y", labelcolor="C3")
            ax2.axhline(0, color="C3", linestyle=":", alpha=0.5)

    ax1.set_title(
        f"Local error dynamics — {exp['model']} + LVP{title_suffix}"
    )
    ax1.grid(True, alpha=0.3)
    h1, l1 = ax1.get_legend_handles_labels()
    if ax2 is not None:
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, l1 + l2, loc="best")
    else:
        ax1.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_aggregators_network_mae(
    results: List[Dict],
    model_name: str,
    out_path: Path,
    title_suffix: str,
) -> None:
    """Single panel: aggregation methods only (best model from ablation)."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    subset = [r for r in results if r["model"] == model_name]
    order = {a: i for i, a in enumerate(AGGREGATORS_COMPARE)}
    subset.sort(key=lambda x: order.get(x["aggregator"], 99))

    for exp in subset:
        agg = exp["aggregator"]
        label = "LVP (proposed)" if agg == "lvp" else agg.replace("_", " ").title()
        xs, ys = _network_mae_series(exp)
        lw = 2.8 if agg == "lvp" else 1.6
        ax.plot(xs, ys, "o-", linewidth=lw, markersize=5, label=label)
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Network MAE (absolute, amt units)")
    ax.set_title(
        f"Network MAE vs round — aggregators ({model_name}){title_suffix}"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Full LVP vs baselines comparison (3 plots).")
    p.add_argument("--base-path", type=str, default=str(_FL.parent))
    p.add_argument(
        "--ablation-summary",
        type=str,
        default=str(DEFAULT_ABLATION_SUMMARY),
        help="JSON with best_network_mae_alpha_sweep (tau, alpha, model).",
    )
    p.add_argument("--out-dir", type=str, default=str(_FL / "artifacts" / "full_comparison"))
    p.add_argument("--n-clients", type=int, default=8)
    p.add_argument("--rounds", type=int, default=20, help="Rounds for dynamics (>=10 recommended).")
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--malicious-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-reuters", action="store_true")
    p.add_argument(
        "--include-markov",
        action="store_true",
        help="Include MarkovSwitchingRegressionModel (slow).",
    )
    p.add_argument(
        "--include-structural",
        action="store_true",
        help="Include StructuralTimeSeriesModel (params mix strings; LVP may fail).",
    )
    p.add_argument("--local-fit-maxiter", type=int, default=10)
    p.add_argument("--eval-fit-maxiter", type=int, default=0)
    p.add_argument(
        "--attack-strategy",
        type=str,
        default="label_flip",
        choices=["label_flip", "noise", "noise_colluded", "random"],
    )
    p.add_argument("--attack-scale", type=float, default=2.5)
    p.add_argument(
        "--network-eval-mode",
        type=str,
        default="refit",
        choices=["proxy", "refit"],
    )
    args = p.parse_args()

    base = Path(args.base_path).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_model, tau_star, alpha_star = load_best_hparams(Path(args.ablation_summary))
    model_list: List[str] = []
    for m in MODEL_REGISTRY.keys():
        if m == "MarkovSwitchingRegressionModel" and not args.include_markov:
            continue
        if m == "StructuralTimeSeriesModel" and not args.include_structural:
            continue
        model_list.append(m)

    title_suffix = f" (tau={tau_star:.3g}, alpha={alpha_star!s})"

    print("Loading MCC clients...")
    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=not args.no_reuters)
    clients = build_clients_from_mcc(mcc_df, exog, n_clients=args.n_clients)
    if len(clients) < 2:
        raise RuntimeError("Need at least 2 clients.")
    profiles = build_client_information_profiles(clients, "mcc")

    # Same topic overlay as ablation (pairwise Jaccard varies with tau)
    profiles = [
        frozenset(p) | {f"sync_topic_{idx % 4}"}
        for idx, p in enumerate(profiles)
    ]
    print(f"Best from ablation: model={best_model}, tau={tau_star}, lvp_alpha={alpha_star}")

    all_results: List[Dict[str, Any]] = []

    # 1) All models, LVP only
    print("--- Suite: models (LVP) ---")
    all_results.extend(
        run_suite(
            base_path=base,
            clients=clients,
            profiles=profiles,
            model_names=model_list,
            aggregators=["lvp"],
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            malicious_frac=args.malicious_frac,
            seed=args.seed,
            similarity_tau=tau_star,
            lvp_alpha=alpha_star,
            local_fit_maxiter=args.local_fit_maxiter,
            eval_fit_maxiter=args.eval_fit_maxiter,
            attack_strategy=args.attack_strategy,
            attack_scale=args.attack_scale,
            network_eval_mode=args.network_eval_mode,
        )
    )

    # 2) Best model only × all aggregators (LVP uses tau/alpha from ablation)
    print("--- Suite: aggregators x best model only ---")
    all_results.extend(
        run_suite(
            base_path=base,
            clients=clients,
            profiles=profiles,
            model_names=[best_model],
            aggregators=AGGREGATORS_COMPARE,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            malicious_frac=args.malicious_frac,
            seed=args.seed,
            similarity_tau=tau_star,
            lvp_alpha=alpha_star,
            local_fit_maxiter=args.local_fit_maxiter,
            eval_fit_maxiter=args.eval_fit_maxiter,
            attack_strategy=args.attack_strategy,
            attack_scale=args.attack_scale,
            network_eval_mode=args.network_eval_mode,
        )
    )

    # Dedupe by (model, aggregator): second suite overlaps LVP from suite 1
    seen = set()
    unique: List[Dict[str, Any]] = []
    for r in all_results:
        key = (r["model"], r["aggregator"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    all_results = unique

    json_path = out_dir / "full_comparison_raw.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_model,
                "similarity_tau": tau_star,
                "lvp_alpha": alpha_star,
                "rounds": args.rounds,
                "malicious_frac": args.malicious_frac,
                "results": [
                    {
                        "model": r["model"],
                        "aggregator": r["aggregator"],
                        "rounds": r["rounds"],
                        "history": r["history"],
                    }
                    for r in all_results
                ],
            },
            f,
            indent=2,
            default=str,
        )
    print(f"Saved: {json_path}")

    plot_models_network_mae(
        all_results,
        model_list,
        out_dir / "fig1_models_network_mae.png",
        title_suffix,
    )

    lvp_best = next(
        (
            r
            for r in all_results
            if r["model"] == best_model and r["aggregator"] == "lvp"
        ),
        None,
    )
    if lvp_best is None:
        raise RuntimeError("Missing LVP run for best model.")
    plot_local_mae_agents(
        lvp_best,
        out_dir / "fig2_local_mae_agents.png",
        title_suffix,
    )

    plot_aggregators_network_mae(
        all_results,
        best_model,
        out_dir / "fig3_aggregators_network_mae.png",
        title_suffix,
    )

    meta = {
        "best_model": best_model,
        "similarity_tau": tau_star,
        "lvp_alpha": alpha_star,
        "models_plotted": model_list,
        "aggregators": AGGREGATORS_COMPARE,
        "rounds": args.rounds,
        "notes": {
            "fig3": (
                "Single panel: best model from ablation; lines = FedAvg, LVP (tau,alpha from ablation), "
                "Krum, weighted median, SCAFFOLD."
            ),
            "nan_mitigation": (
                "Numeric params stabilized after each round (sigma clipped, nan/inf cleaned); "
                "MAE uses finite errors only."
            ),
            "fig2_axes": (
                "Left: mean local MAE (nan-mean); right: % change vs round 1."
            ),
        },
        "figures": [
            str(out_dir / "fig1_models_network_mae.png"),
            str(out_dir / "fig2_local_mae_agents.png"),
            str(out_dir / "fig3_aggregators_network_mae.png"),
        ],
    }
    with open(out_dir / "full_comparison_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print("Done.")


if __name__ == "__main__":
    main()
