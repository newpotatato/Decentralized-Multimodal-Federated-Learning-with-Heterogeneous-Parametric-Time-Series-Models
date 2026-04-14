#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

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


def _parse_grid(s: str) -> List[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("alpha grid is empty")
    return vals


def _resolve_pkg(default_pkg: str) -> Path:
    p = Path(default_pkg)
    if p.exists():
        return p
    alt = _FL / "artifacts" / "article_package_current_run_20260410"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Package directory not found: {default_pkg}")


def _plot_component_curves(
    out_dir: Path,
    comp_idx: int,
    rounds: np.ndarray,
    alpha_series: Dict[float, np.ndarray],
    comp_size: int,
    with_size_in_title: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for i, alpha in enumerate(sorted(alpha_series.keys())):
        y = alpha_series[alpha]
        color = colors[i % len(colors)] if colors else None
        ax.plot(rounds, y, marker="o", linewidth=1.9, markersize=4, color=color, label=f"alpha={alpha:.2f}")

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Component DeltaL2")
    if with_size_in_title:
        ax.set_title(f"Component {comp_idx + 1} (nodes={comp_size}) - coherence by alpha")
    else:
        ax.set_title(f"Component {comp_idx + 1} - coherence by alpha")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / f"component_{comp_idx + 1:02d}.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot per-component LVP coherence curves for alpha sweep (2 title variants)")
    p.add_argument(
        "--pkg-dir",
        type=str,
        default=str(_FL / "artifacts" / "article_package_current_run_20260410" / "experiment_with_more_rounds"),
    )
    p.add_argument("--alpha-grid", type=str, default="0.15,0.25,0.35,0.45,0.53,0.60")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rounds-override", type=int, default=0, help="If >0, use this rounds count instead of scenario rounds")
    p.add_argument("--local-fit-maxiter", type=int, default=4)
    p.add_argument("--eval-fit-maxiter", type=int, default=0)
    args = p.parse_args()

    pkg = _resolve_pkg(args.pkg_dir)
    cfg_path = pkg / "raw" / "seeds" / "seed42" / "fig3_decentralized_methods.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing scenario config: {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    scenario = cfg.get("scenario", {})

    alpha_grid = _parse_grid(args.alpha_grid)
    tau = float(cfg.get("similarity_tau", 0.35))
    similarity_mode = str(cfg.get("similarity_mode", "jaccard"))
    lambda_jaccard = float(cfg.get("lambda_jaccard", 0.5))
    tau_cos_min = float(cfg.get("tau_cos_min", -1.0))
    self_weight = float(cfg.get("lvp_self_weight", 0.0))
    rounds_to_run = int(args.rounds_override) if int(args.rounds_override) > 0 else int(scenario.get("rounds", 10))

    base = _FL.parent
    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=True)
    clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=20,
        column_partition=str(scenario.get("column_partition", "contiguous")),
    )
    profiles = _topic_profiles(clients)
    ModelClass = MODEL_REGISTRY["DynamicLinearModel"]

    # alpha -> rounds, component deltas matrix (rounds x n_components)
    runs: Dict[float, Dict[str, Any]] = {}
    component_sizes_ref: List[int] = []

    for alpha in alpha_grid:
        exp = run_one_model(
            "DynamicLinearModel",
            ModelClass,
            clients,
            profiles,
            aggregator="lvp",
            rounds=rounds_to_run,
            local_epochs=int(scenario.get("local_epochs", 1)),
            malicious_frac=float(scenario.get("malicious_frac", 0.25)),
            seed=int(args.seed),
            attack_strategy=str(scenario.get("attack_strategy", "noise_colluded")),
            attack_scale=float(scenario.get("attack_scale", 5.0)),
            similarity_tau=tau,
            similarity_mode=similarity_mode,
            lambda_jaccard=lambda_jaccard,
            tau_cos_min=tau_cos_min,
            lvp_alpha=float(alpha),
            lvp_self_weight=self_weight,
            strict_errors=True,
            network_eval_mode=str(scenario.get("network_eval_mode", "proxy")),
            num_workers=1,
            local_fit_maxiter=int(args.local_fit_maxiter),
            eval_fit_maxiter=int(args.eval_fit_maxiter),
        )

        hist = exp.get("history") or []
        rounds = np.asarray([int(h.get("round", 0)) for h in hist], dtype=float)

        if not hist:
            continue

        # Determine stable component count from first round in current run.
        if not component_sizes_ref:
            component_sizes_ref = [int(x) for x in (hist[0].get("sync_component_sizes") or [])]

        n_comp = len(component_sizes_ref)
        mat = np.full((len(hist), n_comp), np.nan, dtype=float)
        for ridx, row in enumerate(hist):
            vals = row.get("sync_component_l2_by_component") or []
            for cidx in range(min(len(vals), n_comp)):
                mat[ridx, cidx] = float(vals[cidx])

        runs[float(alpha)] = {
            "rounds": rounds,
            "component_matrix": mat,
            "final_mae": float(hist[-1].get("network_mae", float("inf"))),
        }

    if not runs:
        raise RuntimeError("No alpha runs produced data")

    n_comp = len(component_sizes_ref)
    rounds_ref = next(iter(runs.values()))["rounds"]

    with_size_dir = pkg / "plots" / "coherence" / "components" / "with_size_in_title"
    without_size_dir = pkg / "plots" / "coherence" / "components" / "without_size_in_title"

    for cidx in range(n_comp):
        alpha_series: Dict[float, np.ndarray] = {}
        for alpha, rec in runs.items():
            alpha_series[alpha] = rec["component_matrix"][:, cidx]

        _plot_component_curves(
            with_size_dir,
            cidx,
            rounds_ref,
            alpha_series,
            int(component_sizes_ref[cidx]),
            with_size_in_title=True,
        )
        _plot_component_curves(
            without_size_dir,
            cidx,
            rounds_ref,
            alpha_series,
            int(component_sizes_ref[cidx]),
            with_size_in_title=False,
        )

    summary = {
        "pkg": str(pkg),
        "seed": int(args.seed),
        "alpha_grid": [float(a) for a in alpha_grid],
        "rounds_used": int(rounds_to_run),
        "component_sizes": component_sizes_ref,
        "n_components": int(n_comp),
        "output_dirs": {
            "with_size_in_title": str(with_size_dir),
            "without_size_in_title": str(without_size_dir),
        },
        "final_mae_by_alpha": {str(a): runs[a]["final_mae"] for a in sorted(runs.keys())},
    }

    summary_path = pkg / "plots" / "coherence" / "components" / "component_alpha_curves_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_lines = [
        "# Component-wise coherence by alpha",
        "",
        f"Seed: {args.seed}",
        f"Alphas: {', '.join(f'{a:.2f}' for a in sorted(runs.keys()))}",
        f"Components: {n_comp}",
        f"Component sizes: {component_sizes_ref}",
        "",
        f"With-size variant: {with_size_dir}",
        f"Without-size variant: {without_size_dir}",
    ]
    report_path = pkg / "plots" / "coherence" / "components" / "component_alpha_curves_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved: {summary_path}")
    print(f"Saved: {report_path}")
    print(f"Saved plots in: {with_size_dir}")
    print(f"Saved plots in: {without_size_dir}")


if __name__ == "__main__":
    main()
