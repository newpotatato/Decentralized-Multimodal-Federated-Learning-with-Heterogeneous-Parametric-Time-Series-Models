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


def _parse_alpha_grid(s: str) -> List[float]:
    vals = []
    for x in s.split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    if not vals:
        raise ValueError("alpha grid is empty")
    return vals


def main() -> None:
    p = argparse.ArgumentParser(description="Plot LVP coherence by alpha for current scenario")
    p.add_argument("--alpha-grid", type=str, default="0.15,0.25,0.35,0.45,0.53,0.60")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    pkg = _FL / "artifacts" / "article_package_current_run_20260410"
    seed_cfg_path = pkg / "raw" / "seeds" / "seed42" / "fig3_decentralized_methods.json"
    if not seed_cfg_path.exists():
        raise FileNotFoundError(f"Missing scenario config: {seed_cfg_path}")

    cfg = json.loads(seed_cfg_path.read_text(encoding="utf-8"))
    scenario = cfg.get("scenario", {})

    alpha_grid = _parse_alpha_grid(args.alpha_grid)
    tau = float(cfg.get("similarity_tau", 0.35))
    self_weight = float(cfg.get("lvp_self_weight", 0.0))

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

    out_rows: List[Dict[str, Any]] = []
    curves: Dict[float, Dict[str, np.ndarray]] = {}

    for alpha in alpha_grid:
        exp = run_one_model(
            "DynamicLinearModel",
            ModelClass,
            clients,
            profiles,
            aggregator="lvp",
            rounds=int(scenario.get("rounds", 10)),
            local_epochs=int(scenario.get("local_epochs", 1)),
            malicious_frac=float(scenario.get("malicious_frac", 0.25)),
            seed=int(args.seed),
            attack_strategy=str(scenario.get("attack_strategy", "noise_colluded")),
            attack_scale=float(scenario.get("attack_scale", 5.0)),
            similarity_tau=tau,
            similarity_mode=str(cfg.get("similarity_mode", "jaccard")),
            lambda_jaccard=float(cfg.get("lambda_jaccard", 0.5)),
            tau_cos_min=float(cfg.get("tau_cos_min", -1.0)),
            lvp_alpha=float(alpha),
            lvp_self_weight=self_weight,
            strict_errors=True,
            network_eval_mode=str(scenario.get("network_eval_mode", "proxy")),
            num_workers=1,
        )
        hist = exp.get("history") or []
        rounds = np.asarray([int(h.get("round", 0)) for h in hist], dtype=float)
        coherence = np.asarray([float(h.get("sync_component_l2_mean", 0.0)) for h in hist], dtype=float)
        curves[float(alpha)] = {
            "rounds": rounds,
            "coherence": coherence,
        }
        final_mae = float(hist[-1].get("network_mae", float("inf"))) if hist else float("inf")
        out_rows.append({
            "alpha": float(alpha),
            "final_network_mae": final_mae,
            "rounds": int(len(hist)),
        })

    # Plot
    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, alpha in enumerate(alpha_grid):
        rec = curves[float(alpha)]
        x = rec["rounds"]
        y = rec["coherence"]
        color = colors[idx % len(colors)] if colors else None
        ax.plot(x, y, marker="o", linewidth=1.9, markersize=4, color=color, label=f"alpha={alpha:.2f}")

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Coherence (mean component DeltaL2)")
    ax.set_title("LVP coherence by round for different alpha (fixed scenario)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()

    out_dir = pkg / "plots" / "coherence"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_path = out_dir / "lvp_coherence_by_alpha_current_scenario.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "scenario": {
            "seed": int(args.seed),
            "rounds": int(scenario.get("rounds", 10)),
            "local_epochs": int(scenario.get("local_epochs", 1)),
            "malicious_frac": float(scenario.get("malicious_frac", 0.25)),
            "attack_strategy": str(scenario.get("attack_strategy", "noise_colluded")),
            "attack_scale": float(scenario.get("attack_scale", 5.0)),
            "network_eval_mode": str(scenario.get("network_eval_mode", "proxy")),
            "similarity_tau": tau,
            "similarity_mode": str(cfg.get("similarity_mode", "jaccard")),
            "lambda_jaccard": float(cfg.get("lambda_jaccard", 0.5)),
            "tau_cos_min": float(cfg.get("tau_cos_min", -1.0)),
            "lvp_self_weight": self_weight,
        },
        "alpha_grid": [float(a) for a in alpha_grid],
        "rows": out_rows,
        "figure": str(fig_path),
    }

    summary_path = out_dir / "lvp_coherence_by_alpha_current_scenario_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report_path = out_dir / "lvp_coherence_by_alpha_current_scenario_report.md"
    lines = [
        "# LVP Coherence by Alpha (Current Scenario)",
        "",
        f"Seed: {args.seed}",
        f"Fixed tau: {tau:.2f}",
        f"Fixed self_weight: {self_weight:.2f}",
        "",
        "## Final network MAE by alpha",
    ]
    for row in sorted(out_rows, key=lambda r: r["alpha"]):
        lines.append(f"- alpha={row['alpha']:.2f}: final MAE={row['final_network_mae']:.6f}")
    lines += ["", f"Figure: {fig_path}"]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {fig_path}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
