#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "core"))
sys.path.insert(0, str(_FL / "data_loaders"))

from data_utils import build_client_information_profiles, build_clients_from_mcc, load_mcc_series
from run_real_experiments import MODEL_REGISTRY, _build_exogenous, run_one_model


DECENTRALIZED_AGGS = ["lvp", "decentralized_fedavg", "defta", "balance", "push_sum"]
AGG_LABELS = {
    "lvp": "LVP",
    "decentralized_fedavg": "Decentralized FedAvg",
    "defta": "DeFTA",
    "balance": "BALANCE",
    "push_sum": "Push-Sum",
}

MODEL_LABELS = {
    "DynamicLinearModel": "Dynamic Linear",
    "ARMAXModel": "ARMAX",
    "KalmanFilterModel": "Kalman Filter",
    "StructuralTimeSeriesModel": "Structural TS",
}


def _topic_profiles(clients: List, n_groups: int = 4) -> List:
    profiles = build_client_information_profiles(clients, "mcc")
    return [frozenset(p) | {f"sync_topic_{idx % n_groups}"} for idx, p in enumerate(profiles)]


def _run_cmd(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _extract_model_data(results: List[Dict[str, Any]], model_name: str) -> Tuple[List[int], List[float]]:
    for rec in results:
        if rec.get("model") == model_name:
            hist = rec.get("history") or []
            rounds = [int(h.get("round", 0)) for h in hist]
            mae = [float(h.get("network_mae", 0.0)) for h in hist]
            return rounds, mae
    return [], []


def _plot_models_learning_curves(all_models_json: Path, out_dir: Path, title: str, out_name: str) -> None:
    payload = json.loads(all_models_json.read_text(encoding="utf-8"))
    results = payload.get("results", [])

    model_data: Dict[str, Tuple[List[int], List[float]]] = {}
    for model in sorted(set(r.get("model") for r in results if r.get("model"))):
        rounds, mae = _extract_model_data(results, model)
        if rounds and mae:
            model_data[model] = (rounds, mae)

    if not model_data:
        raise RuntimeError(f"No model data in {all_models_json}")

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, model in enumerate(sorted(model_data.keys())):
        rounds, mae = model_data[model]
        lw = 2.5 if model == "DynamicLinearModel" else 1.7
        color = colors[idx % len(colors)] if colors else None
        ax.plot(rounds, mae, "o-", linewidth=lw, markersize=5, color=color, label=MODEL_LABELS.get(model, model))

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Network MAE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / out_name
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _run_all_models_lvp(
    *,
    out_json: Path,
    rounds: int,
    seed: int,
    n_clients: int,
    column_partition: str,
    malicious_frac: float,
    attack_strategy: str,
    attack_scale: float,
    similarity_tau: float,
    similarity_mode: str,
    lambda_jaccard: float,
    tau_cos_min: float,
    lvp_alpha: float,
    lvp_self_weight: float,
    network_eval_mode: str,
) -> None:
    base_path = _FL.parent
    mcc_df = load_mcc_series(base_path)
    exog = _build_exogenous(base_path, mcc_df, use_reuters=True)
    clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=n_clients,
        column_partition=column_partition,
    )
    profiles = _topic_profiles(clients)

    models = ["DynamicLinearModel", "ARMAXModel", "KalmanFilterModel", "StructuralTimeSeriesModel"]
    all_results: List[Dict[str, Any]] = []

    for model_name in models:
        ModelClass = MODEL_REGISTRY[model_name]
        exp = run_one_model(
            model_name,
            ModelClass,
            clients,
            profiles,
            aggregator="lvp",
            rounds=rounds,
            local_epochs=1,
            malicious_frac=malicious_frac,
            seed=seed,
            attack_strategy=attack_strategy,
            attack_scale=attack_scale,
            similarity_tau=similarity_tau,
            similarity_mode=similarity_mode,
            lambda_jaccard=lambda_jaccard,
            tau_cos_min=tau_cos_min,
            lvp_alpha=lvp_alpha,
            lvp_self_weight=lvp_self_weight,
            strict_errors=True,
            network_eval_mode=network_eval_mode,
            num_workers=1,
        )
        all_results.append(exp)

    summary = {
        "models_run": models,
        "scenario": {
            "column_partition": column_partition,
            "malicious_frac": malicious_frac,
            "attack_strategy": attack_strategy,
            "attack_scale": attack_scale,
            "rounds": rounds,
            "local_epochs": 1,
            "seed": seed,
            "network_eval_mode": network_eval_mode,
        },
        "results": all_results,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _build_lvp_coherence_by_alpha(
    *,
    out_dir: Path,
    alpha_grid: List[float],
    rounds: int,
    seed: int,
    n_clients: int,
    column_partition: str,
    malicious_frac: float,
    attack_strategy: str,
    attack_scale: float,
    similarity_tau: float,
    similarity_mode: str,
    lambda_jaccard: float,
    tau_cos_min: float,
    lvp_self_weight: float,
    network_eval_mode: str,
) -> None:
    base_path = _FL.parent
    mcc_df = load_mcc_series(base_path)
    exog = _build_exogenous(base_path, mcc_df, use_reuters=True)
    clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=n_clients,
        column_partition=column_partition,
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
            rounds=rounds,
            local_epochs=1,
            malicious_frac=malicious_frac,
            seed=seed,
            attack_strategy=attack_strategy,
            attack_scale=attack_scale,
            similarity_tau=similarity_tau,
            similarity_mode=similarity_mode,
            lambda_jaccard=lambda_jaccard,
            tau_cos_min=tau_cos_min,
            lvp_alpha=float(alpha),
            lvp_self_weight=lvp_self_weight,
            strict_errors=True,
            network_eval_mode=network_eval_mode,
            num_workers=1,
        )
        hist = exp.get("history") or []
        rounds_x = np.asarray([int(h.get("round", 0)) for h in hist], dtype=float)
        coherence = np.asarray([float(h.get("sync_component_l2_mean", 0.0)) for h in hist], dtype=float)
        curves[float(alpha)] = {"rounds": rounds_x, "coherence": coherence}
        final_mae = float(hist[-1].get("network_mae", float("inf"))) if hist else float("inf")
        out_rows.append({"alpha": float(alpha), "final_network_mae": final_mae, "rounds": int(len(hist))})

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, alpha in enumerate(alpha_grid):
        rec = curves[float(alpha)]
        color = colors[idx % len(colors)] if colors else None
        ax.plot(rec["rounds"], rec["coherence"], marker="o", linewidth=1.9, markersize=4, color=color, label=f"alpha={alpha:.2f}")

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Coherence (mean component DeltaL2)")
    ax.set_title("LVP coherence by round for different alpha (fixed scenario)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = out_dir / "lvp_coherence_by_alpha_current_scenario.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "scenario": {
            "seed": int(seed),
            "rounds": int(rounds),
            "local_epochs": 1,
            "malicious_frac": float(malicious_frac),
            "attack_strategy": str(attack_strategy),
            "attack_scale": float(attack_scale),
            "network_eval_mode": str(network_eval_mode),
            "similarity_tau": float(similarity_tau),
            "similarity_mode": str(similarity_mode),
            "lambda_jaccard": float(lambda_jaccard),
            "tau_cos_min": float(tau_cos_min),
            "lvp_self_weight": float(lvp_self_weight),
        },
        "alpha_grid": [float(a) for a in alpha_grid],
        "rows": out_rows,
        "figure": str(fig_path),
    }
    (out_dir / "lvp_coherence_by_alpha_current_scenario_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# LVP Coherence by Alpha (Current Scenario)",
        "",
        f"Seed: {seed}",
        f"Fixed tau: {similarity_tau:.2f}",
        f"Fixed self_weight: {lvp_self_weight:.2f}",
        "",
        "## Final network MAE by alpha",
    ]
    for row in sorted(out_rows, key=lambda r: r["alpha"]):
        lines.append(f"- alpha={row['alpha']:.2f}: final MAE={row['final_network_mae']:.6f}")
    lines += ["", f"Figure: {fig_path}"]
    (out_dir / "lvp_coherence_by_alpha_current_scenario_report.md").write_text("\n".join(lines), encoding="utf-8")


def _build_tuning_derived_plots(tuning_summary: Path, ablation_dir: Path) -> None:
    payload = json.loads(tuning_summary.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    if not rows:
        raise RuntimeError(f"No rows in {tuning_summary}")

    alpha_grid = sorted({float(r["alpha"]) for r in rows})
    self_grid = sorted({float(r["self_weight"]) for r in rows})

    mat = np.full((len(self_grid), len(alpha_grid)), np.nan, dtype=float)
    for r in rows:
        ai = alpha_grid.index(float(r["alpha"]))
        si = self_grid.index(float(r["self_weight"]))
        mat[si, ai] = float(r["objective"])

    best = min(rows, key=lambda r: float(r["objective"]))
    best_alpha = float(best["alpha"])
    best_sw = float(best["self_weight"])

    fig1, ax1 = plt.subplots(figsize=(7.4, 4.8))
    im = ax1.imshow(mat, origin="lower", aspect="auto", cmap="viridis")
    ax1.set_xticks(range(len(alpha_grid)))
    ax1.set_xticklabels([f"{x:.2f}" for x in alpha_grid], rotation=30)
    ax1.set_yticks(range(len(self_grid)))
    ax1.set_yticklabels([f"{x:.2f}" for x in self_grid])
    ax1.set_xlabel("lvp_alpha")
    ax1.set_ylabel("lvp_self_weight")
    ax1.set_title("Tuning heatmap: objective")
    ax1.scatter([alpha_grid.index(best_alpha)], [self_grid.index(best_sw)], s=120, marker="*", color="red", edgecolors="white", linewidths=0.8)
    fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    fig1.tight_layout()
    heatmap_png = ablation_dir / "lvp_tuning_left_top_heatmap.png"
    fig1.savefig(heatmap_png, dpi=170, bbox_inches="tight")
    plt.close(fig1)

    sw_to_vals: Dict[float, List[float]] = {}
    for row in rows:
        sw = float(row["self_weight"])
        sw_to_vals.setdefault(sw, []).append(float(row["objective"]))
    sw_x = sorted(sw_to_vals.keys())
    sw_mean = [float(np.mean(sw_to_vals[x])) for x in sw_x]
    sw_std = [float(np.std(sw_to_vals[x], ddof=0)) for x in sw_x]
    best_idx = int(np.argmin(sw_mean))

    fig2, ax2 = plt.subplots(figsize=(7.4, 4.8))
    ax2.plot(sw_x, sw_mean, "o-", linewidth=2.2, markersize=5, label="mean objective")
    ax2.fill_between(sw_x, np.asarray(sw_mean) - np.asarray(sw_std), np.asarray(sw_mean) + np.asarray(sw_std), alpha=0.15, label="+-1 std")
    ax2.scatter([sw_x[best_idx]], [sw_mean[best_idx]], s=100, marker="*", color="red", zorder=5, label=f"best self_weight={sw_x[best_idx]:.2f}")
    ax2.set_xlabel("lvp_self_weight")
    ax2.set_ylabel("Objective (mean + 0.5*std)")
    ax2.set_title("Self-weight sweep (multiseed)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=8)
    fig2.tight_layout()
    sw_png = ablation_dir / "lvp_tuning_left_bottom_self_weight_curve.png"
    fig2.savefig(sw_png, dpi=170, bbox_inches="tight")
    plt.close(fig2)

    fig3, (a1, a2) = plt.subplots(2, 1, figsize=(7.4, 9.0))
    im = a1.imshow(mat, origin="lower", aspect="auto", cmap="viridis")
    a1.set_xticks(range(len(alpha_grid)))
    a1.set_xticklabels([f"{x:.2f}" for x in alpha_grid], rotation=30)
    a1.set_yticks(range(len(self_grid)))
    a1.set_yticklabels([f"{x:.2f}" for x in self_grid])
    a1.set_xlabel("lvp_alpha")
    a1.set_ylabel("lvp_self_weight")
    a1.set_title("Tuning heatmap: objective")
    a1.scatter([alpha_grid.index(best_alpha)], [self_grid.index(best_sw)], s=120, marker="*", color="red", edgecolors="white", linewidths=0.8)
    fig3.colorbar(im, ax=a1, fraction=0.046, pad=0.04)

    a2.plot(sw_x, sw_mean, "o-", linewidth=2.2, markersize=5, label="mean objective")
    a2.fill_between(sw_x, np.asarray(sw_mean) - np.asarray(sw_std), np.asarray(sw_mean) + np.asarray(sw_std), alpha=0.15, label="+-1 std")
    a2.scatter([sw_x[best_idx]], [sw_mean[best_idx]], s=100, marker="*", color="red", zorder=5, label=f"best self_weight={sw_x[best_idx]:.2f}")
    a2.set_xlabel("lvp_self_weight")
    a2.set_ylabel("Objective")
    a2.set_title("Self-weight sweep (multiseed)")
    a2.grid(True, alpha=0.3)
    a2.legend(loc="best", fontsize=8)

    fig3.tight_layout()
    two_panel_png = ablation_dir / "lvp_tuning_two_left_panels.png"
    fig3.savefig(two_panel_png, dpi=170, bbox_inches="tight")
    plt.close(fig3)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build experiment_with_more_rounds package with article-like plots")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--seeds", type=str, default="42,52,62")
    p.add_argument("--out-dir", type=str, default=str(_FL / "artifacts" / "article_package_current_run_20260410" / "experiment_with_more_rounds"))
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--column-partition", type=str, default="contiguous", choices=["contiguous", "strided"])
    p.add_argument("--malicious-frac", type=float, default=0.25)
    p.add_argument("--attack-strategy", type=str, default="noise_colluded")
    p.add_argument("--attack-scale", type=float, default=5.0)
    p.add_argument("--similarity-tau", type=float, default=0.35)
    p.add_argument("--similarity-mode", type=str, default="jaccard", choices=["jaccard", "jaccard_cosine_hybrid"])
    p.add_argument("--lambda-jaccard", type=float, default=0.5)
    p.add_argument("--tau-cos-min", type=float, default=-1.0)
    p.add_argument("--lvp-alpha", type=float, default=0.6)
    p.add_argument("--lvp-self-weight", type=float, default=0.0)
    p.add_argument("--network-eval-mode", type=str, default="proxy", choices=["proxy", "refit"])
    p.add_argument("--alpha-grid", type=str, default="0.15,0.25,0.35,0.45,0.53,0.60")
    p.add_argument("--grid-workers", type=int, default=1)
    p.add_argument("--skip-seed-runs", action="store_true", help="Reuse existing raw/seeds outputs instead of rerunning fig3_decentralized_methods")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    alpha_grid = [float(x.strip()) for x in args.alpha_grid.split(",") if x.strip()]

    raw_seed_dirs: List[Path] = []
    for seed in seeds:
        seed_out = out_dir / "raw" / "seeds" / f"seed{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        raw_seed_dirs.append(seed_out)
        if not args.skip_seed_runs:
            _run_cmd([
                sys.executable,
                str(_EXP / "fig3_decentralized_methods.py"),
                "--out-dir", str(seed_out),
                "--rounds", str(args.rounds),
                "--seed", str(seed),
                "--n-clients", str(args.n_clients),
                "--column-partition", args.column_partition,
                "--malicious-frac", str(args.malicious_frac),
                "--attack-strategy", args.attack_strategy,
                "--attack-scale", str(args.attack_scale),
                "--similarity-tau", str(args.similarity_tau),
                "--similarity-mode", args.similarity_mode,
                "--lambda-jaccard", str(args.lambda_jaccard),
                "--tau-cos-min", str(args.tau_cos_min),
                "--lvp-alpha", str(args.lvp_alpha),
                "--lvp-self-weight", str(args.lvp_self_weight),
                "--network-eval-mode", args.network_eval_mode,
            ])

    _run_cmd([
        sys.executable,
        str(_EXP / "aggregate_fig3_multiseed.py"),
        "--input-dirs",
        *[str(d) for d in raw_seed_dirs],
        "--out-dir", str(out_dir / "plots" / "multiseed"),
    ])

    _run_cmd([
        sys.executable,
        str(_EXP / "fig3_boxplot_with_without_fedavg_multiseed.py"),
        "--input-dirs",
        *[str(d) for d in raw_seed_dirs],
        "--out-dir", str(out_dir / "plots" / "boxplots" / "all_methods"),
        "--single-panel",
    ])
    _run_cmd([
        sys.executable,
        str(_EXP / "fig3_boxplot_with_without_fedavg_multiseed.py"),
        "--input-dirs",
        *[str(d) for d in raw_seed_dirs],
        "--out-dir", str(out_dir / "plots" / "boxplots" / "all_methods_zoom"),
        "--single-panel",
        "--show-zoom-inset",
    ])
    _run_cmd([
        sys.executable,
        str(_EXP / "fig3_boxplot_with_without_fedavg_multiseed.py"),
        "--input-dirs",
        *[str(d) for d in raw_seed_dirs],
        "--out-dir", str(out_dir / "plots" / "boxplots" / "left2"),
        "--single-panel",
        "--exclude-aggregators", "defta,balance,push_sum",
    ])

    for seed_dir in raw_seed_dirs:
        seed = int(seed_dir.name.replace("seed", ""))
        _run_cmd([
            sys.executable,
            str(_EXP / "plot_decentralized_coherence.py"),
            "--input-json", str(seed_dir / "fig3_decentralized_methods.json"),
            "--out-dir", str(out_dir / "plots" / "coherence" / f"seed{seed}"),
        ])

    _run_cmd([
        sys.executable,
        str(_EXP / "plot_decentralized_coherence_multiseed.py"),
        "--input-dirs",
        *[str(d) for d in raw_seed_dirs],
        "--out-dir", str(out_dir / "plots" / "coherence" / "multiseed"),
    ])

    all_models_json = out_dir / "raw" / "all_models_lvp_seed42.json"
    _run_all_models_lvp(
        out_json=all_models_json,
        rounds=args.rounds,
        seed=42,
        n_clients=args.n_clients,
        column_partition=args.column_partition,
        malicious_frac=args.malicious_frac,
        attack_strategy=args.attack_strategy,
        attack_scale=args.attack_scale,
        similarity_tau=args.similarity_tau,
        similarity_mode=args.similarity_mode,
        lambda_jaccard=args.lambda_jaccard,
        tau_cos_min=args.tau_cos_min,
        lvp_alpha=args.lvp_alpha,
        lvp_self_weight=args.lvp_self_weight,
        network_eval_mode=args.network_eval_mode,
    )

    dynamics_dir = out_dir / "plots" / "dynamics"
    _plot_models_learning_curves(
        all_models_json,
        dynamics_dir,
        "Model comparison: Learning curves (LVP aggregator, current scenario)",
        "all_models_current_scenario_learning_curves.png",
    )
    _plot_models_learning_curves(
        all_models_json,
        dynamics_dir,
        "Model comparison: Learning curves (LVP aggregator)",
        "model_comparison_learning_curves.png",
    )

    _build_lvp_coherence_by_alpha(
        out_dir=out_dir / "plots" / "coherence",
        alpha_grid=alpha_grid,
        rounds=args.rounds,
        seed=42,
        n_clients=args.n_clients,
        column_partition=args.column_partition,
        malicious_frac=args.malicious_frac,
        attack_strategy=args.attack_strategy,
        attack_scale=args.attack_scale,
        similarity_tau=args.similarity_tau,
        similarity_mode=args.similarity_mode,
        lambda_jaccard=args.lambda_jaccard,
        tau_cos_min=args.tau_cos_min,
        lvp_self_weight=args.lvp_self_weight,
        network_eval_mode=args.network_eval_mode,
    )

    ablation_dir = out_dir / "plots" / "ablation"
    _run_cmd([
        sys.executable,
        str(_EXP / "ablate_tau_alpha_article_rounds10.py"),
        "--out-dir", str(ablation_dir),
        "--rounds", str(args.rounds),
        "--seed-list", args.seeds,
        "--column-partition", args.column_partition,
        "--malicious-frac", str(args.malicious_frac),
        "--attack-strategy", args.attack_strategy,
        "--attack-scale", str(args.attack_scale),
        "--network-eval-mode", args.network_eval_mode,
        "--similarity-mode", args.similarity_mode,
        "--lambda-jaccard", str(args.lambda_jaccard),
        "--tau-cos-min", str(args.tau_cos_min),
        "--fixed-alpha", str(args.lvp_alpha),
        "--fixed-tau", str(args.similarity_tau),
        "--grid-workers", str(args.grid_workers),
        "--num-workers", "1",
    ])

    _run_cmd([
        sys.executable,
        str(_EXP / "tune_lvp_alpha_self_weight_rounds10.py"),
        "--out-dir", str(ablation_dir),
        "--rounds", str(args.rounds),
        "--seed-list", args.seeds,
        "--column-partition", args.column_partition,
        "--malicious-frac", str(args.malicious_frac),
        "--attack-strategy", args.attack_strategy,
        "--attack-scale", str(args.attack_scale),
        "--network-eval-mode", args.network_eval_mode,
        "--similarity-mode", args.similarity_mode,
        "--lambda-jaccard", str(args.lambda_jaccard),
        "--tau-cos-min", str(args.tau_cos_min),
        "--tau", str(args.similarity_tau),
        "--grid-workers", str(args.grid_workers),
        "--num-workers", "1",
    ])

    _build_tuning_derived_plots(ablation_dir / "lvp_alpha_self_weight_tuning_summary.json", ablation_dir)

    # Re-create self_weight ablation in this package from local tuning summary (same naming as current package).
    payload = json.loads((ablation_dir / "lvp_alpha_self_weight_tuning_summary.json").read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    sw_to_vals: Dict[float, List[float]] = {}
    for row in rows:
        sw_to_vals.setdefault(float(row["self_weight"]), []).append(float(row["objective"]))
    sw_list = sorted(sw_to_vals.keys())
    means = [float(np.mean(sw_to_vals[sw])) for sw in sw_list]
    stds = [float(np.std(sw_to_vals[sw], ddof=0)) for sw in sw_list]
    best_idx = int(np.argmin(means))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.plot(sw_list, means, "o-", linewidth=2.2, markersize=5, label="mean objective")
    ax.fill_between(sw_list, np.asarray(means) - np.asarray(stds), np.asarray(means) + np.asarray(stds), alpha=0.15, label="+-1 std")
    ax.scatter([sw_list[best_idx]], [means[best_idx]], s=100, marker="*", color="red", zorder=5, label=f"best self_weight={sw_list[best_idx]:.2f}")
    ax.set_xlabel("lvp_self_weight")
    ax.set_ylabel("Objective (mean + 0.5*std)")
    ax.set_title(f"Ablation for rounds={args.rounds} scenario: self_weight sweep (multiseed)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    sw_ablation_png = ablation_dir / "self_weight_ablation_multiseed.png"
    fig.savefig(sw_ablation_png, dpi=170, bbox_inches="tight")
    plt.close(fig)

    (ablation_dir / "self_weight_ablation_multiseed_report.md").write_text(
        "\n".join([
            f"# Self-weight Ablation (rounds={args.rounds} scenario, multiseed)",
            "",
            f"Best self_weight: {sw_list[best_idx]:.6f} (objective={means[best_idx]:.6f}, std={stds[best_idx]:.6f})",
            "",
            f"Figure: {sw_ablation_png}",
        ]),
        encoding="utf-8",
    )

    manifest = {
        "rounds": int(args.rounds),
        "seeds": seeds,
        "out_dir": str(out_dir),
        "generated": {
            "raw": [str(out_dir / "raw" / "all_models_lvp_seed42.json")] + [str(d / "fig3_decentralized_methods.json") for d in raw_seed_dirs],
            "plots_root": str(out_dir / "plots"),
        },
    }
    (out_dir / "EXPERIMENT_WITH_MORE_ROUNDS_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
