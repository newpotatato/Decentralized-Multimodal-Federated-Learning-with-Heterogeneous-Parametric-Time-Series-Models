#!/usr/bin/env python3
"""Ablation for the selected decentralized scenario: Jaccard vs hybrid LVP.

Fixed scenario (matches the selected article plot):
- model: DynamicLinearModel
- partition: contiguous
- malicious_frac: 0.25
- attack_strategy: noise_colluded
- attack_scale: 5.0
- rounds: 10
- local_epochs: 1
- seed list: configurable (default 42, 52, 62)

The ablation sweeps lambda_jaccard for the hybrid similarity and compares it to
pure Jaccard (lambda=1.0 equivalent) under the same scenario.
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


DEFAULT_LAMBDAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _final_mae(exp: Dict[str, Any]) -> float:
    hist = exp.get("history", [])
    if not hist:
        return float("inf")
    return float(hist[-1].get("network_mae", float("inf")))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablate hybrid Jaccard coefficient on the selected scenario")
    p.add_argument("--base-path", type=str, default=str(_FL.parent))
    p.add_argument("--out-dir", type=str, default=str(_FL / "artifacts" / "selected_scenario_hybrid_ablation"))
    p.add_argument("--model", type=str, default="DynamicLinearModel", choices=list(MODEL_REGISTRY.keys()))
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--seed-list", type=str, default="42,52,62")
    p.add_argument("--similarity-tau", type=float, default=0.35)
    p.add_argument("--tau-cos-min", type=float, default=0.0)
    p.add_argument("--lvp-alpha", type=float, default=0.53)
    p.add_argument("--lambda-grid", type=str, default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument("--attack-scale", type=float, default=5.0)
    p.add_argument("--local-fit-maxiter", type=int, default=10)
    p.add_argument("--eval-fit-maxiter", type=int, default=0)
    p.add_argument("--network-eval-mode", type=str, default="proxy", choices=["proxy", "refit"])
    p.add_argument("--no-reuters", action="store_true")
    return p.parse_args()


def _parse_csv_floats(text: str) -> List[float]:
    out: List[float] = []
    for item in (text or "").split(","):
        item = item.strip()
        if not item:
            continue
        out.append(float(item))
    return out


def main() -> None:
    args = parse_args()
    base = Path(args.base_path).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(x) for x in (args.seed_list or "").split(",") if x.strip()]
    lambdas = _parse_csv_floats(args.lambda_grid)

    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=not args.no_reuters)
    clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=args.n_clients,
        column_partition="contiguous",
    )
    profiles = build_client_information_profiles(clients, "mcc")
    if len(clients) < 2:
        raise RuntimeError("Need at least 2 clients.")

    ModelClass = MODEL_REGISTRY[args.model]
    records: List[Dict[str, Any]] = []

    for seed in seeds:
        for similarity_mode in ["jaccard", "jaccard_cosine_hybrid"]:
            for lam in lambdas:
                if similarity_mode == "jaccard" and lam != 1.0:
                    continue
                exp = run_one_model(
                    args.model,
                    ModelClass,
                    clients,
                    profiles,
                    aggregator="lvp",
                    rounds=args.rounds,
                    local_epochs=args.local_epochs,
                    malicious_frac=0.25,
                    seed=seed,
                    attack_strategy="noise_colluded",
                    attack_scale=float(args.attack_scale),
                    similarity_tau=float(args.similarity_tau),
                    similarity_mode=similarity_mode,
                    lambda_jaccard=float(lam),
                    tau_cos_min=float(args.tau_cos_min),
                    lvp_alpha=float(args.lvp_alpha),
                    lvp_self_weight=0.0,
                    krum_f=-1,
                    local_fit_maxiter=args.local_fit_maxiter,
                    eval_fit_maxiter=args.eval_fit_maxiter,
                    strict_errors=True,
                    network_eval_mode=args.network_eval_mode,
                )
                records.append(
                    {
                        "seed": seed,
                        "similarity_mode": similarity_mode,
                        "lambda_jaccard": float(lam),
                        "final_network_mae": _final_mae(exp),
                        "history": exp["history"],
                    }
                )
                print(
                    f"seed={seed} mode={similarity_mode} lambda={lam:.2f} final_mae={records[-1]['final_network_mae']:.6f}"
                )

    # Summaries by lambda for hybrid and by pure Jaccard.
    summary: Dict[str, Any] = {
        "config": {
            "model": args.model,
            "n_clients": args.n_clients,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "seeds": seeds,
            "similarity_tau": args.similarity_tau,
            "tau_cos_min": args.tau_cos_min,
            "lvp_alpha": args.lvp_alpha,
            "attack_scale": args.attack_scale,
            "network_eval_mode": args.network_eval_mode,
            "lambda_grid": lambdas,
        },
        "records": records,
    }

    grouped: Dict[str, List[float]] = {}
    for row in records:
        key = f"{row['similarity_mode']}|{row['lambda_jaccard']:.2f}"
        grouped.setdefault(key, []).append(float(row["final_network_mae"]))

    summary_rows: List[Dict[str, Any]] = []
    for key, vals in grouped.items():
        mode, lam_txt = key.split("|")
        arr = np.asarray(vals, dtype=float)
        summary_rows.append(
            {
                "similarity_mode": mode,
                "lambda_jaccard": float(lam_txt),
                "n": int(arr.size),
                "mean_final_mae": float(np.mean(arr)),
                "std_final_mae": float(np.std(arr, ddof=0)),
                "median_final_mae": float(np.median(arr)),
                "min_final_mae": float(np.min(arr)),
                "max_final_mae": float(np.max(arr)),
            }
        )

    summary["summary_rows"] = sorted(summary_rows, key=lambda r: (r["similarity_mode"], r["lambda_jaccard"]))

    json_path = out_dir / "selected_scenario_hybrid_ablation.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Plot.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    hybrid_rows = [r for r in summary["summary_rows"] if r["similarity_mode"] == "jaccard_cosine_hybrid"]
    hybrid_rows.sort(key=lambda r: r["lambda_jaccard"])
    xs = [r["lambda_jaccard"] for r in hybrid_rows]
    ys = [r["mean_final_mae"] for r in hybrid_rows]
    es = [r["std_final_mae"] for r in hybrid_rows]
    ax.errorbar(xs, ys, yerr=es, fmt="o-", capsize=4, linewidth=2.2, markersize=5, label="Hybrid")

    jacc_rows = [r for r in summary["summary_rows"] if r["similarity_mode"] == "jaccard"]
    if jacc_rows:
        j = jacc_rows[0]
        ax.axhline(j["mean_final_mae"], color="#444444", linestyle="--", linewidth=1.6, label="Pure Jaccard")

    ax.set_xlabel("lambda_jaccard")
    ax.set_ylabel("Final network MAE")
    ax.set_title("Selected scenario ablation: pure Jaccard vs hybrid LVP")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    fig_path = out_dir / "selected_scenario_hybrid_ablation.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    best_hybrid = min(hybrid_rows, key=lambda r: r["mean_final_mae"]) if hybrid_rows else None
    best_jaccard = jacc_rows[0] if jacc_rows else None

    md_lines = [
        "# Selected Scenario Hybrid Ablation",
        "",
        f"Model: {args.model}",
        f"Scenario: partition=contiguous, mal=0.25, attack=noise_colluded, scale={args.attack_scale}, rounds={args.rounds}, local_epochs={args.local_epochs}",
        "",
        "## Summary by configuration",
    ]
    for row in summary["summary_rows"]:
        md_lines.append(
            f"- {row['similarity_mode']} lambda={row['lambda_jaccard']:.2f}: mean={row['mean_final_mae']:.6f}, std={row['std_final_mae']:.6f}, n={row['n']}"
        )
    if best_hybrid:
        md_lines.extend([
            "",
            f"## Best hybrid", 
            f"- lambda_jaccard={best_hybrid['lambda_jaccard']:.2f}",
            f"- mean_final_mae={best_hybrid['mean_final_mae']:.6f}",
        ])
    if best_jaccard:
        md_lines.extend([
            "",
            "## Pure Jaccard",
            f"- mean_final_mae={best_jaccard['mean_final_mae']:.6f}",
        ])

    md_path = out_dir / "selected_scenario_hybrid_ablation_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"Saved: {json_path}")
    print(f"Saved: {fig_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
