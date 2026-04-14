#!/usr/bin/env python3
"""Benchmark several LVP-oriented scenarios and compare decentralized methods.

The script runs a small set of pre-defined scenarios that differ by topology,
evaluation mode and LVP damping. For each scenario it reuses the decentralized
comparison experiment, stores a dedicated plot/JSON/report, and writes a final
cross-scenario analysis that balances final MAE and stability (max jump).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "core"))
sys.path.insert(0, str(_FL / "data_loaders"))

from data_utils import build_client_information_profiles, build_clients_from_mcc, load_mcc_series  # noqa: E402
from run_real_experiments import MODEL_REGISTRY, _build_exogenous, run_one_model  # noqa: E402


DECENTRALIZED_METHODS = ["lvp", "decentralized_fedavg", "defta", "balance", "push_sum"]


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    column_partition: str
    similarity_mode: str
    similarity_tau: float
    lambda_jaccard: float
    tau_cos_min: float
    lvp_alpha: float
    lvp_self_weight: float
    network_eval_mode: str
    sync_topic_groups: int


DEFAULT_SCENARIOS: List[ScenarioConfig] = [
    ScenarioConfig(
        name="stable_refit",
        column_partition="strided",
        similarity_mode="jaccard_cosine_hybrid",
        similarity_tau=0.82,
        lambda_jaccard=0.6,
        tau_cos_min=0.15,
        lvp_alpha=0.25,
        lvp_self_weight=0.30,
        network_eval_mode="refit",
        sync_topic_groups=4,
    ),
    ScenarioConfig(
        name="balanced_refit",
        column_partition="strided",
        similarity_mode="jaccard_cosine_hybrid",
        similarity_tau=0.77,
        lambda_jaccard=0.5,
        tau_cos_min=0.05,
        lvp_alpha=0.25,
        lvp_self_weight=0.25,
        network_eval_mode="refit",
        sync_topic_groups=4,
    ),
    ScenarioConfig(
        name="competitive_refit",
        column_partition="strided",
        similarity_mode="jaccard_cosine_hybrid",
        similarity_tau=0.72,
        lambda_jaccard=0.6,
        tau_cos_min=0.10,
        lvp_alpha=0.25,
        lvp_self_weight=0.15,
        network_eval_mode="refit",
        sync_topic_groups=4,
    ),
]


def _topic_profiles(clients: List, n_groups: int) -> List[frozenset]:
    profiles = build_client_information_profiles(clients, "mcc")
    return [frozenset(p) | {f"sync_topic_{idx % n_groups}"} for idx, p in enumerate(profiles)]


def _history(exp: Dict[str, Any]) -> List[float]:
    return [float(r["network_mae"]) for r in exp.get("history") or []]


def _final_metrics(exp: Dict[str, Any]) -> Dict[str, float]:
    vals = np.asarray(_history(exp), dtype=float)
    if vals.size == 0:
        return {"final_mae": float("inf"), "max_jump": float("inf"), "round_std": float("inf")}
    jumps = np.abs(np.diff(vals)) if vals.size > 1 else np.asarray([0.0], dtype=float)
    return {
        "final_mae": float(vals[-1]),
        "max_jump": float(np.max(jumps)),
        "round_std": float(np.std(vals)),
    }


def _plot_comparison(curves: Dict[str, Dict[str, Any]], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    labels = {
        "lvp": "LVP",
        "decentralized_fedavg": "Decentralized FedAvg",
        "defta": "DeFTA",
        "balance": "BALANCE",
        "push_sum": "Push-Sum",
    }
    order = DECENTRALIZED_METHODS
    # Create two subplots: abs MAE and delta
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))
    
    for agg in order:
        if agg not in curves:
            continue
        c = curves[agg]
        xs = np.asarray(c["round"], dtype=int)
        m = np.asarray(c["mean_network_mae"], dtype=float)
        s = np.asarray(c["std_network_mae"], dtype=float)
        lw = 2.8 if agg == "lvp" else 1.8
        
        # Plot 1: Absolute MAE
        ax1.plot(xs, m, "o-", linewidth=lw, markersize=4.5, label=labels.get(agg, agg))
        ax1.fill_between(xs, m - s, m + s, alpha=0.12)
        
        # Plot 2: Delta from initial value (first round = baseline)
        if len(m) > 0:
            m_delta = (m - m[0])
            s_delta = s  # std deviation remains same
            ax2.plot(xs, m_delta, "o-", linewidth=lw, markersize=4.5, label=labels.get(agg, agg))
            ax2.fill_between(xs, m_delta - s_delta, m_delta + s_delta, alpha=0.12)
    
    # Configure axis 1 (absolute)
    ax1.set_xlabel("Communication round", fontsize=11)
    ax1.set_ylabel("Network MAE", fontsize=11)
    ax1.set_title(f"{title} (Absolute MAE)", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=9)
    
    # Configure axis 2 (delta)
    ax2.set_xlabel("Communication round", fontsize=11)
    ax2.set_ylabel("Δ MAE (from round 1)", fontsize=11)
    ax2.set_title(f"{title} (Convergence delta)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", fontsize=9)
    ax2.axhline(y=0, color="k", linestyle="--", linewidth=0.8, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark multiple LVP scenarios and compare decentralized methods")
    parser.add_argument("--base-path", type=str, default=str(_FL.parent))
    parser.add_argument("--out-dir", type=str, default=str(_FL / "artifacts" / "lvp_scenario_benchmark"))
    parser.add_argument("--model", type=str, default="DynamicLinearModel")
    parser.add_argument("--n-clients", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--local-fit-maxiter", type=int, default=10)
    parser.add_argument("--eval-fit-maxiter", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--malicious-frac", type=float, default=0.25)
    parser.add_argument("--attack-strategy", type=str, default="noise_colluded")
    parser.add_argument("--attack-scale", type=float, default=2.5)
    parser.add_argument("--seed-list", type=str, default="42,52")
    parser.add_argument("--scenario-names", type=str, default="stable_refit,competitive_refit,paper_fair_proxy")
    parser.add_argument("--scenario-limit", type=int, default=0)
    parser.add_argument("--score-beta", type=float, default=0.3)
    parser.add_argument("--score-gamma", type=float, default=0.2)
    args = parser.parse_args()

    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.base_path).resolve()

    selected_names = [x.strip() for x in args.scenario_names.split(",") if x.strip()]
    scenarios = [s for s in DEFAULT_SCENARIOS if s.name in selected_names]
    if args.scenario_limit > 0:
        scenarios = scenarios[: args.scenario_limit]
    if not scenarios:
        raise ValueError("No scenarios selected for benchmark")

    seeds = [int(x.strip()) for x in args.seed_list.split(",") if x.strip()]
    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=True)
    ModelClass = MODEL_REGISTRY[args.model]

    scenario_summaries: List[Dict[str, Any]] = []

    for sc in scenarios:
        print(f"=== Scenario: {sc.name} ===")
        clients = build_clients_from_mcc(
            mcc_df,
            exog,
            n_clients=args.n_clients,
            column_partition=sc.column_partition,
        )
        profiles = _topic_profiles(clients, sc.sync_topic_groups)

        exps: Dict[str, List[Dict[str, Any]]] = {m: [] for m in DECENTRALIZED_METHODS}
        for seed in seeds:
            for agg in DECENTRALIZED_METHODS:
                exp = run_one_model(
                    args.model,
                    ModelClass,
                    clients,
                    profiles,
                    aggregator=agg,
                    rounds=args.rounds,
                    local_epochs=args.local_epochs,
                    malicious_frac=args.malicious_frac,
                    seed=seed,
                    attack_strategy=args.attack_strategy,
                    attack_scale=args.attack_scale,
                    similarity_tau=sc.similarity_tau,
                    similarity_mode=sc.similarity_mode,
                    lambda_jaccard=sc.lambda_jaccard,
                    tau_cos_min=sc.tau_cos_min,
                    lvp_alpha=sc.lvp_alpha if agg == "lvp" else None,
                    lvp_self_weight=sc.lvp_self_weight if agg == "lvp" else 0.0,
                    krum_f=-1,
                    local_fit_maxiter=args.local_fit_maxiter,
                    eval_fit_maxiter=args.eval_fit_maxiter,
                    strict_errors=True,
                    network_eval_mode=sc.network_eval_mode,
                )
                exps[agg].append(exp)

        curves = {}
        for agg, items in exps.items():
            min_len = min(len(e["history"]) for e in items)
            history_matrix = np.asarray(
                [[float(r["network_mae"]) for r in e["history"][:min_len]] for e in items],
                dtype=float,
            )
            curves[agg] = {
                "round": list(range(1, min_len + 1)),
                "mean_network_mae": np.mean(history_matrix, axis=0).tolist(),
                "std_network_mae": np.std(history_matrix, axis=0).tolist(),
            }
        finals = {agg: [_final_metrics(e) for e in items] for agg, items in exps.items()}
        final_summary = {
            agg: {
                "final_mae_mean": float(np.mean([m["final_mae"] for m in ms])),
                "final_mae_std": float(np.std([m["final_mae"] for m in ms])),
                "max_jump_mean": float(np.mean([m["max_jump"] for m in ms])),
                "round_std_mean": float(np.mean([m["round_std"] for m in ms])),
            }
            for agg, ms in finals.items()
        }
        ranking = sorted(final_summary.items(), key=lambda kv: kv[1]["final_mae_mean"])

        fig_path = out_dir / f"{sc.name}_decentralized_mae.png"
        _plot_comparison(
            curves,
            fig_path,
            title=(
                f"Decentralized comparison | {sc.name} | partition={sc.column_partition} | "
                f"eval={sc.network_eval_mode} | lvp=tau={sc.similarity_tau:.2f}, alpha={sc.lvp_alpha:.2f}, self={sc.lvp_self_weight:.2f}"
            ),
        )

        json_path = out_dir / f"{sc.name}_decentralized_metrics.json"
        payload = {
            "scenario": asdict(sc),
            "curves": curves,
            "final_summary": final_summary,
            "final_ranking": [{"aggregator": k, **v} for k, v in ranking],
            "artifacts": {"figure": str(fig_path)},
        }
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        report_path = out_dir / f"{sc.name}_decentralized_report.md"
        report_lines = [
            f"# Scenario: {sc.name}",
            "",
            f"Partition: {sc.column_partition}",
            f"Eval mode: {sc.network_eval_mode}",
            f"LVP: tau={sc.similarity_tau:.3f}, alpha={sc.lvp_alpha:.3f}, self={sc.lvp_self_weight:.3f}",
            "",
            "## Final ranking",
        ]
        for row in payload["final_ranking"]:
            report_lines.append(
                f"- {row['aggregator']}: {row['final_mae_mean']:.6f} (std={row['final_mae_std']:.6f})"
            )
        report_path.write_text("\n".join(report_lines), encoding="utf-8")

        scenario_summaries.append(
            {
                "scenario": sc.name,
                "final_ranking": payload["final_ranking"],
                "best_method": ranking[0][0],
                "lvp_final_mae": final_summary["lvp"]["final_mae_mean"],
                "lvp_max_jump": final_summary["lvp"]["max_jump_mean"],
                "lvp_round_std": final_summary["lvp"]["round_std_mean"],
                "figure": str(fig_path),
                "json": str(json_path),
                "report": str(report_path),
            }
        )

        print(f"Saved: {fig_path}")
        print(f"Saved: {json_path}")
        print(f"Saved: {report_path}")

    def _scenario_score(row: Dict[str, Any]) -> float:
        return float(row["lvp_final_mae"] + args.score_beta * row["lvp_max_jump"] + args.score_gamma * row["lvp_round_std"])

    scenario_summaries.sort(key=_scenario_score)
    best = scenario_summaries[0]

    summary_payload = {
        "scenarios": scenario_summaries,
        "best_scenario": best,
        "selection_rule": {
            "score": "lvp_final_mae + beta*max_jump + gamma*round_std",
            "beta": args.score_beta,
            "gamma": args.score_gamma,
        },
    }
    summary_json = out_dir / "benchmark_summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    summary_lines = [
        "# LVP Scenario Benchmark",
        "",
        "## Best scenario",
        f"- {best['scenario']}",
        f"- lvp_final_mae: {best['lvp_final_mae']:.6f}",
        f"- lvp_max_jump: {best['lvp_max_jump']:.6f}",
        f"- lvp_round_std: {best['lvp_round_std']:.6f}",
        "",
        "## Scenario ranking",
    ]
    for row in scenario_summaries:
        summary_lines.append(
            f"- {row['scenario']}: score={_scenario_score(row):.6f}, lvp_final={row['lvp_final_mae']:.6f}, max_jump={row['lvp_max_jump']:.6f}"
        )
    summary_md = out_dir / "benchmark_summary.md"
    summary_md.write_text("\n".join(summary_lines), encoding="utf-8")

    print(f"Saved: {summary_json}")
    print(f"Saved: {summary_md}")
    print(f"Best scenario: {best['scenario']}")


if __name__ == "__main__":
    main()
