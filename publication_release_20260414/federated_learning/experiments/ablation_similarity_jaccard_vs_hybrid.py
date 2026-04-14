#!/usr/bin/env python3
"""
Ablation 2.0: Jaccard vs Jaccard+Cosine hybrid similarity for decentralized LVP.

Design (default):
- Model: DynamicLinearModel
- Aggregator: LVP (optional FedAvg reference)
- Partitions: contiguous, strided
- Malicious fractions: 0.2, 0.35
- Seeds: 42, 123, 456, 789, 2024
- Rounds: 20

Outputs:
- artifacts/similarity_ablation/similarity_ablation_raw.json
- artifacts/similarity_ablation/similarity_ablation_summary.json
- artifacts/similarity_ablation/plots/*.png
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import json
import sys
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "data_loaders"))
sys.path.insert(0, str(_FL / "core"))

from data_utils import build_client_information_profiles, build_clients_from_mcc, load_mcc_series
from run_real_experiments import MODEL_REGISTRY, _build_exogenous, run_one_model


ARTIFACTS = _FL / "artifacts" / "similarity_ablation"
_WORKER_SCENARIO_DATA: Dict[str, Tuple[List, List[frozenset]]] = {}


@dataclass(frozen=True)
class Scenario:
    partition: str
    malicious_frac: float


def _auc_from_history(exp: Dict) -> float:
    xs = np.array([int(r["round"]) for r in exp["history"]], dtype=float)
    ys = np.array([float(r["network_mae"]) for r in exp["history"]], dtype=float)
    if xs.size < 2:
        return float(ys[0]) if ys.size else float("nan")
    return float(np.trapezoid(ys, xs))


def _final_mae(exp: Dict) -> float:
    return float(exp["history"][-1]["network_mae"])


def _safe_wilcoxon(x: List[float], y: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if len(x) != len(y) or len(x) < 2:
        return None, None
    try:
        from scipy.stats import wilcoxon

        res = wilcoxon(x, y, alternative="less")
        stat_raw, pval_raw = tuple(res)
        stat = float(np.ravel(np.asarray(stat_raw, dtype=float))[0])
        pval = float(np.ravel(np.asarray(pval_raw, dtype=float))[0])
        return stat, pval
    except Exception:
        return None, None


@dataclass(frozen=True)
class ExpTask:
    """Single experiment task for parallel execution."""
    scenario_key: str
    partition: str
    malicious_frac: float
    seed: int
    similarity_mode: str
    aggregator: str
    model_name: str
    attack_strategy: str
    rounds: int
    local_epochs: int
    attack_scale: float
    similarity_tau: float
    lambda_jaccard: float
    tau_cos_min: float
    lvp_alpha: Optional[float]
    local_fit_maxiter: int
    eval_fit_maxiter: int
    strict_errors: bool
    network_eval_mode: str


def _init_worker(scenario_data: Dict[str, Tuple[List, List[frozenset]]]) -> None:
    global _WORKER_SCENARIO_DATA
    _WORKER_SCENARIO_DATA = scenario_data


def _run_single_exp(task: ExpTask) -> Dict:
    """Execute a single federated experiment (called in parallel)."""
    clients, profiles = _WORKER_SCENARIO_DATA[task.scenario_key]
    ModelClass = MODEL_REGISTRY[task.model_name]

    exp = run_one_model(
        task.model_name,
        ModelClass,
        clients,
        profiles,
        aggregator=task.aggregator,
        rounds=task.rounds,
        local_epochs=task.local_epochs,
        malicious_frac=task.malicious_frac,
        seed=task.seed,
        attack_strategy=task.attack_strategy,
        attack_scale=task.attack_scale,
        similarity_tau=task.similarity_tau,
        similarity_mode=task.similarity_mode,
        lambda_jaccard=task.lambda_jaccard,
        tau_cos_min=task.tau_cos_min,
        lvp_alpha=task.lvp_alpha,
        krum_f=-1,
        local_fit_maxiter=task.local_fit_maxiter,
        eval_fit_maxiter=task.eval_fit_maxiter,
        strict_errors=task.strict_errors,
        network_eval_mode=task.network_eval_mode,
    )
    
    return {
        "partition": task.partition,
        "malicious_frac": task.malicious_frac,
        "seed": int(task.seed),
        "similarity_mode": task.similarity_mode,
        "aggregator": task.aggregator,
        "final_network_mae": _final_mae(exp),
        "auc_network_mae": _auc_from_history(exp),
        "history": exp["history"],
    }


def _plot_boxplots(rows: List[Dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    scenarios = sorted({(r["partition"], float(r["malicious_frac"])) for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    axes = axes.flatten()

    for idx, sc in enumerate(scenarios[:4]):
        part, mal = sc
        ax = axes[idx]
        sub = [r for r in rows if r["partition"] == part and float(r["malicious_frac"]) == mal and r["aggregator"] == "lvp"]
        j = [float(r["final_network_mae"]) for r in sub if r["similarity_mode"] == "jaccard"]
        h = [float(r["final_network_mae"]) for r in sub if r["similarity_mode"] == "jaccard_cosine_hybrid"]
        data = [j, h]
        ax.boxplot(data, labels=["Jaccard", "Jaccard+Cos"], showmeans=True)
        ax.set_title(f"partition={part}, malicious={mal:.2f}")
        ax.set_ylabel("Final network MAE")
        ax.grid(True, alpha=0.2)

    path = out_dir / "boxplot_final_mae_j_vs_hybrid.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_effects(summary_rows: List[Dict], out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = [f"{r['partition']} | mal={r['malicious_frac']:.2f}" for r in summary_rows]
    effects = [float(r["median_delta_final_mae"]) for r in summary_rows]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(labels, effects, color=["#2E8B57" if e > 0 else "#B22222" for e in effects])
    ax.axhline(0.0, color="black", linestyle=":", linewidth=1)
    ax.set_ylabel("Median Δ final MAE = Jaccard - Hybrid")
    ax.set_title("Effect size by scenario (positive => hybrid better)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(True, axis="y", alpha=0.25)

    path = out_dir / "effect_median_delta_final_mae.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Tuple[Dict, Dict]:
    base_path = Path(args.base_path).resolve()
    model_name = args.model
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    mcc_df = load_mcc_series(base_path)
    exog = _build_exogenous(base_path, mcc_df, use_reuters=not args.no_reuters)

    scenarios = [
        Scenario(partition="contiguous", malicious_frac=0.2),
        Scenario(partition="contiguous", malicious_frac=0.35),
        Scenario(partition="strided", malicious_frac=0.2),
        Scenario(partition="strided", malicious_frac=0.35),
    ]
    modes = ["jaccard", "jaccard_cosine_hybrid"]
    aggregators = ["lvp"] + (["fedavg"] if args.include_fedavg else [])

    # Pre-build all clients and profiles (avoid pickling mcc_df repeatedly)
    scenario_data: Dict[str, Tuple[List, List[frozenset]]] = {}
    for sc in scenarios:
        clients = build_clients_from_mcc(
            mcc_df,
            exog,
            n_clients=args.n_clients,
            column_partition=sc.partition,
        )
        if len(clients) < 2:
            raise RuntimeError(f"Insufficient clients for scenario={sc}")
        profiles = build_client_information_profiles(clients, "mcc")
        key = f"{sc.partition}|{sc.malicious_frac:.2f}"
        scenario_data[key] = (clients, profiles)

    # Build task list for parallel execution
    tasks: List[ExpTask] = []
    total = len(scenarios) * len(args.seeds) * len(modes) * len(aggregators)
    
    for sc in scenarios:
        key = f"{sc.partition}|{sc.malicious_frac:.2f}"
        for seed in args.seeds:
            for mode in modes:
                for agg in aggregators:
                    task = ExpTask(
                        scenario_key=key,
                        partition=sc.partition,
                        malicious_frac=sc.malicious_frac,
                        seed=seed,
                        similarity_mode=mode,
                        aggregator=agg,
                        model_name=model_name,
                        attack_strategy=args.attack_strategy,
                        rounds=args.rounds,
                        local_epochs=args.local_epochs,
                        attack_scale=args.attack_scale,
                        similarity_tau=args.similarity_tau,
                        lambda_jaccard=args.lambda_jaccard,
                        tau_cos_min=args.tau_cos_min,
                        lvp_alpha=args.lvp_alpha,
                        local_fit_maxiter=args.local_fit_maxiter,
                        eval_fit_maxiter=args.eval_fit_maxiter,
                        strict_errors=args.strict_errors,
                        network_eval_mode=args.network_eval_mode,
                    )
                    tasks.append(task)

    print(f"Running {total} experiments with {mp.cpu_count()} CPUs...")
    
    # Use multiprocessing Pool for parallel execution
    rows: List[Dict] = []
    n_proc = max(1, min(mp.cpu_count() - 1, args.max_workers))
    chunksize = max(1, len(tasks) // max(1, n_proc * 4))
    with mp.Pool(
        processes=n_proc,
        initializer=_init_worker,
        initargs=(scenario_data,),
    ) as pool:
        # Use imap_unordered for better load balancing (order doesn't matter)
        for idx, result in enumerate(pool.imap_unordered(_run_single_exp, tasks, chunksize=chunksize), 1):
            rows.append(result)
            print(f"[{idx}/{total}] {result['partition']} mal={result['malicious_frac']:.2f} "
                  f"seed={result['seed']} mode={result['similarity_mode']} agg={result['aggregator']}")

    # Paired stats for LVP only (sequential - no bottleneck)
    summary_rows: List[Dict] = []
    for sc in scenarios:
        pairs_final_j: List[float] = []
        pairs_final_h: List[float] = []
        pairs_auc_j: List[float] = []
        pairs_auc_h: List[float] = []

        for seed in args.seeds:
            j = next(
                (
                    r
                    for r in rows
                    if r["partition"] == sc.partition
                    and float(r["malicious_frac"]) == sc.malicious_frac
                    and r["seed"] == seed
                    and r["similarity_mode"] == "jaccard"
                    and r["aggregator"] == "lvp"
                ),
                None,
            )
            h = next(
                (
                    r
                    for r in rows
                    if r["partition"] == sc.partition
                    and float(r["malicious_frac"]) == sc.malicious_frac
                    and r["seed"] == seed
                    and r["similarity_mode"] == "jaccard_cosine_hybrid"
                    and r["aggregator"] == "lvp"
                ),
                None,
            )
            if j is None or h is None:
                continue
            pairs_final_j.append(float(j["final_network_mae"]))
            pairs_final_h.append(float(h["final_network_mae"]))
            pairs_auc_j.append(float(j["auc_network_mae"]))
            pairs_auc_h.append(float(h["auc_network_mae"]))

        deltas_final = [a - b for a, b in zip(pairs_final_j, pairs_final_h)]
        deltas_auc = [a - b for a, b in zip(pairs_auc_j, pairs_auc_h)]
        stat_f, p_f = _safe_wilcoxon(pairs_final_h, pairs_final_j)  # H1: hybrid < jaccard
        stat_a, p_a = _safe_wilcoxon(pairs_auc_h, pairs_auc_j)

        summary_rows.append(
            {
                "partition": sc.partition,
                "malicious_frac": sc.malicious_frac,
                "n_pairs": len(deltas_final),
                "median_delta_final_mae": float(np.median(deltas_final)) if deltas_final else None,
                "mean_delta_final_mae": float(np.mean(deltas_final)) if deltas_final else None,
                "median_delta_auc": float(np.median(deltas_auc)) if deltas_auc else None,
                "mean_delta_auc": float(np.mean(deltas_auc)) if deltas_auc else None,
                "wilcoxon_stat_final": stat_f,
                "wilcoxon_pvalue_final": p_f,
                "wilcoxon_stat_auc": stat_a,
                "wilcoxon_pvalue_auc": p_a,
                "jaccard_final_mean": float(np.mean(pairs_final_j)) if pairs_final_j else None,
                "hybrid_final_mean": float(np.mean(pairs_final_h)) if pairs_final_h else None,
            }
        )

    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    raw_payload = {
        "config": {
            "model": model_name,
            "n_clients": args.n_clients,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "seeds": args.seeds,
            "similarity_tau": args.similarity_tau,
            "lambda_jaccard": args.lambda_jaccard,
            "tau_cos_min": args.tau_cos_min,
            "lvp_alpha": args.lvp_alpha,
            "attack_strategy": args.attack_strategy,
            "attack_scale": args.attack_scale,
            "include_fedavg": args.include_fedavg,
            "strict_errors": args.strict_errors,
            "network_eval_mode": args.network_eval_mode,
            "max_workers": args.max_workers,
        },
        "rows": rows,
    }
    summary_payload = {
        "config": raw_payload["config"],
        "scenario_summary": summary_rows,
    }

    (out_dir / "similarity_ablation_raw.json").write_text(
        json.dumps(raw_payload, indent=2), encoding="utf-8"
    )
    (out_dir / "similarity_ablation_summary.json").write_text(
        json.dumps(summary_payload, indent=2), encoding="utf-8"
    )

    _plot_boxplots(rows, plot_dir)
    _plot_effects(summary_rows, plot_dir)

    return raw_payload, summary_payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ablation: Jaccard vs Jaccard+Cosine for LVP")
    p.add_argument("--base-path", type=str, default=str(_FL.parent))
    p.add_argument("--out-dir", type=str, default=str(ARTIFACTS))
    p.add_argument("--model", type=str, default="DynamicLinearModel")
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456, 789, 2024])
    p.add_argument("--similarity-tau", type=float, default=0.35)
    p.add_argument("--lambda-jaccard", type=float, default=0.5)
    p.add_argument("--tau-cos-min", type=float, default=0.0)
    p.add_argument(
        "--lvp-alpha",
        type=float,
        default=0.53,
        help="If <=0, run_one_model will use graph-based auto alpha",
    )
    p.add_argument("--attack-strategy", type=str, default="label_flip")
    p.add_argument("--attack-scale", type=float, default=3.0)
    p.add_argument("--local-fit-maxiter", type=int, default=10)
    p.add_argument("--eval-fit-maxiter", type=int, default=0)
    p.add_argument(
        "--network-eval-mode",
        type=str,
        default="proxy",
        choices=["refit", "proxy"],
        help="Use 'proxy' for faster ablation runs, 'refit' for exact post-sync network metrics.",
    )
    p.add_argument(
        "--strict-errors",
        action="store_true",
        help="Raise on model failures instead of silently substituting penalties.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Upper bound for multiprocessing workers.",
    )
    p.add_argument("--no-reuters", action="store_true")
    p.add_argument("--include-fedavg", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.lvp_alpha is not None and args.lvp_alpha <= 0:
        args.lvp_alpha = None

    _, summary = run(args)

    print("\n=== Hypothesis check (preliminary) ===")
    for row in summary["scenario_summary"]:
        part = row["partition"]
        mal = row["malicious_frac"]
        md = row["median_delta_final_mae"]
        pval = row["wilcoxon_pvalue_final"]
        print(
            f"scenario partition={part}, malicious={mal:.2f}: "
            f"median_delta_final_mae={md}, wilcoxon_p={pval}"
        )


if __name__ == "__main__":
    main()
