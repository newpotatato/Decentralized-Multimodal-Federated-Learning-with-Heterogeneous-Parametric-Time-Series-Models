#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "core"))
sys.path.insert(0, str(_FL / "data_loaders"))

from data_utils import build_client_information_profiles, build_clients_from_mcc, load_mcc_series  # noqa: E402
from run_real_experiments import MODEL_REGISTRY, _build_exogenous, run_one_model  # noqa: E402

DEFAULT_METHOD_TOPOLOGIES = {
    "lvp": "hybrid",
    "decentralized_fedavg": "ring",
    "defta": "star",
    "balance": "line",
    "push_sum": "ring",
}

VALID_STRUCTURAL_TOPOLOGIES = {"ring", "star", "complete", "line"}


def _parse_csv(text: str) -> List[str]:
    return [x.strip() for x in (text or "").split(",") if x.strip()]


def _parse_float_csv(text: str) -> List[float]:
    vals = [float(x.strip()) for x in (text or "").split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one float value")
    return vals


def _parse_int_csv(text: str) -> List[int]:
    vals = [int(x.strip()) for x in (text or "").split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected at least one integer value")
    return vals


def _topic_profiles(clients: List, n_groups: int = 4) -> List:
    profiles = build_client_information_profiles(clients, "mcc")
    return [frozenset(p) | {f"sync_topic_{idx % n_groups}"} for idx, p in enumerate(profiles)]


def _final_mae(exp: Dict[str, Any]) -> float:
    hist = exp.get("history") or []
    if not hist:
        return float("inf")
    return float(hist[-1].get("network_mae", float("inf")))


def _aggregate_histories(exps: List[Dict[str, Any]]) -> Dict[str, Any]:
    min_len = min(len(e.get("history") or []) for e in exps)
    rounds = list(range(1, min_len + 1))
    matrix = np.asarray(
        [[float(e["history"][r - 1]["network_mae"]) for r in rounds] for e in exps],
        dtype=float,
    )
    return {
        "round": rounds,
        "mean_network_mae": np.mean(matrix, axis=0).tolist(),
        "std_network_mae": np.std(matrix, axis=0).tolist(),
    }


def _run_boxplot(seed_dirs: List[Path], out_dir: Path) -> None:
    cmd = [
        sys.executable,
        str(_EXP / "fig3_boxplot_with_without_fedavg_multiseed.py"),
        "--input-dirs",
        *[str(p) for p in seed_dirs],
        "--out-dir",
        str(out_dir),
        "--single-panel",
        "--show-zoom-inset",
    ]
    subprocess.run(cmd, check=True)


def _parse_topology_map(text: str) -> Dict[str, str]:
    mapping = dict(DEFAULT_METHOD_TOPOLOGIES)
    for chunk in _parse_csv(text):
        if ":" not in chunk:
            raise ValueError(f"Invalid topology mapping: {chunk!r}")
        method, topo = [part.strip().lower() for part in chunk.split(":", 1)]
        mapping[method] = topo
    return mapping


def _resolve_policy(
    method: str,
    topology_name: str,
    lvp_tau: float,
    lambda_jaccard: float = 0.6,
    lvp_alpha: float = 0.6,
) -> Dict[str, Any]:
    topo = (topology_name or "").strip().lower()
    if topo == "hybrid":
        return {
            "similarity_mode": "jaccard_cosine_hybrid",
            "similarity_tau": float(lvp_tau),
            "topology_mode": "similarity",
            "lambda_jaccard": float(lambda_jaccard),
            "tau_cos_min": 0.15,
            "lvp_alpha": float(lvp_alpha) if method == "lvp" else None,
            "lvp_self_weight": 0.0,
        }
    if topo == "jaccard":
        return {
            "similarity_mode": "jaccard",
            "similarity_tau": float(lvp_tau),
            "topology_mode": "similarity",
            "lambda_jaccard": float(lambda_jaccard),
            "tau_cos_min": -1.0,
            "lvp_alpha": None,
            "lvp_self_weight": 0.0,
        }
    if topo in VALID_STRUCTURAL_TOPOLOGIES:
        return {
            "similarity_mode": "jaccard",
            "similarity_tau": float(lvp_tau),
            "topology_mode": topo,
            "lambda_jaccard": float(lambda_jaccard),
            "tau_cos_min": -1.0,
            "lvp_alpha": None if method != "lvp" else float(lvp_alpha),
            "lvp_self_weight": 0.0,
        }
    raise ValueError(f"Unknown topology policy for {method}: {topology_name!r}")


def main() -> None:
    p = argparse.ArgumentParser(description="Compare methods on distinct topology policies vs LVP hybrid topology")
    p.add_argument("--base-path", type=str, default=str(_FL.parent))
    p.add_argument("--out-dir", type=str, default=str(_FL / "artifacts" / "topology_policy_comparison"))
    p.add_argument("--model", type=str, default="DynamicLinearModel", choices=sorted(MODEL_REGISTRY.keys()))
    p.add_argument("--methods", type=str, default="lvp,decentralized_fedavg,defta,balance,push_sum")
    p.add_argument("--seeds", type=str, default="42,52,62")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--local-fit-maxiter", type=int, default=10)
    p.add_argument("--eval-fit-maxiter", type=int, default=0)
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--column-partition", type=str, default="contiguous", choices=["contiguous", "strided", "random", "random_strided"])
    p.add_argument("--malicious-frac", type=float, default=0.25)
    p.add_argument("--attack-strategy", type=str, default="noise_colluded", choices=["label_flip", "noise", "noise_colluded", "random"])
    p.add_argument("--attack-scale", type=float, default=5.0)
    p.add_argument("--network-eval-mode", type=str, default="proxy", choices=["proxy", "refit"])
    p.add_argument("--lvp-tau", type=float, default=0.79)
    p.add_argument("--lambda-jaccard", type=float, default=0.6)
    p.add_argument("--lvp-alpha", type=float, default=0.6)
    p.add_argument("--topology-map", type=str, default="")
    p.add_argument("--sync-topic-groups", type=int, default=4)
    p.add_argument("--no-reuters", action="store_true", default=False)
    args = p.parse_args()

    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.base_path).resolve()

    methods = _parse_csv(args.methods)
    seeds = _parse_int_csv(args.seeds)
    topo_map = _parse_topology_map(args.topology_map)

    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=not args.no_reuters)
    ModelClass = MODEL_REGISTRY[args.model]

    print(f"[setup] out_dir={out_dir}", flush=True)
    print(f"[setup] methods={methods}, seeds={seeds}, lvp_tau={args.lvp_tau}", flush=True)
    print(f"[setup] topology_map={topo_map}", flush=True)

    scenario_meta = {
        "column_partition": args.column_partition,
        "malicious_frac": args.malicious_frac,
        "attack_strategy": args.attack_strategy,
        "attack_scale": args.attack_scale,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "network_eval_mode": args.network_eval_mode,
        "lvp_tau": float(args.lvp_tau),
    }

    raw_seed_dirs: List[Path] = []
    best_policy_rows: List[Dict[str, Any]] = []

    for seed in seeds:
        print(f"[eval] seed={seed}", flush=True)
        seed_dir = out_dir / "raw" / "seeds" / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        raw_seed_dirs.append(seed_dir)

        clients = build_clients_from_mcc(
            mcc_df,
            exog,
            n_clients=args.n_clients,
            column_partition=args.column_partition,
            partition_seed=seed,
        )
        profiles = _topic_profiles(clients, n_groups=args.sync_topic_groups)

        results: List[Dict[str, Any]] = []
        for method in methods:
            if method not in topo_map:
                raise ValueError(f"Missing topology policy for method {method!r}")
            policy = _resolve_policy(
                method, topo_map[method], float(args.lvp_tau),
                lambda_jaccard=args.lambda_jaccard,
                lvp_alpha=args.lvp_alpha,
            )
            print(
                f"  [run] method={method} topology={topo_map[method]} mode={policy['similarity_mode']} tau={policy['similarity_tau']:.4f}",
                flush=True,
            )
            exp = run_one_model(
                args.model,
                ModelClass,
                clients,
                profiles,
                aggregator=method,
                rounds=args.rounds,
                local_epochs=args.local_epochs,
                malicious_frac=args.malicious_frac,
                seed=seed,
                attack_strategy=args.attack_strategy,
                attack_scale=args.attack_scale,
                similarity_tau=float(policy["similarity_tau"]),
                similarity_mode=str(policy["similarity_mode"]),
                lambda_jaccard=float(policy["lambda_jaccard"]),
                tau_cos_min=float(policy["tau_cos_min"]),
                topology_mode=str(policy["topology_mode"]),
                lvp_alpha=policy["lvp_alpha"],
                lvp_self_weight=float(policy["lvp_self_weight"]),
                krum_f=-1,
                local_fit_maxiter=args.local_fit_maxiter,
                eval_fit_maxiter=args.eval_fit_maxiter,
                strict_errors=True,
                network_eval_mode=args.network_eval_mode,
            )
            results.append(exp)
            best_policy_rows.append(
                {
                    "seed": seed,
                    "method": method,
                    "topology": topo_map[method],
                    "final_network_mae": _final_mae(exp),
                }
            )

        payload = {
            "model": args.model,
            "scenario": {**scenario_meta, "seed": seed},
            "topology_map": topo_map,
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
            "final_network_mae_by_aggregator": {r["aggregator"]: _final_mae(r) for r in results},
        }
        (seed_dir / "fig3_decentralized_methods.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

        ordered = sorted(payload["final_network_mae_by_aggregator"].items(), key=lambda kv: kv[1])
        lines = [
            "# Topology Policy Comparison (Single Seed Run)",
            "",
            f"Model: {args.model}",
            f"Seed: {seed}",
            "",
            "## Topology map",
        ]
        for method in methods:
            lines.append(f"- {method}: {topo_map[method]}")
        lines.extend(["", "## Final MAE by method"])
        for name, val in ordered:
            lines.append(f"- {name}: {float(val):.6f}")
        (seed_dir / "fig3_decentralized_methods_report.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"[eval-done] seed={seed} -> {seed_dir}", flush=True)

    box_dir = out_dir / "plots" / "boxplots"
    box_dir.mkdir(parents=True, exist_ok=True)
    _run_boxplot(raw_seed_dirs, box_dir)

    summary = {
        "model": args.model,
        "methods": methods,
        "seeds": seeds,
        "scenario": scenario_meta,
        "topology_map": topo_map,
        "lvp_tau": float(args.lvp_tau),
        "final_metrics": {},
        "raw_seed_rows": best_policy_rows,
        "boxplot": str(box_dir / "fig3_boxplot_single_multiseed.png"),
    }

    # Summarize the chosen topology policies across seeds.
    final_by_method: Dict[str, List[float]] = {m: [] for m in methods}
    for row in best_policy_rows:
        final_by_method[row["method"]].append(float(row["final_network_mae"]))
    summary["final_metrics"] = {
        method: {
            "final_mae_mean": float(np.mean(vals)) if vals else float("inf"),
            "final_mae_std": float(np.std(vals)) if vals else float("inf"),
        }
        for method, vals in final_by_method.items()
    }
    (out_dir / "topology_policy_comparison_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    report_lines = [
        "# Method Topology Policy Comparison",
        "",
        f"Model: {args.model}",
        f"Seeds: {', '.join(str(s) for s in seeds)}",
        f"LVP tau: {args.lvp_tau}",
        "",
        "## Topology map",
    ]
    for method in methods:
        report_lines.append(f"- {method}: {topo_map[method]}")
    report_lines.extend([
        "",
        "## Final MAE mean +- std",
    ])
    for method in methods:
        metric = summary["final_metrics"][method]
        report_lines.append(
            f"- {method}: {metric['final_mae_mean']:.6f} +- {metric['final_mae_std']:.6f}"
        )
    report_lines.extend(
        [
            "",
            f"Boxplot: {box_dir / 'fig3_boxplot_single_multiseed.png'}",
            f"Boxplot report: {box_dir / 'fig3_boxplot_with_vs_without_fedavg_multiseed_report.md'}",
        ]
    )
    (out_dir / "topology_policy_comparison_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Done. Outputs in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
