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

from data_utils import build_client_information_profiles, build_clients_from_mcc, load_mcc_series
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
    return [frozenset(p) | {f"sync_topic_{idx % n_groups}"} for idx, p in enumerate(profiles)]


def _final_mae(exp: Dict[str, Any]) -> float:
    hist = exp.get("history") or []
    if not hist:
        return float("inf")
    return float(hist[-1].get("network_mae", float("inf")))


def _parse_float_list(raw: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty float list")
    return vals


def _parse_int_list(raw: str) -> List[int]:
    vals = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("Empty int list")
    return vals


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


def main() -> None:
    p = argparse.ArgumentParser(description="Select best topology per decentralized method and run multiseed.")
    p.add_argument("--base-path", type=str, default=str(_FL.parent))
    p.add_argument("--out-dir", type=str, default=str(_FL / "artifacts" / "best_topology_per_method"))
    p.add_argument("--model", type=str, default="DynamicLinearModel", choices=sorted(MODEL_REGISTRY.keys()))
    p.add_argument("--methods", type=str, default=",".join(DECENTRALIZED_AGGS))
    p.add_argument("--modes", type=str, default="jaccard,jaccard_cosine_hybrid")
    p.add_argument("--tau-grid", type=str, default="0.25,0.35,0.45,0.57,0.69,0.79")
    p.add_argument("--seeds", type=str, default="42,52,62")
    p.add_argument("--selection-seed", type=int, default=42)
    p.add_argument("--selection-rounds", type=int, default=6)
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--column-partition", type=str, default="contiguous", choices=["contiguous", "strided"])
    p.add_argument("--malicious-frac", type=float, default=0.25)
    p.add_argument("--attack-strategy", type=str, default="noise_colluded", choices=["label_flip", "noise", "noise_colluded", "random"])
    p.add_argument("--attack-scale", type=float, default=5.0)
    p.add_argument("--lambda-jaccard", type=float, default=0.5)
    p.add_argument("--tau-cos-min", type=float, default=-1.0)
    p.add_argument("--lvp-alpha", type=float, default=0.6)
    p.add_argument("--network-eval-mode", type=str, default="proxy", choices=["proxy", "refit"])
    p.add_argument("--no-reuters", action="store_true")
    p.add_argument("--local-fit-maxiter", type=int, default=10)
    p.add_argument("--eval-fit-maxiter", type=int, default=0)
    p.add_argument("--selection-local-fit-maxiter", type=int, default=4)
    p.add_argument("--selection-eval-fit-maxiter", type=int, default=0)
    args = p.parse_args()

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    tau_grid = _parse_float_list(args.tau_grid)
    seeds = _parse_int_list(args.seeds)

    for m in methods:
        if m not in DECENTRALIZED_AGGS:
            raise ValueError(f"Unsupported method for this runner: {m}")
    for mode in modes:
        if mode not in ("jaccard", "jaccard_cosine_hybrid"):
            raise ValueError(f"Unsupported similarity mode: {mode}")

    base = Path(args.base_path).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[setup] out_dir={out_dir}", flush=True)

    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=not args.no_reuters)
    clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=args.n_clients,
        column_partition=args.column_partition,
    )
    if len(clients) < 2:
        raise RuntimeError("Need at least 2 clients.")
    profiles = _topic_profiles(clients)

    model_name = args.model
    model_cls = MODEL_REGISTRY[model_name]
    print(
        f"[setup] model={model_name}, methods={methods}, modes={modes}, tau_grid={tau_grid}, selection_seed={args.selection_seed}, selection_rounds={args.selection_rounds}",
        flush=True,
    )

    selection_rows: List[Dict[str, Any]] = []
    best_cfg: Dict[str, Dict[str, Any]] = {}
    checkpoint_path = out_dir / "selection_rows_checkpoint.json"

    # Resume topology search from checkpoint when available.
    if checkpoint_path.exists():
        try:
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                selection_rows = loaded
                print(f"[resume] loaded {len(selection_rows)} selection rows", flush=True)
        except Exception as exc:
            print(f"[resume] failed to read checkpoint: {exc}", flush=True)

    existing_vals: Dict[Tuple[str, str, float, int], float] = {}
    for row in selection_rows:
        try:
            key = (
                str(row["method"]),
                str(row["mode"]),
                float(row["tau"]),
                int(row["seed"]),
            )
            existing_vals[key] = float(row["final_network_mae"])
        except Exception:
            continue

    # Topology search on one seed for each method.
    for method in methods:
        print(f"[search] method={method}", flush=True)
        best_val = float("inf")
        best: Dict[str, Any] = {}
        # Seed best from already computed rows for this method.
        for row in selection_rows:
            if str(row.get("method")) != method:
                continue
            if int(row.get("seed", -1)) != int(args.selection_seed):
                continue
            if str(row.get("mode")) not in modes:
                continue
            if float(row.get("tau")) not in tau_grid:
                continue
            val = float(row.get("final_network_mae", float("inf")))
            if val < best_val:
                best_val = val
                best = {
                    "method": method,
                    "mode": str(row["mode"]),
                    "tau": float(row["tau"]),
                    "seed": int(args.selection_seed),
                    "final_network_mae": val,
                }
        for mode in modes:
            for tau in tau_grid:
                print(f"  [candidate] method={method} mode={mode} tau={tau:.4f}", flush=True)
                key = (method, mode, float(tau), int(args.selection_seed))
                if key in existing_vals:
                    val = float(existing_vals[key])
                    row = {
                        "method": method,
                        "mode": mode,
                        "tau": float(tau),
                        "seed": int(args.selection_seed),
                        "final_network_mae": float(val),
                    }
                    print(
                        f"  [candidate-skip] method={method} mode={mode} tau={tau:.4f} mae={val:.6f}",
                        flush=True,
                    )
                else:
                    exp = run_one_model(
                        model_name,
                        model_cls,
                        clients,
                        profiles,
                        aggregator=method,
                        rounds=args.selection_rounds,
                        local_epochs=args.local_epochs,
                        malicious_frac=args.malicious_frac,
                        seed=int(args.selection_seed),
                        attack_strategy=args.attack_strategy,
                        attack_scale=args.attack_scale,
                        similarity_tau=float(tau),
                        similarity_mode=mode,
                        lambda_jaccard=args.lambda_jaccard,
                        tau_cos_min=args.tau_cos_min,
                        lvp_alpha=(float(args.lvp_alpha) if method == "lvp" else None),
                        lvp_self_weight=0.0,
                        krum_f=-1,
                        local_fit_maxiter=args.selection_local_fit_maxiter,
                        eval_fit_maxiter=args.selection_eval_fit_maxiter,
                        strict_errors=True,
                        network_eval_mode=args.network_eval_mode,
                    )
                    val = _final_mae(exp)
                    row = {
                        "method": method,
                        "mode": mode,
                        "tau": float(tau),
                        "seed": int(args.selection_seed),
                        "final_network_mae": float(val),
                    }
                    selection_rows.append(row)
                    existing_vals[key] = float(val)
                if val < best_val:
                    best_val = val
                    best = row
                    print(
                        f"  [best-update] method={method} mode={mode} tau={tau:.4f} mae={val:.6f}",
                        flush=True,
                    )
                checkpoint_path.write_text(
                    json.dumps(selection_rows, indent=2, default=str), encoding="utf-8"
                )
        if not best:
            raise RuntimeError(f"No valid candidate found for method={method}")
        best_cfg[method] = best
        print(
            f"[search-done] method={method} best_mode={best['mode']} best_tau={float(best['tau']):.4f} best_mae={float(best['final_network_mae']):.6f}",
            flush=True,
        )

    # Evaluate multiseed with per-method best topology.
    raw_seed_dirs: List[Path] = []
    for seed in seeds:
        print(f"[eval] seed={seed}", flush=True)
        seed_dir = out_dir / "raw" / "seeds" / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        raw_seed_dirs.append(seed_dir)

        results: List[Dict[str, Any]] = []
        for method in methods:
            cfg = best_cfg[method]
            print(
                f"  [run] seed={seed} method={method} mode={cfg['mode']} tau={float(cfg['tau']):.4f}",
                flush=True,
            )
            exp = run_one_model(
                model_name,
                model_cls,
                clients,
                profiles,
                aggregator=method,
                rounds=args.rounds,
                local_epochs=args.local_epochs,
                malicious_frac=args.malicious_frac,
                seed=int(seed),
                attack_strategy=args.attack_strategy,
                attack_scale=args.attack_scale,
                similarity_tau=float(cfg["tau"]),
                similarity_mode=str(cfg["mode"]),
                lambda_jaccard=args.lambda_jaccard,
                tau_cos_min=args.tau_cos_min,
                lvp_alpha=(float(args.lvp_alpha) if method == "lvp" else None),
                lvp_self_weight=0.0,
                krum_f=-1,
                local_fit_maxiter=args.local_fit_maxiter,
                eval_fit_maxiter=args.eval_fit_maxiter,
                strict_errors=True,
                network_eval_mode=args.network_eval_mode,
            )
            results.append(exp)

        payload = {
            "best_model": model_name,
            "scenario": {
                "column_partition": args.column_partition,
                "malicious_frac": args.malicious_frac,
                "attack_strategy": args.attack_strategy,
                "attack_scale": args.attack_scale,
                "rounds": args.rounds,
                "local_epochs": args.local_epochs,
                "seed": int(seed),
                "network_eval_mode": args.network_eval_mode,
                "topology_selection_seed": int(args.selection_seed),
            },
            "best_topology_by_aggregator": best_cfg,
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
        }
        (seed_dir / "fig3_decentralized_methods.json").write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )

        ordered = sorted(payload["final_network_mae_by_aggregator"].items(), key=lambda kv: kv[1])
        lines = [
            "# Best Topology Per Method (Single Seed Run)",
            "",
            f"Model: {model_name}",
            f"Seed: {seed}",
            "",
            "## Best topology map",
        ]
        for method in methods:
            cfg = best_cfg[method]
            lines.append(
                f"- {method}: mode={cfg['mode']}, tau={float(cfg['tau']):.4f}, selection_mae={float(cfg['final_network_mae']):.6f}"
            )
        lines.extend(["", "## Final MAE by method"])
        for name, val in ordered:
            lines.append(f"- {name}: {float(val):.6f}")
        (seed_dir / "fig3_decentralized_methods_report.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"[eval-done] seed={seed} -> {seed_dir}", flush=True)

    # Aggregate boxplot over seeds.
    box_dir = out_dir / "plots" / "boxplots"
    box_dir.mkdir(parents=True, exist_ok=True)
    _run_boxplot(raw_seed_dirs, box_dir)
    print(f"[boxplot-done] {box_dir}", flush=True)

    # Save selection and summary artifacts.
    summary = {
        "model": model_name,
        "methods": methods,
        "modes": modes,
        "tau_grid": tau_grid,
        "selection_seed": int(args.selection_seed),
        "selection_rounds": int(args.selection_rounds),
        "seeds": seeds,
        "scenario": {
            "column_partition": args.column_partition,
            "malicious_frac": args.malicious_frac,
            "attack_strategy": args.attack_strategy,
            "attack_scale": args.attack_scale,
            "rounds": args.rounds,
            "local_epochs": args.local_epochs,
            "network_eval_mode": args.network_eval_mode,
        },
        "best_topology_by_method": best_cfg,
        "selection_rows": selection_rows,
        "boxplot": str(box_dir / "fig3_boxplot_single_multiseed.png"),
    }
    (out_dir / "best_topology_selection_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# Best Topology Per Method - Multiseed",
        "",
        f"Model: {model_name}",
        f"Methods: {', '.join(methods)}",
        f"Selection seed: {args.selection_seed}",
        f"Eval seeds: {', '.join(str(s) for s in seeds)}",
        "",
        "## Selected topology per method",
        "",
        "| Method | Mode | Tau | Selection MAE |",
        "|---|---:|---:|---:|",
    ]
    for method in methods:
        cfg = best_cfg[method]
        lines.append(
            f"| {method} | {cfg['mode']} | {float(cfg['tau']):.4f} | {float(cfg['final_network_mae']):.6f} |"
        )
    lines.extend([
        "",
        f"Boxplot: {box_dir / 'fig3_boxplot_single_multiseed.png'}",
        f"Boxplot report: {box_dir / 'fig3_boxplot_with_vs_without_fedavg_multiseed_report.md'}",
    ])
    (out_dir / "best_topology_per_method_report.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"Done. Outputs in: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
