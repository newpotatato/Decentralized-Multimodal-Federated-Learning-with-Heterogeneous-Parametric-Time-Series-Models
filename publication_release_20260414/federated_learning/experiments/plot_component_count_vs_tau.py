#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "core"))
sys.path.insert(0, str(_FL / "data_loaders"))

from data_utils import build_client_information_profiles, build_clients_from_mcc, load_mcc_series
from decentralized_lvp import build_neighbor_graph
from run_real_experiments import _build_exogenous, _components_undirected


def _parse_grid(s: str) -> List[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("tau grid is empty")
    return vals


def main() -> None:
    p = argparse.ArgumentParser(description="Plot number of connected components vs tau")
    p.add_argument(
        "--pkg-dir",
        type=str,
        default=str(_FL / "artifacts" / "article_package_current_run_20260410" / "experiment_with_more_rounds"),
    )
    p.add_argument("--tau-grid", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--column-partition", type=str, default="contiguous", choices=["contiguous", "strided"])
    args = p.parse_args()

    pkg = Path(args.pkg_dir)
    out_dir = pkg / "plots" / "coherence" / "components"
    out_dir.mkdir(parents=True, exist_ok=True)

    tau_grid = _parse_grid(args.tau_grid)

    base = _FL.parent
    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=True)
    clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=int(args.n_clients),
        column_partition=str(args.column_partition),
    )
    profiles = build_client_information_profiles(clients, "mcc")

    n_components = []
    largest_sizes = []

    for tau in tau_grid:
        neighbors, _ = build_neighbor_graph(profiles, tau=tau, similarity_mode="jaccard")
        comps = _components_undirected(neighbors)
        n_components.append(len(comps))
        largest_sizes.append(max((len(c) for c in comps), default=0))

    fig, ax1 = plt.subplots(figsize=(10.2, 5.8))
    ax2 = ax1.twinx()

    ax1.plot(tau_grid, n_components, "o-", linewidth=2.2, markersize=5, color="#1f77b4", label="Connected components")
    ax2.plot(tau_grid, largest_sizes, "s--", linewidth=1.8, markersize=4, color="#d62728", alpha=0.85, label="Largest component size")

    ax1.set_xlabel("similarity_tau")
    ax1.set_ylabel("Number of connected components", color="#1f77b4")
    ax2.set_ylabel("Largest component size", color="#d62728")
    ax1.set_title("Topology split vs tau (Jaccard graph)")
    ax1.grid(True, alpha=0.3)

    ax1.tick_params(axis="y", colors="#1f77b4")
    ax2.tick_params(axis="y", colors="#d62728")

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="best")

    plt.tight_layout()
    fig_path = out_dir / "component_count_vs_tau.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    payload = {
        "tau_grid": tau_grid,
        "n_components": n_components,
        "largest_component_size": largest_sizes,
        "n_clients": int(args.n_clients),
        "column_partition": str(args.column_partition),
        "figure": str(fig_path),
    }
    summary_path = out_dir / "component_count_vs_tau_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_lines = [
        "# Component count vs tau",
        "",
        f"Clients: {args.n_clients}",
        f"Partition: {args.column_partition}",
        "",
        "## tau -> connected components, largest component",
    ]
    for tau, nc, ls in zip(tau_grid, n_components, largest_sizes):
        report_lines.append(f"- tau={tau:.2f}: components={nc}, largest={ls}")
    report_lines += ["", f"Figure: {fig_path}"]
    (out_dir / "component_count_vs_tau_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved: {fig_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
