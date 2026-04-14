#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np


def _parse_grid(s: str) -> List[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("tau grid is empty")
    return vals


def main() -> None:
    p = argparse.ArgumentParser(description="Draw full readable component-vs-tau graph from summary JSON")
    p.add_argument(
        "--summary-json",
        type=str,
        default="federated_learning/artifacts/article_package_current_run_20260410/experiment_with_more_rounds/plots/coherence/components/component_count_vs_tau_summary.json",
    )
    p.add_argument("--tau-current", type=float, default=0.35)
    args = p.parse_args()

    summary_path = Path(args.summary_json)
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary: {summary_path}")

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    tau = np.asarray(payload["tau_grid"], dtype=float)
    n_comp = np.asarray(payload["n_components"], dtype=float)
    largest = np.asarray(payload["largest_component_size"], dtype=float)
    n_clients = int(payload.get("n_clients", 20))

    # First tau where graph starts splitting into >1 components.
    split_tau = None
    split_idx = np.where(n_comp > 1)[0]
    if split_idx.size:
        split_tau = float(tau[int(split_idx[0])])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 8.0), sharex=True)

    ax1.plot(tau, n_comp, "o-", linewidth=2.5, markersize=6, color="#1f77b4")
    ax1.set_ylabel("Connected components")
    ax1.set_ylim(0.5, max(float(np.max(n_comp)) + 0.8, 2.0))
    ax1.grid(True, alpha=0.3)

    ax2.plot(tau, largest, "s--", linewidth=2.3, markersize=5, color="#d62728")
    ax2.set_xlabel("similarity_tau")
    ax2.set_ylabel("Largest component size")
    ax2.set_ylim(0.5, n_clients + 1)
    ax2.grid(True, alpha=0.3)

    for ax in (ax1, ax2):
        ax.axvline(float(args.tau_current), color="black", linestyle=":", linewidth=1.6)

    ax1.text(float(args.tau_current) + 0.005, ax1.get_ylim()[1] * 0.93, f"tau_current={args.tau_current:.2f}", fontsize=9)
    if split_tau is not None:
        ax1.axvline(split_tau, color="#2ca02c", linestyle="--", linewidth=1.4)
        ax1.text(split_tau + 0.005, ax1.get_ylim()[1] * 0.84, f"first split tau={split_tau:.2f}", fontsize=9, color="#2ca02c")

    # Add endpoint labels for readability.
    ax1.annotate(f"{int(n_comp[-1])}", (tau[-1], n_comp[-1]), xytext=(8, 6), textcoords="offset points", fontsize=9)
    ax2.annotate(f"{int(largest[-1])}", (tau[-1], largest[-1]), xytext=(8, 6), textcoords="offset points", fontsize=9)

    fig.suptitle("Full topology diagnostics vs tau (readable view)", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_dir = summary_path.parent
    fig_path = out_dir / "component_count_vs_tau_full.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report_path = out_dir / "component_count_vs_tau_full_report.md"
    lines = [
        "# Full Component-vs-Tau Graph",
        "",
        f"Input summary: {summary_path}",
        f"tau_current: {args.tau_current:.2f}",
        f"first_split_tau: {split_tau if split_tau is not None else 'none'}",
        f"figure: {fig_path}",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {fig_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
