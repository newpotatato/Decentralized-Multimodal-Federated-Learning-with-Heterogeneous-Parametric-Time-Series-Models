#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """1D ablation plot for self_weight aggregated across seeds and alpha values."""
    p = Path("federated_learning/artifacts/tune_lvp_alpha_self_weight_rounds10_seed42_52_62")
    if not p.exists():
        raise FileNotFoundError(f"Expected tuning summary at {p}")
    
    summary_path = p / "lvp_alpha_self_weight_tuning_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected summary JSON at {summary_path}")
    
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = summary.get("rows", [])
    
    if not rows:
        raise RuntimeError("No rows found in tuning summary")
    
    # Aggregate by self_weight: collect all objectives for each self_weight value
    sw_to_means: Dict[float, List[float]] = {}
    for row in rows:
        sw = float(row["self_weight"])
        obj = float(row["objective"])
        if sw not in sw_to_means:
            sw_to_means[sw] = []
        sw_to_means[sw].append(obj)
    
    # Compute mean and std for each self_weight
    sw_list = sorted(sw_to_means.keys())
    means = []
    stds = []
    for sw in sw_list:
        vals = np.asarray(sw_to_means[sw], dtype=float)
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals, ddof=0)))
    
    best_idx = int(np.argmin(means))
    best_sw = sw_list[best_idx]
    best_mean = means[best_idx]
    best_std = stds[best_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    xs = sw_list
    ys = means
    ax.plot(xs, ys, "o-", linewidth=2.2, markersize=5, label="mean objective")
    ax.fill_between(xs, np.asarray(ys) - np.asarray(stds), np.asarray(ys) + np.asarray(stds), alpha=0.15, label="±1 std")
    ax.scatter([best_sw], [best_mean], s=100, marker="*", color="red", zorder=5, label=f"best self_weight={best_sw:.2f}")
    ax.set_xlabel("lvp_self_weight")
    ax.set_ylabel("Objective (mean + 0.5*std)")
    ax.set_title("Ablation for rounds=10 scenario: self_weight sweep (multiseed)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    
    out_dir = Path("federated_learning/artifacts/article_package_current_run_20260410/plots/ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    png_path = out_dir / "self_weight_ablation_multiseed.png"
    fig.savefig(png_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    
    # Report
    report_path = out_dir / "self_weight_ablation_multiseed_report.md"
    report_text = "\n".join([
        "# Self-weight Ablation (rounds=10 article scenario, multiseed)",
        "",
        f"Best self_weight: {best_sw:.6f} (objective={best_mean:.6f}, std={best_std:.6f})",
        "",
        f"Figure: {png_path}",
    ])
    report_path.write_text(report_text, encoding="utf-8")
    
    print(f"Saved: {png_path}")
    print(f"Saved: {report_path}")
    print(f"Best self_weight={best_sw:.3f}, objective={best_mean:.3f}")


if __name__ == "__main__":
    main()
