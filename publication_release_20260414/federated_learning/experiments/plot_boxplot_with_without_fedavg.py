#!/usr/bin/env python3
"""Build boxplots for one experiment in two views: with FedAvg and without FedAvg.

Samples are per-round network MAE values from each aggregator history.
Input JSON: output of fig3_decentralized_methods.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


LABEL_MAP = {
    "lvp": "LVP",
    "decentralized_fedavg": "Decentralized FedAvg",
    "defta": "DeFTA",
    "balance": "BALANCE",
    "push_sum": "Push-Sum",
}

ORDER = ["lvp", "decentralized_fedavg", "defta", "balance", "push_sum"]


def _load_series(payload: Dict[str, Any]) -> Dict[str, List[float]]:
    out: Dict[str, List[float]] = {}
    for rec in payload.get("results", []):
        agg = str(rec.get("aggregator", "unknown"))
        hist = rec.get("history") or []
        vals = [float(r.get("network_mae", np.nan)) for r in hist]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out[agg] = vals
    return out


def _draw_panel(
    ax: Axes,
    by_agg: Dict[str, List[float]],
    include_fedavg: bool,
    title: str,
    style: str,
) -> None:
    if include_fedavg:
        aggs = [a for a in ORDER if a in by_agg]
    else:
        aggs = [a for a in ORDER if a in by_agg and a != "decentralized_fedavg"]

    data = [by_agg[a] for a in aggs]
    labels = [LABEL_MAP.get(a, a) for a in aggs]

    if style == "legacy":
        ax.boxplot(data, tick_labels=labels, patch_artist=True)
    else:
        # Use percentile whiskers and hide default fliers to avoid clutter when IQR is tiny.
        bp = ax.boxplot(
            data,
            tick_labels=labels,
            patch_artist=True,
            showfliers=False,
            whis=(5, 95),
            showmeans=True,
            meanprops={"marker": "D", "markerfacecolor": "black", "markeredgecolor": "black", "markersize": 4},
            medianprops={"color": "#1a1a1a", "linewidth": 2.0},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
        )
        box_colors = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
        for i, patch in enumerate(bp.get("boxes", [])):
            patch.set_alpha(0.45)
            patch.set_facecolor(box_colors[i % len(box_colors)])

        # Add light jittered points so the distribution shape remains visible.
        rng = np.random.default_rng(42)
        for i, vals in enumerate(data, start=1):
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            jitter = rng.uniform(-0.08, 0.08, size=arr.size)
            ax.scatter(
                np.full(arr.size, i) + jitter,
                arr,
                s=12,
                alpha=0.35,
                color="tab:blue",
                edgecolors="none",
                zorder=3,
            )
    ax.set_title(title)
    ax.set_ylabel("Network MAE (over rounds)")
    ax.tick_params(axis="x", labelrotation=18)
    ax.grid(True, axis="y", alpha=0.3)

    # Independent panel scaling so one extreme method does not flatten the other panel.
    flat = np.asarray([v for grp in data for v in grp], dtype=float)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return

    if include_fedavg:
        # FedAvg may include extreme spikes; log scale keeps all boxes visible.
        positive = flat[flat > 0]
        if positive.size:
            ax.set_yscale("log")
            lo = float(np.min(positive)) * 0.95
            hi = float(np.max(positive)) * 1.05
            if hi > lo > 0:
                ax.set_ylim(lo, hi)
    else:
        lo = float(np.min(flat))
        hi = float(np.max(flat))
        pad = max((hi - lo) * 0.08, 1.0)
        ax.set_ylim(lo - pad, hi + pad)


def main() -> None:
    p = argparse.ArgumentParser(description="Boxplot for previous experiment: with and without FedAvg")
    p.add_argument("--input-json", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument(
        "--panel",
        type=str,
        default="both",
        choices=["both", "with", "without"],
        help="Render both panels or only one panel (with/without FedAvg)",
    )
    p.add_argument(
        "--style",
        type=str,
        default="improved",
        choices=["legacy", "improved"],
        help="legacy = original look, improved = cleaner styling",
    )
    args = p.parse_args()

    in_path = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    by_agg = _load_series(payload)
    if not by_agg:
        raise RuntimeError("No usable aggregator series found in input JSON")

    if args.panel == "both":
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=False)
        _draw_panel(axes[0], by_agg, include_fedavg=True, title="With FedAvg", style=args.style)
        _draw_panel(axes[1], by_agg, include_fedavg=False, title="Without FedAvg", style=args.style)
        suffix = "legacy" if args.style == "legacy" else "improved"
        fig_path = out_dir / f"fig3_boxplot_with_vs_without_fedavg_{suffix}.png"
    elif args.panel == "with":
        fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.8), sharey=False)
        _draw_panel(ax, by_agg, include_fedavg=True, title="With FedAvg", style=args.style)
        suffix = "legacy" if args.style == "legacy" else "improved"
        fig_path = out_dir / f"fig3_boxplot_with_fedavg_only_{suffix}.png"
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.8), sharey=False)
        _draw_panel(ax, by_agg, include_fedavg=False, title="Without FedAvg", style=args.style)
        suffix = "legacy" if args.style == "legacy" else "improved"
        fig_path = out_dir / f"fig3_boxplot_without_fedavg_only_{suffix}.png"

    plt.tight_layout()
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "# Boxplot with vs without FedAvg",
        "",
        f"Input: {in_path}",
        "",
        "Samples: per-round network MAE values for each method.",
        "",
        "## Mean over rounds by method",
    ]
    for agg in ORDER:
        if agg not in by_agg:
            continue
        arr = np.asarray(by_agg[agg], dtype=float)
        lines.append(f"- {LABEL_MAP.get(agg, agg)}: mean={float(np.mean(arr)):.6f}, std={float(np.std(arr, ddof=0)):.6f}")

    report_path = out_dir / "fig3_boxplot_with_vs_without_fedavg_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {fig_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()