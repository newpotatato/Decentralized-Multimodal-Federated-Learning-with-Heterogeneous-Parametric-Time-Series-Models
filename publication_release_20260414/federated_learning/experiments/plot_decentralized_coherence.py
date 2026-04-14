#!/usr/bin/env python3
"""Plot synchronization coherence for decentralized aggregators over rounds.

The input is the JSON report from ``fig3_decentralized_methods.py``.
The main coherence measure is ``sync_component_l2_mean`` recorded in the
per-round history for each aggregator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


LABEL_MAP = {
    "lvp": "LVP",
    "decentralized_fedavg": "Decentralized FedAvg",
    "defta": "DeFTA",
    "balance": "BALANCE",
    "push_sum": "Push-Sum",
}


def _extract_series(payload: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    series: Dict[str, Dict[str, np.ndarray]] = {}
    for rec in payload.get("results", []):
        agg = str(rec.get("aggregator", "unknown"))
        hist = rec.get("history") or []
        xs = np.asarray([float(row.get("round", 0.0)) for row in hist], dtype=float)
        coherence = np.asarray(
            [float(row.get("sync_component_l2_mean", 0.0)) for row in hist],
            dtype=float,
        )
        comp_count = np.asarray(
            [float(row.get("sync_component_count", 0.0)) for row in hist],
            dtype=float,
        )
        series[agg] = {
            "rounds": xs,
            "coherence": coherence,
            "components": comp_count,
        }
    return series


def _plot_coherence(series: Dict[str, Dict[str, np.ndarray]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()

    order = ["lvp", "decentralized_fedavg", "defta", "balance", "push_sum"]
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for idx, agg in enumerate(order):
        if agg not in series:
            continue
        xs = series[agg]["rounds"]
        coherence = series[agg]["coherence"]
        components = series[agg]["components"]
        color = colors[idx % len(colors)] if colors else None
        ax1.plot(
            xs,
            coherence,
            marker="o",
            linewidth=2.4 if agg == "lvp" else 1.7,
            markersize=4,
            alpha=0.95 if agg == "lvp" else 0.8,
            label=LABEL_MAP.get(agg, agg),
            color=color,
        )
        ax2.plot(
            xs,
            components,
            linestyle="--",
            linewidth=1.0,
            alpha=0.25,
            color=color,
        )

    ax1.set_xlabel("Communication round")
    ax1.set_ylabel("Mean component ΔL2 (coherence measure)")
    ax2.set_ylabel("Connected components", color="tab:gray")
    ax1.set_title("Decentralized coherence by round")
    ax1.grid(True, alpha=0.3)

    ax2.tick_params(axis="y", colors="tab:gray")
    ax2.set_ylim(bottom=0)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="best", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot decentralized coherence over rounds.")
    parser.add_argument(
        "--input-json",
        type=str,
        required=True,
        help="Path to fig3_decentralized_methods.json",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory for the coherence plot and report",
    )
    args = parser.parse_args()

    in_path = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    series = _extract_series(payload)
    if not series:
        raise RuntimeError("No aggregator histories found in input JSON")

    fig_path = out_dir / "fig3_decentralized_coherence.png"
    _plot_coherence(series, fig_path)

    lines: List[str] = [
        "# Decentralized Coherence Report",
        "",
        f"Input: {in_path}",
        "",
        "## Final round coherence",
    ]
    ranked = []
    for agg, rec in series.items():
        coherence = rec["coherence"]
        if coherence.size == 0:
            continue
        ranked.append((agg, float(coherence[-1]), float(np.mean(coherence))))
    ranked.sort(key=lambda x: x[1])
    for agg, last_val, mean_val in ranked:
        lines.append(f"- {agg}: last={last_val:.6f}, mean={mean_val:.6f}")
    lines.extend([
        "",
        "## Notes",
        "- Lower mean component ΔL2 means the synchronized states moved less inside each connected component.",
        "- Dashed lines on the twin axis show the connected-component count for context.",
    ])
    report_path = out_dir / "fig3_decentralized_coherence_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {fig_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()