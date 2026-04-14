#!/usr/bin/env python3
"""Build two decentralized comparison plots: with FedAvg and without FedAvg.

Input: JSON output from fig3_decentralized_methods.py
Output:
- fig3_decentralized_methods_network_mae_with_fedavg.png
- fig3_decentralized_methods_network_mae_without_fedavg.png
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

ORDER = ["lvp", "decentralized_fedavg", "defta", "balance", "push_sum"]


def _extract_series(payload: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for rec in payload.get("results", []):
        agg = str(rec.get("aggregator", "unknown"))
        hist = rec.get("history") or []
        xs = np.asarray([int(r.get("round", 0)) for r in hist], dtype=int)
        ys = np.asarray([float(r.get("network_mae", np.nan)) for r in hist], dtype=float)
        out[agg] = {"x": xs, "y": ys}
    return out


def _plot(
    series: Dict[str, Dict[str, np.ndarray]],
    title: str,
    out_path: Path,
    y_max: float | None = None,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for agg in ORDER:
        if agg not in series:
            continue
        xs = series[agg]["x"]
        ys = series[agg]["y"]
        lw = 3.0 if agg == "lvp" else 1.8
        alpha = 1.0 if agg == "lvp" else 0.9
        ax.plot(xs, ys, "o-", linewidth=lw, markersize=5, alpha=alpha, label=LABEL_MAP.get(agg, agg))

    if y_max is not None and np.isfinite(y_max) and y_max > 0:
        ax.set_ylim(0.0, float(y_max))

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Network MAE (absolute, amt units)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build two MAE plots: with and without FedAvg")
    parser.add_argument("--input-json", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    in_path = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    series = _extract_series(payload)
    if not series:
        raise RuntimeError("No series found in input JSON")

    scenario = payload.get("scenario") or {}
    model = str(payload.get("best_model", "model"))
    subtitle = (
        f"{model} | partition={scenario.get('column_partition')} | "
        f"mal={scenario.get('malicious_frac')} | attack={scenario.get('attack_strategy')}"
    )

    with_fedavg_path = out_dir / "fig3_decentralized_methods_network_mae_with_fedavg.png"
    without_fedavg_path = out_dir / "fig3_decentralized_methods_network_mae_without_fedavg.png"
    series_wo_fedavg = {k: v for k, v in series.items() if k != "decentralized_fedavg"}

    _plot(series, f"Network MAE vs round — with FedAvg ({subtitle})", with_fedavg_path, y_max=None)
    _plot(series_wo_fedavg, f"Network MAE vs round — without FedAvg ({subtitle})", without_fedavg_path, y_max=None)

    print(f"Saved: {with_fedavg_path}")
    print(f"Saved: {without_fedavg_path}")


if __name__ == "__main__":
    main()