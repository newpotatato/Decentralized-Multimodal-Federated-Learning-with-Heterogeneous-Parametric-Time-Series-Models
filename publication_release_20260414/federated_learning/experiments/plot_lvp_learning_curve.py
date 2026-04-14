#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _extract_histories(payload: Dict[str, Any]) -> Dict[str, tuple[List[int], List[float]]]:
    """Extract rounds and network MAE for all aggregators from a single seed."""
    out: Dict[str, tuple[List[int], List[float]]] = {}
    for rec in payload.get("results", []):
        agg = str(rec.get("aggregator", "")).lower()
        hist = rec.get("history") or []
        rounds = [int(h.get("round", 0)) for h in hist]
        mae = [float(h.get("network_mae", 0.0)) for h in hist]
        if rounds and mae:
            out[agg] = (rounds, mae)
    return out


def main() -> None:
    pkg = Path("federated_learning/artifacts/article_package_current_run_20260410")
    seeds = [42, 52, 62]
    
    # Order and labels for aggregators
    agg_order = ["lvp", "decentralized_fedavg", "defta", "balance", "push_sum"]
    agg_labels = {
        "lvp": "LVP",
        "decentralized_fedavg": "Decentralized FedAvg",
        "defta": "DeFTA",
        "balance": "BALANCE",
        "push_sum": "Push-Sum",
    }
    
    # Read all seed data
    seed_data: Dict[int, Dict[str, tuple[List[int], List[float]]]] = {}
    for seed in seeds:
        json_path = pkg / "raw" / "seeds" / f"seed{seed}" / "fig3_decentralized_methods.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Missing {json_path}")
        
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        histories = _extract_histories(payload)
        seed_data[seed] = histories
    
    if not seed_data:
        raise RuntimeError("No seed data found")
    
    # Aggregate across seeds for each aggregator
    agg_aggregated: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
    ref_rounds = None
    
    for agg in agg_order:
        mae_curves: List[np.ndarray] = []
        for seed in seeds:
            if agg in seed_data[seed]:
                _, mae = seed_data[seed][agg]
                mae_curves.append(np.asarray(mae, dtype=float))
        
        if not mae_curves:
            continue
        
        # Use first seed's rounds as reference
        if ref_rounds is None:
            _, ref_rounds_list = seed_data[seeds[0]][agg]
            ref_rounds = ref_rounds_list
        
        n_rounds = len(ref_rounds)
        mae_arr = np.asarray([m[:n_rounds] for m in mae_curves], dtype=float)
        mae_mean = np.mean(mae_arr, axis=0)
        mae_std = np.std(mae_arr, axis=0, ddof=0)
        agg_aggregated[agg] = (mae_mean, mae_std)
    
    if not agg_aggregated:
        raise RuntimeError("No aggregator data found")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    
    x = ref_rounds
    for idx, agg in enumerate(agg_order):
        if agg not in agg_aggregated:
            continue
        
        mae_mean, mae_std = agg_aggregated[agg]
        color = colors[idx % len(colors)] if colors else None
        lw = 2.5 if agg == "lvp" else 1.7
        
        ax.plot(x, mae_mean, "o-", linewidth=lw, markersize=5, color=color, label=agg_labels.get(agg, agg))
        ax.fill_between(x, mae_mean - mae_std, mae_mean + mae_std, alpha=0.15, color=color)
    
    ax.set_xlabel("Communication round", fontsize=11)
    ax.set_ylabel("Network MAE", fontsize=11)
    ax.set_title("Aggregator comparison: Learning curves (mean±std across seeds 42, 52, 62)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    
    out_dir = pkg / "plots" / "dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = out_dir / "all_aggregators_learning_curve.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    
    # Report
    report_path = out_dir / "all_aggregators_learning_curve_report.md"
    report_lines = [
        "# Aggregator Comparison: Learning Curves",
        "",
        f"Seeds: {', '.join(str(s) for s in seeds)}",
        f"Rounds: {len(ref_rounds)}",
        "",
        "## Final round MAE (round {})".format(ref_rounds[-1]),
    ]
    for agg in agg_order:
        if agg in agg_aggregated:
            mae_mean, mae_std = agg_aggregated[agg]
            final_mae = float(mae_mean[-1])
            final_std = float(mae_std[-1])
            report_lines.append(f"- {agg_labels.get(agg, agg)}: {final_mae:.2f} ± {final_std:.2f}")
    
    report_lines.extend([
        "",
        f"Figure: {fig_path}",
    ])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    
    print(f"Saved: {fig_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
