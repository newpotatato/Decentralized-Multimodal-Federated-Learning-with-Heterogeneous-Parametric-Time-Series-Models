#!/usr/bin/env python3
"""Plot learning curves for all models run in current scenario."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _extract_model_data(results: List[Dict[str, Any]], model_name: str) -> tuple[List[int], List[float]]:
    """Extract rounds and network MAE for a specific model."""
    for rec in results:
        if rec.get("model") == model_name:
            hist = rec.get("history") or []
            rounds = [int(h.get("round", 0)) for h in hist]
            mae = [float(h.get("network_mae", 0.0)) for h in hist]
            return rounds, mae
    return [], []


def main() -> None:
    pkg = Path("federated_learning/artifacts/article_package_current_run_20260410")
    json_path = pkg / "raw" / "all_models_lvp_seed42.json"
    
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")
    
    data = json.loads(json_path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    
    # Get unique models
    models_in_data = set(r.get("model") for r in results if r.get("model"))
    print(f"Found models: {sorted(models_in_data)}")
    
    # Extract data for each model
    model_data: Dict[str, tuple[List[int], List[float]]] = {}
    for model in sorted(models_in_data):
        rounds, mae = _extract_model_data(results, model)
        if rounds and mae:
            model_data[model] = (rounds, mae)
    
    if not model_data:
        raise RuntimeError("No model data found")
    
    # Plot
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    
    label_map = {
        "DynamicLinearModel": "Dynamic Linear",
        "ARMAXModel": "ARMAX",
        "KalmanFilterModel": "Kalman Filter",
        "StructuralTimeSeriesModel": "Structural TS",
        "MarkovSwitchingRegressionModel": "Markov Switching",
    }
    
    for idx, model in enumerate(sorted(model_data.keys())):
        rounds, mae = model_data[model]
        x = rounds
        y = mae
        color = colors[idx % len(colors)] if colors else None
        lw = 2.5 if model == "DynamicLinearModel" else 1.7
        
        ax.plot(x, y, "o-", linewidth=lw, markersize=5, color=color, label=label_map.get(model, model))
    
    ax.set_xlabel("Communication round", fontsize=11)
    ax.set_ylabel("Network MAE", fontsize=11)
    ax.set_title("Model comparison: Learning curves (LVP aggregator, current scenario)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    
    out_dir = pkg / "plots" / "dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = out_dir / "all_models_current_scenario_learning_curves.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    
    # Report
    report_path = out_dir / "all_models_current_scenario_report.md"
    report_lines = [
        "# All Models: Learning Curves (LVP aggregator, current scenario)",
        "",
        f"Models tested: {len(models_in_data)}",
        "",
        "## Final round MAE",
    ]
    for model in sorted(model_data.keys()):
        _, mae = model_data[model]
        final_mae = float(mae[-1])
        report_lines.append(f"- {label_map.get(model, model)}: {final_mae:.2f}")
    
    report_lines.extend([
        "",
        f"Data source: {json_path}",
        f"Figure: {fig_path}",
    ])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    
    print(f"\nSaved: {fig_path}")
    print(f"Saved: {report_path}")
    print("\nFinal MAE by model:")
    for model in sorted(model_data.keys()):
        _, mae = model_data[model]
        print(f"  {label_map.get(model, model)}: {float(mae[-1]):.2f}")


if __name__ == "__main__":
    main()
