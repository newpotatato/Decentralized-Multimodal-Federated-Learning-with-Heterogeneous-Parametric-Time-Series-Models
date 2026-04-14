#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _extract_model_history(payload: List[Dict[str, Any]], model_name: str, agg_name: str = "lvp") -> tuple[List[int], List[float]]:
    """Extract rounds and network MAE for specific model and aggregator."""
    for rec in payload:
        if rec.get("model") == model_name and rec.get("aggregator") == agg_name:
            hist = rec.get("history") or []
            rounds = [int(h.get("round", 0)) for h in hist]
            mae = [float(h.get("network_mae", 0.0)) for h in hist]
            return rounds, mae
    return [], []


def main() -> None:
    json_path = Path("federated_learning/artifacts/full_scale_5models.json")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")
    
    data = json.loads(json_path.read_text(encoding="utf-8"))
    results = data.get("results", [])
    
    # Get unique models
    models = sorted(set(r.get("model") for r in results if r.get("model")))
    print(f"Found models: {models}")
    
    # Extract data for each model with LVP aggregator
    model_data: Dict[str, tuple[List[int], List[float]]] = {}
    for model in models:
        rounds, mae = _extract_model_history(results, model, "lvp")
        if rounds and mae:
            model_data[model] = (rounds, mae)
    
    if not model_data:
        print("No LVP data found for models. Trying fedavg...")
        for model in models:
            rounds, mae = _extract_model_history(results, model, "fedavg")
            if rounds and mae:
                model_data[model] = (rounds, mae)
    
    if not model_data:
        raise RuntimeError("No model data found")
    
    # Plot all models with LVP
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
    ax.set_title("Model comparison: Learning curves (LVP aggregator)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    
    out_dir = Path("federated_learning/artifacts/article_package_current_run_20260410/plots/dynamics")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig_path = out_dir / "model_comparison_learning_curves.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    
    # Report
    report_path = out_dir / "model_comparison_learning_curves_report.md"
    report_lines = [
        "# Model Comparison: Learning Curves (LVP aggregator)",
        "",
        f"Models: {', '.join(sorted(model_data.keys()))}",
        "",
        "## Final round MAE",
    ]
    for model in sorted(model_data.keys()):
        _, mae = model_data[model]
        final_mae = float(mae[-1])
        report_lines.append(f"- {label_map.get(model, model)}: {final_mae:.2f}")
    
    report_lines.extend([
        "",
        f"Source: {json_path}",
        f"Figure: {fig_path}",
    ])
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    
    print(f"Saved: {fig_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
