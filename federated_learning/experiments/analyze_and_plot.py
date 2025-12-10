#!/usr/bin/env python3
"""
Analyze federated learning results and generate comparison plots.

Metrics:
1. Final MSE (last round avg across clients) - lower is better
2. Convergence speed (MSE improvement over rounds)
3. Robustness to malicious clients (MSE growth with increasing malicious_frac)
4. Heterogeneity (variance across clients)
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(json_path: Path) -> Dict:
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(results: List[Dict]) -> pd.DataFrame:
    """Extract key metrics from each experiment."""
    rows = []
    for exp in results:
        model = exp["model"]
        agg = exp["aggregator"]
        mal_frac = exp["malicious_frac"]
        n_rounds = exp["rounds"]
        
        history = exp["history"]
        if not history:
            continue
        
        # Final round metrics
        final_round = history[-1]
        client_mses = list(final_round["client_mse"].values())
        final_avg_mse = np.mean(client_mses)
        final_std_mse = np.std(client_mses)
        
        # Convergence: MSE improvement from first to last round
        first_round = history[0]
        first_avg_mse = np.mean(list(first_round["client_mse"].values()))
        improvement = (first_avg_mse - final_avg_mse) / max(first_avg_mse, 1e-9)
        
        rows.append({
            "model": model,
            "aggregator": agg,
            "malicious_frac": mal_frac,
            "rounds": n_rounds,
            "final_mse": final_avg_mse,
            "mse_std": final_std_mse,
            "improvement": improvement,
            "first_mse": first_avg_mse,
        })
    
    return pd.DataFrame(rows)


def plot_final_mse_comparison(df: pd.DataFrame, output_dir: Path):
    """Compare final MSE by aggregator and malicious fraction."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by malicious_frac
    for mal_frac in sorted(df["malicious_frac"].unique()):
        subset = df[df["malicious_frac"] == mal_frac]
        
        ax = axes[0]
        for agg in ["fedavg", "lvp"]:
            agg_data = subset[subset["aggregator"] == agg]
            if not agg_data.empty:
                x = agg_data["model"].values
                y = agg_data["final_mse"].values
                label = f"{agg} (mal={mal_frac:.1f})"
                ax.plot(x, y, marker='o', label=label, alpha=0.7)
        
        ax.set_xlabel("Model")
        ax.set_ylabel("Final MSE")
        ax.set_title("Final MSE by Model & Aggregator")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=15)
    
    # Robustness: MSE growth with malicious fraction
    ax = axes[1]
    for agg in ["fedavg", "lvp"]:
        agg_data = df[df["aggregator"] == agg].groupby("malicious_frac")["final_mse"].mean().reset_index()
        ax.plot(agg_data["malicious_frac"], agg_data["final_mse"], marker='s', label=agg, linewidth=2)
    
    ax.set_xlabel("Malicious Fraction")
    ax.set_ylabel("Avg Final MSE")
    ax.set_title("Robustness: MSE vs Malicious Fraction")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    out_path = output_dir / "final_mse_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_convergence(results: List[Dict], output_dir: Path):
    """Plot MSE convergence for benign vs worst-case malicious runs."""
    if not results:
        return

    mal_values = sorted({exp["malicious_frac"] for exp in results})
    benign_mal = mal_values[0]
    worst_mal = mal_values[-1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    def _plot_for(mal_target: float, ax, title: str):
        subset = [exp for exp in results if exp["malicious_frac"] == mal_target]
        for exp in subset:
            agg = exp["aggregator"]
            rounds = [r_data["round"] for r_data in exp["history"]]
            avg_mses = [np.mean(list(r_data["client_mse"].values())) for r_data in exp["history"]]
            ax.plot(rounds, avg_mses, marker='o', label=agg, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylabel("Avg MSE")
        ax.legend()
        ax.grid(True, alpha=0.3)

    _plot_for(benign_mal, axes[0], f"Convergence (mal={benign_mal:.2f})")
    _plot_for(worst_mal, axes[1], f"Convergence (mal={worst_mal:.2f})")

    plt.tight_layout()
    out_path = output_dir / "convergence_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def plot_heterogeneity(results: List[Dict], output_dir: Path):
    """Plot MSE variance across clients (heterogeneity measure)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_by_mal = {}
    for exp in results:
        agg = exp["aggregator"]
        mal_frac = exp["malicious_frac"]
        final_round = exp["history"][-1]
        client_mses = list(final_round["client_mse"].values())
        std_mse = np.std(client_mses)
        
        key = (mal_frac, agg)
        if key not in data_by_mal:
            data_by_mal[key] = []
        data_by_mal[key].append(std_mse)
    
    # Compute mean for each (malicious_frac, aggregator) pair
    mal_fracs = sorted(set(k[0] for k in data_by_mal.keys()))
    x_labels = [f"mal={m:.1f}" for m in mal_fracs]
    x = np.arange(len(mal_fracs))
    width = 0.35
    
    fedavg_stds = []
    lvp_stds = []
    
    for mal_frac in mal_fracs:
        fedavg_vals = data_by_mal.get((mal_frac, "fedavg"), [0])
        lvp_vals = data_by_mal.get((mal_frac, "lvp"), [0])
        fedavg_stds.append(np.mean(fedavg_vals))
        lvp_stds.append(np.mean(lvp_vals))
    
    ax.bar(x - width/2, fedavg_stds, width, label='FedAvg', alpha=0.8)
    ax.bar(x + width/2, lvp_stds, width, label='LVP', alpha=0.8)
    
    ax.set_xlabel("Malicious Fraction")
    ax.set_ylabel("Avg MSE Std Dev (heterogeneity)")
    ax.set_title("Client Heterogeneity: MSE Variance Across Clients")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    out_path = output_dir / "heterogeneity.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()


def generate_summary_table(df: pd.DataFrame, output_dir: Path):
    """Save summary table as CSV."""
    summary = df.groupby(["aggregator", "malicious_frac"]).agg({
        "final_mse": ["mean", "std"],
        "improvement": "mean",
        "mse_std": "mean",
    }).reset_index()
    
    out_path = output_dir / "summary_metrics.csv"
    summary.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    
    return summary


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_and_plot.py <results.json> [output_dir]")
        sys.exit(1)
    
    results_path = Path(sys.argv[1])
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else results_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading results from {results_path}...")
    data = load_results(results_path)
    results = data.get("results", [])
    
    if not results:
        print("No results found in JSON.")
        sys.exit(1)
    
    print(f"Found {len(results)} experiments.")
    
    # Compute metrics
    df = compute_metrics(results)
    print("\nMetrics DataFrame:")
    print(df.to_string())
    
    # Generate plots
    print("\nGenerating plots...")
    plot_final_mse_comparison(df, output_dir)
    plot_convergence(results, output_dir)
    plot_heterogeneity(results, output_dir)
    
    # Summary table
    print("\nGenerating summary table...")
    summary = generate_summary_table(df, output_dir)
    print(summary.to_string())
    
    print(f"\n✅ All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
