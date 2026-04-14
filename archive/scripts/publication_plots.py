#!/usr/bin/env python3
"""
Generate publication-quality figures with confidence intervals
for Byzantine resilience experiments.

Creates LaTeX-ready plots with:
- Error bars (standard deviation across seeds)
- High-resolution vector graphics (PDF + PNG)
- Proper font sizes for papers
- Color schemes optimized for printing
"""

import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import rc

# LaTeX-style fonts
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 11})
rc('text', usetex=False)  # Set to True if LaTeX is installed


class PublicationPlotter:
    def __init__(self, output_dir="publication_results/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Publication color scheme (colorblind-friendly)
        self.colors = {
            "mal_0": "#0173B2",   # Blue
            "mal_20": "#DE8F05",  # Orange
            "mal_40": "#CC78BC",  # Purple
            "fedavg": "#CA3542",  # Red
            "lvp": "#029E73",     # Green
        }
        
        self.model_names = {
            "armaX": "ARMAX",
            "statespace": "DynamicLinear",
            "kalman": "KalmanFilter",
            "structural": "StructuralTS",
            "markov_reg": "MarkovReg",
        }
    
    def plot_learning_curves(self, model_data, model_key, mal_fracs, rounds=8):
        """Plot learning curves with confidence intervals"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        for mal_frac in mal_fracs:
            config_key = f"c5_m{int(mal_frac*100)}_r{rounds}"
            if config_key not in model_data:
                continue
            
            result = model_data[config_key]
            color_key = f"mal_{int(mal_frac*100)}"
            color = self.colors.get(color_key, "#000000")
            
            # Plot FedAvg (dashed)
            if "fedavg" in result:
                mean = np.array(result["fedavg"]["mean"])
                std = np.array(result["fedavg"]["std"])
                x = np.arange(len(mean))
                
                line, = ax.plot(x, mean, 
                              linestyle='--', linewidth=2.5, 
                              marker='o', markersize=6,
                              color=color, alpha=0.8,
                              label=f'FedAvg {mal_frac:.0%}')
                
                # Confidence interval (±1 std)
                ax.fill_between(x, mean - std, mean + std, 
                               color=color, alpha=0.15)
            
            # Plot LVP (solid)
            if "lvp" in result:
                mean = np.array(result["lvp"]["mean"])
                std = np.array(result["lvp"]["std"])
                x = np.arange(len(mean))
                
                line, = ax.plot(x, mean, 
                              linestyle='-', linewidth=3, 
                              marker='s', markersize=6,
                              color=color, alpha=1.0,
                              label=f'LVP {mal_frac:.0%}')
                
                ax.fill_between(x, mean - std, mean + std, 
                               color=color, alpha=0.2)
        
        ax.set_xlabel('Training Round', fontsize=13, fontweight='bold')
        ax.set_ylabel('Loss (log scale)', fontsize=13, fontweight='bold')
        ax.set_title(f'{self.model_names.get(model_key, model_key)}: Learning Curves', 
                    fontsize=14, fontweight='bold')
        
        ax.legend(fontsize=10, loc='best', framealpha=0.95)
        ax.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        ax.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def plot_final_loss_comparison(self, model_data, model_key, mal_fracs, rounds=8):
        """Bar chart with error bars for final loss"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        mal_frac_vals = []
        fedavg_means = []
        fedavg_stds = []
        lvp_means = []
        lvp_stds = []
        
        for mal_frac in mal_fracs:
            config_key = f"c5_m{int(mal_frac*100)}_r{rounds}"
            if config_key not in model_data:
                continue
            
            result = model_data[config_key]
            mal_frac_vals.append(int(mal_frac * 100))
            
            # FedAvg final loss
            if "fedavg" in result:
                mean = result["fedavg"]["mean"][-1]
                std = result["fedavg"]["std"][-1]
                fedavg_means.append(mean)
                fedavg_stds.append(std)
            else:
                fedavg_means.append(0)
                fedavg_stds.append(0)
            
            # LVP final loss
            if "lvp" in result:
                mean = result["lvp"]["mean"][-1]
                std = result["lvp"]["std"][-1]
                lvp_means.append(mean)
                lvp_stds.append(std)
            else:
                lvp_means.append(0)
                lvp_stds.append(0)
        
        if not mal_frac_vals:
            return fig
        
        x = np.arange(len(mal_frac_vals))
        width = 0.35
        
        # Bar chart with error bars
        bars1 = ax.bar(x - width/2, fedavg_means, width, 
                      yerr=fedavg_stds, capsize=5,
                      label='FedAvg', color=self.colors["fedavg"], 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        bars2 = ax.bar(x + width/2, lvp_means, width, 
                      yerr=lvp_stds, capsize=5,
                      label='LVP', color=self.colors["lvp"], 
                      alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.1f}',
                          ha='center', va='bottom', fontsize=10, 
                          fontweight='bold')
        
        ax.set_xlabel('Malicious Clients (%)', fontsize=13, fontweight='bold')
        ax.set_ylabel('Final Loss', fontsize=13, fontweight='bold')
        ax.set_title(f'{self.model_names.get(model_key, model_key)}: Final Loss Comparison', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{m}%' for m in mal_frac_vals], fontsize=11)
        ax.legend(fontsize=11, loc='upper left', framealpha=0.95)
        ax.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=1)
        
        plt.tight_layout()
        return fig
    
    def plot_combined_figure(self, model_data, model_key, mal_fracs, rounds=8):
        """Combined two-panel figure (like in the attachments)"""
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 6))
        
        # LEFT: Learning Curves
        for mal_frac in mal_fracs:
            config_key = f"c5_m{int(mal_frac*100)}_r{rounds}"
            if config_key not in model_data:
                continue
            
            result = model_data[config_key]
            color_key = f"mal_{int(mal_frac*100)}"
            color = self.colors.get(color_key, "#000000")
            
            # FedAvg (dashed)
            if "fedavg" in result:
                mean = np.array(result["fedavg"]["mean"])
                std = np.array(result["fedavg"]["std"])
                x = np.arange(len(mean))
                
                ax_left.plot(x, mean, linestyle='--', linewidth=2.5, 
                           marker='o', markersize=7, color=color, alpha=0.75,
                           label=f'FedAvg {mal_frac:.0%}')
                ax_left.fill_between(x, mean - std, mean + std, 
                                    color=color, alpha=0.15)
            
            # LVP (solid)
            if "lvp" in result:
                mean = np.array(result["lvp"]["mean"])
                std = np.array(result["lvp"]["std"])
                x = np.arange(len(mean))
                
                ax_left.plot(x, mean, linestyle='-', linewidth=3, 
                           marker='s', markersize=7, color=color, alpha=1.0,
                           label=f'LVP {mal_frac:.0%}')
                ax_left.fill_between(x, mean - std, mean + std, 
                                    color=color, alpha=0.2)
        
        ax_left.set_xlabel('Training Round', fontsize=13, fontweight='bold')
        ax_left.set_ylabel('Loss (log scale)', fontsize=13, fontweight='bold')
        ax_left.set_title('Learning Curves: FedAvg (dashed) vs LVP (solid)', 
                         fontsize=13, fontweight='bold')
        ax_left.legend(fontsize=9, loc='best', framealpha=0.95, ncol=2)
        ax_left.grid(True, alpha=0.3, linestyle=':', linewidth=1)
        ax_left.set_yscale('log')
        
        # RIGHT: Final Loss Comparison
        mal_frac_vals = []
        fedavg_means = []
        fedavg_stds = []
        lvp_means = []
        lvp_stds = []
        
        for mal_frac in mal_fracs:
            config_key = f"c5_m{int(mal_frac*100)}_r{rounds}"
            if config_key not in model_data:
                continue
            
            result = model_data[config_key]
            mal_frac_vals.append(int(mal_frac * 100))
            
            if "fedavg" in result:
                fedavg_means.append(result["fedavg"]["mean"][-1])
                fedavg_stds.append(result["fedavg"]["std"][-1])
            else:
                fedavg_means.append(0)
                fedavg_stds.append(0)
            
            if "lvp" in result:
                lvp_means.append(result["lvp"]["mean"][-1])
                lvp_stds.append(result["lvp"]["std"][-1])
            else:
                lvp_means.append(0)
                lvp_stds.append(0)
        
        if mal_frac_vals:
            x = np.arange(len(mal_frac_vals))
            width = 0.35
            
            bars1 = ax_right.bar(x - width/2, fedavg_means, width, 
                               yerr=fedavg_stds, capsize=5,
                               label='FedAvg', color=self.colors["fedavg"], 
                               alpha=0.8, edgecolor='black', linewidth=1.5)
            
            bars2 = ax_right.bar(x + width/2, lvp_means, width, 
                               yerr=lvp_stds, capsize=5,
                               label='LVP', color=self.colors["lvp"], 
                               alpha=0.8, edgecolor='black', linewidth=1.5)
            
            # Value labels
            for bars, means in [(bars1, fedavg_means), (bars2, lvp_means)]:
                for bar, mean_val in zip(bars, means):
                    if mean_val > 0:
                        ax_right.text(bar.get_x() + bar.get_width()/2., 
                                     bar.get_height(),
                                     f'{mean_val:.1f}',
                                     ha='center', va='bottom', fontsize=10, 
                                     fontweight='bold',
                                     bbox=dict(boxstyle='round,pad=0.3', 
                                             facecolor='yellow', alpha=0.7))
            
            ax_right.set_xlabel('Malicious Clients %', fontsize=13, fontweight='bold')
            ax_right.set_ylabel('Final Loss', fontsize=13, fontweight='bold')
            ax_right.set_title('Final Loss Comparison: LVP Lower = Better', 
                             fontsize=13, fontweight='bold')
            ax_right.set_xticks(x)
            ax_right.set_xticklabels([f'{m}%' for m in mal_frac_vals], fontsize=11)
            ax_right.legend(fontsize=11, loc='upper left', framealpha=0.95)
            ax_right.grid(True, alpha=0.3, axis='y', linestyle=':', linewidth=1)
        
        fig.suptitle(f'Byzantine Resilience: {self.model_names.get(model_key, model_key)} (5 Clients, 8 Rounds)\n' +
                    'FedAvg VULNERABLE to Poisoned Metrics | LVP ROBUST',
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        return fig
    
    def save_figure(self, fig, filename):
        """Save figure in multiple formats"""
        base_path = self.output_dir / filename
        
        # PNG for quick viewing
        fig.savefig(f"{base_path}.png", dpi=300, bbox_inches='tight')
        
        # PDF for LaTeX (vector graphics)
        fig.savefig(f"{base_path}.pdf", bbox_inches='tight')
        
        print(f"  Saved: {base_path}.png and {base_path}.pdf")
        plt.close(fig)
    
    def generate_all_figures(self, results_data):
        """Generate all publication figures"""
        if "results" not in results_data:
            print("ERROR: Invalid results format")
            return
        
        results = results_data["results"]
        mal_fracs = [0.0, 0.2, 0.4]
        
        print("\nGenerating publication figures...")
        print("="*80)
        
        for model_key in results.keys():
            print(f"\nProcessing {self.model_names.get(model_key, model_key)}...")
            model_data = results[model_key]
            
            # Combined figure (main paper figure)
            fig = self.plot_combined_figure(model_data, model_key, mal_fracs)
            self.save_figure(fig, f"fig_byzantine_{model_key}_combined")
            
            # Individual figures for supplementary material
            fig = self.plot_learning_curves(model_data, model_key, mal_fracs)
            self.save_figure(fig, f"fig_byzantine_{model_key}_curves")
            
            fig = self.plot_final_loss_comparison(model_data, model_key, mal_fracs)
            self.save_figure(fig, f"fig_byzantine_{model_key}_bars")
        
        print("\n" + "="*80)
        print(f"✓ All figures saved to: {self.output_dir}")
        print("="*80)


def main():
    if len(sys.argv) < 2:
        print("Usage: python publication_plots.py <aggregated_results.json>")
        sys.exit(1)
    
    results_file = Path(sys.argv[1])
    if not results_file.exists():
        print(f"ERROR: {results_file} not found")
        sys.exit(1)
    
    print(f"Loading results from {results_file}...")
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    plotter = PublicationPlotter()
    plotter.generate_all_figures(results_data)


if __name__ == "__main__":
    main()
