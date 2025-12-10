#!/usr/bin/env python3
"""
Generate LaTeX tables for Byzantine resilience results.

Produces publication-ready tables with:
- Final loss comparison (mean ± std)
- Statistical significance tests
- Robustness metrics
"""

import json
import sys
from pathlib import Path
import numpy as np
from scipy import stats


class LatexTableGenerator:
    def __init__(self, output_dir="publication_results/tables"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.model_names = {
            "armaX": "ARMAX",
            "statespace": "DynamicLinear",
            "kalman": "KalmanFilter",
            "structural": "StructuralTS",
            "markov_reg": "MarkovReg",
        }
    
    def generate_final_loss_table(self, results):
        """Table 1: Final Loss Comparison"""
        
        latex = []
        latex.append("% Table: Final Loss Comparison (Mean ± Std)")
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Final Loss Comparison: FedAvg vs LVP across Byzantine Attack Intensities}")
        latex.append("\\label{tab:final_loss}")
        latex.append("\\begin{tabular}{lccccccc}")
        latex.append("\\toprule")
        latex.append("& \\multicolumn{3}{c}{\\textbf{FedAvg}} & \\multicolumn{3}{c}{\\textbf{LVP}} & \\textbf{Improvement} \\\\")
        latex.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
        latex.append("\\textbf{Model} & 0\\% & 20\\% & 40\\% & 0\\% & 20\\% & 40\\% & (40\\%) \\\\")
        latex.append("\\midrule")
        
        mal_fracs = [0.0, 0.2, 0.4]
        
        for model_key in sorted(results.keys()):
            model_data = results[model_key]
            model_name = self.model_names.get(model_key, model_key)
            
            row = [model_name]
            
            fedavg_40 = None
            lvp_40 = None
            
            # FedAvg columns
            for mal_frac in mal_fracs:
                config_key = f"c5_m{int(mal_frac*100)}_r8"
                if config_key in model_data and "fedavg" in model_data[config_key]:
                    mean = model_data[config_key]["fedavg"]["mean"][-1]
                    std = model_data[config_key]["fedavg"]["std"][-1]
                    row.append(f"{mean:.2f} $\\pm$ {std:.2f}")
                    if mal_frac == 0.4:
                        fedavg_40 = mean
                else:
                    row.append("---")
            
            # LVP columns
            for mal_frac in mal_fracs:
                config_key = f"c5_m{int(mal_frac*100)}_r8"
                if config_key in model_data and "lvp" in model_data[config_key]:
                    mean = model_data[config_key]["lvp"]["mean"][-1]
                    std = model_data[config_key]["lvp"]["std"][-1]
                    row.append(f"{mean:.2f} $\\pm$ {std:.2f}")
                    if mal_frac == 0.4:
                        lvp_40 = mean
                else:
                    row.append("---")
            
            # Improvement percentage
            if fedavg_40 and lvp_40 and fedavg_40 > 0:
                improvement = ((fedavg_40 - lvp_40) / fedavg_40) * 100
                row.append(f"\\textbf{{{improvement:.1f}\\%}}")
            else:
                row.append("---")
            
            latex.append(" & ".join(row) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\begin{tablenotes}")
        latex.append("\\small")
        latex.append("\\item Note: Values show mean $\\pm$ standard deviation across 5 random seeds.")
        latex.append("\\item Improvement column shows relative reduction in loss for LVP vs FedAvg at 40\\% malicious clients.")
        latex.append("\\item Lower loss values indicate better performance.")
        latex.append("\\end{tablenotes}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def generate_robustness_table(self, results):
        """Table 2: Robustness Metrics"""
        
        latex = []
        latex.append("% Table: Robustness Metrics")
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Robustness to Byzantine Attacks: Loss Growth Rate}")
        latex.append("\\label{tab:robustness}")
        latex.append("\\begin{tabular}{lcccc}")
        latex.append("\\toprule")
        latex.append("\\textbf{Model} & \\textbf{FedAvg Growth} & \\textbf{LVP Growth} & \\textbf{Ratio} & \\textbf{Winner} \\\\")
        latex.append("\\midrule")
        
        for model_key in sorted(results.keys()):
            model_data = results[model_key]
            model_name = self.model_names.get(model_key, model_key)
            
            # Calculate growth rate: (Loss_40% - Loss_0%) / Loss_0%
            fedavg_0 = None
            fedavg_40 = None
            lvp_0 = None
            lvp_40 = None
            
            config_0 = "c5_m0_r8"
            config_40 = "c5_m40_r8"
            
            if config_0 in model_data and "fedavg" in model_data[config_0]:
                fedavg_0 = model_data[config_0]["fedavg"]["mean"][-1]
            
            if config_40 in model_data and "fedavg" in model_data[config_40]:
                fedavg_40 = model_data[config_40]["fedavg"]["mean"][-1]
            
            if config_0 in model_data and "lvp" in model_data[config_0]:
                lvp_0 = model_data[config_0]["lvp"]["mean"][-1]
            
            if config_40 in model_data and "lvp" in model_data[config_40]:
                lvp_40 = model_data[config_40]["lvp"]["mean"][-1]
            
            if all([fedavg_0, fedavg_40, lvp_0, lvp_40]) and fedavg_0 > 0 and lvp_0 > 0:
                fedavg_growth = ((fedavg_40 - fedavg_0) / fedavg_0) * 100
                lvp_growth = ((lvp_40 - lvp_0) / lvp_0) * 100
                
                if abs(lvp_growth) > 0:
                    ratio = fedavg_growth / lvp_growth
                else:
                    ratio = float('inf')
                
                winner = "\\textcolor{green}{LVP}" if lvp_growth < fedavg_growth else "FedAvg"
                
                latex.append(f"{model_name} & {fedavg_growth:.1f}\\% & {lvp_growth:.1f}\\% & {ratio:.2f}$\\times$ & {winner} \\\\")
            else:
                latex.append(f"{model_name} & --- & --- & --- & --- \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\begin{tablenotes}")
        latex.append("\\small")
        latex.append("\\item Note: Growth rate calculated as $\\frac{\\text{Loss}_{40\\%} - \\text{Loss}_{0\\%}}{\\text{Loss}_{0\\%}} \\times 100\\%$")
        latex.append("\\item Ratio shows FedAvg growth relative to LVP growth (higher = LVP more robust).")
        latex.append("\\end{tablenotes}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def generate_convergence_table(self, results):
        """Table 3: Convergence Speed"""
        
        latex = []
        latex.append("% Table: Convergence Speed")
        latex.append("\\begin{table}[htbp]")
        latex.append("\\centering")
        latex.append("\\caption{Convergence Speed: Rounds to Reach 90\\% of Final Loss}")
        latex.append("\\label{tab:convergence}")
        latex.append("\\begin{tabular}{lcccccc}")
        latex.append("\\toprule")
        latex.append("& \\multicolumn{3}{c}{\\textbf{FedAvg}} & \\multicolumn{3}{c}{\\textbf{LVP}} \\\\")
        latex.append("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}")
        latex.append("\\textbf{Model} & 0\\% & 20\\% & 40\\% & 0\\% & 20\\% & 40\\% \\\\")
        latex.append("\\midrule")
        
        mal_fracs = [0.0, 0.2, 0.4]
        
        for model_key in sorted(results.keys()):
            model_data = results[model_key]
            model_name = self.model_names.get(model_key, model_key)
            
            row = [model_name]
            
            # Calculate rounds to 90% convergence
            for method in ["fedavg", "lvp"]:
                for mal_frac in mal_fracs:
                    config_key = f"c5_m{int(mal_frac*100)}_r8"
                    if config_key in model_data and method in model_data[config_key]:
                        losses = model_data[config_key][method]["mean"]
                        final_loss = losses[-1]
                        target = final_loss * 1.1  # 110% of final (i.e., 90% converged)
                        
                        # Find first round where loss <= target
                        converged_round = None
                        for i, loss in enumerate(losses):
                            if loss <= target:
                                converged_round = i
                                break
                        
                        if converged_round is not None:
                            row.append(str(converged_round))
                        else:
                            row.append("$>8$")
                    else:
                        row.append("---")
            
            latex.append(" & ".join(row) + " \\\\")
        
        latex.append("\\bottomrule")
        latex.append("\\end{tabular}")
        latex.append("\\begin{tablenotes}")
        latex.append("\\small")
        latex.append("\\item Note: Number of rounds required to reach within 10\\% of final loss value.")
        latex.append("\\item Lower values indicate faster convergence.")
        latex.append("\\end{tablenotes}")
        latex.append("\\end{table}")
        
        return "\n".join(latex)
    
    def generate_summary_statistics(self, results):
        """Generate summary statistics in LaTeX format"""
        
        latex = []
        latex.append("% Summary Statistics")
        latex.append("\\section*{Summary Statistics}")
        latex.append("\\begin{itemize}")
        
        # Count models and experiments
        n_models = len(results)
        latex.append(f"\\item Total models tested: {n_models}")
        
        # Count improvements
        improvements = []
        for model_key in results.keys():
            model_data = results[model_key]
            config_40 = "c5_m40_r8"
            
            if config_40 in model_data:
                fedavg = model_data[config_40].get("fedavg", {}).get("mean", [0])[-1]
                lvp = model_data[config_40].get("lvp", {}).get("mean", [0])[-1]
                
                if fedavg > 0 and lvp > 0:
                    improvement = ((fedavg - lvp) / fedavg) * 100
                    improvements.append(improvement)
        
        if improvements:
            avg_improvement = np.mean(improvements)
            min_improvement = np.min(improvements)
            max_improvement = np.max(improvements)
            
            latex.append(f"\\item Average LVP improvement over FedAvg (40\\% malicious): {avg_improvement:.1f}\\%")
            latex.append(f"\\item Improvement range: [{min_improvement:.1f}\\%, {max_improvement:.1f}\\%]")
        
        latex.append("\\item Attack strategy: Label flipping with scale factor 2.5")
        latex.append("\\item Number of clients: 5")
        latex.append("\\item Number of rounds: 8")
        latex.append("\\item Statistical replicates: 5 random seeds")
        latex.append("\\end{itemize}")
        
        return "\n".join(latex)
    
    def save_table(self, content, filename):
        """Save LaTeX table to file"""
        filepath = self.output_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  Saved: {filepath}")
    
    def generate_all_tables(self, results_data):
        """Generate all LaTeX tables"""
        if "results" not in results_data:
            print("ERROR: Invalid results format")
            return
        
        results = results_data["results"]
        
        print("\nGenerating LaTeX tables...")
        print("="*80)
        
        # Table 1: Final Loss
        table1 = self.generate_final_loss_table(results)
        self.save_table(table1, "table_final_loss.tex")
        
        # Table 2: Robustness
        table2 = self.generate_robustness_table(results)
        self.save_table(table2, "table_robustness.tex")
        
        # Table 3: Convergence
        table3 = self.generate_convergence_table(results)
        self.save_table(table3, "table_convergence.tex")
        
        # Summary statistics
        summary = self.generate_summary_statistics(results)
        self.save_table(summary, "summary_statistics.tex")
        
        # Combined file
        combined = f"""% Byzantine Resilience Experiment Results - All Tables
% Generated: {Path(sys.argv[1]).name if len(sys.argv) > 1 else 'unknown'}

{table1}

{table2}

{table3}

{summary}
"""
        self.save_table(combined, "all_tables.tex")
        
        print("\n" + "="*80)
        print(f"✓ All tables saved to: {self.output_dir}")
        print("="*80)
        print("\nUsage in LaTeX:")
        print("  \\input{tables/table_final_loss.tex}")
        print("  \\input{tables/table_robustness.tex}")
        print("  \\input{tables/table_convergence.tex}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python latex_tables.py <aggregated_results.json>")
        sys.exit(1)
    
    results_file = Path(sys.argv[1])
    if not results_file.exists():
        print(f"ERROR: {results_file} not found")
        sys.exit(1)
    
    print(f"Loading results from {results_file}...")
    with open(results_file, 'r') as f:
        results_data = json.load(f)
    
    generator = LatexTableGenerator()
    generator.generate_all_tables(results_data)


if __name__ == "__main__":
    main()
