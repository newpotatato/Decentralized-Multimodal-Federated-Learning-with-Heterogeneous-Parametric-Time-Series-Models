#!/usr/bin/env python3
"""
Master script for reproducing all Byzantine resilience experiments
for scientific publication.

Usage:
    python reproduce_experiments.py --mode quick    # Fast smoke test
    python reproduce_experiments.py --mode full     # Full experiments (5 seeds)
    python reproduce_experiments.py --mode paper    # Exact paper figures
"""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np


class ExperimentRunner:
    def __init__(self, output_dir="publication_results"):
        self.root = Path(__file__).parent
        self.output_dir = self.root / output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Publication parameters
        self.paper_config = {
            "n_clients": 5,
            "rounds": 8,
            "malicious_fracs": [0.0, 0.2, 0.4],
            "models": ["armaX", "statespace", "kalman", "structural", "markov_reg"],
            "seeds": [42, 123, 456, 789, 2024],  # 5 seeds for statistical significance
        }
        
        self.quick_config = {
            "n_clients": 5,
            "rounds": 5,
            "malicious_fracs": [0.0, 0.2],
            "models": ["statespace", "markov_reg"],  # Fast models only
            "seeds": [42],
        }
    
    def log(self, msg):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] {msg}")
    
    def run_single_experiment(self, config, seed, output_file):
        """Run one experimental configuration"""
        self.log(f"Running experiment with seed={seed}")
        
        script = self.root / "08_federated_learning" / "test_5models.py"
        if not script.exists():
            self.log(f"ERROR: {script} not found")
            return False
        
        # Modify script to use specific seed
        cmd = [
            sys.executable,
            str(script),
            "--seed", str(seed),
            "--output", str(output_file)
        ]
        
        try:
            # Run the experiment
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=str(script.parent),
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                self.log(f"✓ Completed seed={seed}")
                return True
            else:
                self.log(f"✗ Failed seed={seed}: {result.stderr[:200]}")
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"✗ Timeout seed={seed}")
            return False
        except Exception as e:
            self.log(f"✗ Error seed={seed}: {e}")
            return False
    
    def aggregate_results(self, result_files):
        """Aggregate results from multiple seeds"""
        self.log("Aggregating results from multiple runs...")
        
        all_data = []
        for f in result_files:
            if f.exists():
                try:
                    with open(f, 'r') as fp:
                        data = json.load(fp)
                        all_data.append(data)
                except Exception as e:
                    self.log(f"Warning: Could not load {f}: {e}")
        
        if not all_data:
            self.log("ERROR: No valid result files found")
            return None
        
        # Compute mean and std across seeds
        aggregated = {
            "metadata": {
                "n_seeds": len(all_data),
                "aggregation_date": datetime.now().isoformat(),
            },
            "results": {}
        }
        
        # Extract all model keys
        models = set()
        for data in all_data:
            models.update(data.keys())
        
        for model in models:
            aggregated["results"][model] = {}
            
            # Get all configurations
            configs = set()
            for data in all_data:
                if model in data:
                    configs.update(data[model].keys())
            
            for config in configs:
                aggregated["results"][model][config] = {}
                
                # Aggregate fedavg
                fedavg_losses = []
                lvp_losses = []
                
                for data in all_data:
                    if model in data and config in data[model]:
                        if "fedavg" in data[model][config]:
                            fedavg_losses.append(data[model][config]["fedavg"])
                        if "lvp" in data[model][config]:
                            lvp_losses.append(data[model][config]["lvp"])
                
                if fedavg_losses:
                    # Convert to numpy array (each row is a seed, columns are rounds)
                    fedavg_arr = np.array(fedavg_losses)
                    aggregated["results"][model][config]["fedavg"] = {
                        "mean": fedavg_arr.mean(axis=0).tolist(),
                        "std": fedavg_arr.std(axis=0).tolist(),
                        "min": fedavg_arr.min(axis=0).tolist(),
                        "max": fedavg_arr.max(axis=0).tolist(),
                    }
                
                if lvp_losses:
                    lvp_arr = np.array(lvp_losses)
                    aggregated["results"][model][config]["lvp"] = {
                        "mean": lvp_arr.mean(axis=0).tolist(),
                        "std": lvp_arr.std(axis=0).tolist(),
                        "min": lvp_arr.min(axis=0).tolist(),
                        "max": lvp_arr.max(axis=0).tolist(),
                    }
        
        return aggregated
    
    def run_quick_test(self):
        """Run quick smoke test"""
        self.log("="*80)
        self.log("QUICK TEST MODE - Fast verification")
        self.log("="*80)
        
        output_file = self.output_dir / "quick_test_results.json"
        success = self.run_single_experiment(
            self.quick_config,
            seed=42,
            output_file=output_file
        )
        
        if success:
            self.log("✓ Quick test completed successfully")
            return output_file
        else:
            self.log("✗ Quick test failed")
            return None
    
    def run_full_experiments(self):
        """Run full experiments with multiple seeds"""
        self.log("="*80)
        self.log("FULL EXPERIMENT MODE - Multiple seeds for statistics")
        self.log("="*80)
        
        config = self.paper_config
        result_files = []
        
        for seed_idx, seed in enumerate(config["seeds"], 1):
            self.log(f"\n[Seed {seed_idx}/{len(config['seeds'])}] Running with seed={seed}")
            
            output_file = self.output_dir / f"results_seed_{seed}.json"
            success = self.run_single_experiment(config, seed, output_file)
            
            if success:
                result_files.append(output_file)
        
        if not result_files:
            self.log("\n✗ All experiments failed")
            return None
        
        self.log(f"\n✓ Completed {len(result_files)}/{len(config['seeds'])} experiments")
        
        # Aggregate results
        aggregated = self.aggregate_results(result_files)
        if aggregated:
            agg_file = self.output_dir / "aggregated_results.json"
            with open(agg_file, 'w') as f:
                json.dump(aggregated, f, indent=2)
            self.log(f"✓ Saved aggregated results to {agg_file}")
            return agg_file
        
        return None
    
    def run_paper_figures(self):
        """Generate exact figures for paper"""
        self.log("="*80)
        self.log("PAPER FIGURES MODE - Reproduce exact publication figures")
        self.log("="*80)
        
        # Check if aggregated results exist
        agg_file = self.output_dir / "aggregated_results.json"
        if not agg_file.exists():
            self.log("No aggregated results found. Running full experiments first...")
            agg_file = self.run_full_experiments()
            if not agg_file:
                return False
        
        # Generate publication-quality plots
        self.log("\nGenerating publication figures...")
        plot_script = self.root / "publication_plots.py"
        
        if not plot_script.exists():
            self.log(f"ERROR: {plot_script} not found")
            return False
        
        try:
            result = subprocess.run(
                [sys.executable, str(plot_script), str(agg_file)],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                self.log("✓ Publication figures generated")
                return True
            else:
                self.log(f"✗ Figure generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"✗ Error generating figures: {e}")
            return False
    
    def generate_latex_tables(self):
        """Generate LaTeX tables for paper"""
        self.log("\nGenerating LaTeX tables...")
        
        agg_file = self.output_dir / "aggregated_results.json"
        if not agg_file.exists():
            self.log("ERROR: No aggregated results found")
            return False
        
        table_script = self.root / "latex_tables.py"
        if not table_script.exists():
            self.log(f"ERROR: {table_script} not found")
            return False
        
        try:
            result = subprocess.run(
                [sys.executable, str(table_script), str(agg_file)],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                self.log("✓ LaTeX tables generated")
                return True
            else:
                self.log(f"✗ Table generation failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.log(f"✗ Error generating tables: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce Byzantine resilience experiments for publication"
    )
    parser.add_argument(
        "--mode",
        choices=["quick", "full", "paper", "all"],
        default="quick",
        help="Experiment mode: quick (smoke test), full (5 seeds), paper (figures only), all (full + figures)"
    )
    parser.add_argument(
        "--output-dir",
        default="publication_results",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    runner = ExperimentRunner(output_dir=args.output_dir)
    
    start_time = datetime.now()
    runner.log(f"Starting experiment reproduction - Mode: {args.mode}")
    
    success = False
    
    if args.mode == "quick":
        result = runner.run_quick_test()
        success = result is not None
    
    elif args.mode == "full":
        result = runner.run_full_experiments()
        success = result is not None
    
    elif args.mode == "paper":
        success = runner.run_paper_figures()
        if success:
            runner.generate_latex_tables()
    
    elif args.mode == "all":
        result = runner.run_full_experiments()
        if result:
            success = runner.run_paper_figures()
            if success:
                runner.generate_latex_tables()
    
    elapsed = datetime.now() - start_time
    runner.log("="*80)
    if success:
        runner.log(f"✓ EXPERIMENT COMPLETED in {elapsed}")
        runner.log(f"Results saved to: {runner.output_dir}")
    else:
        runner.log(f"✗ EXPERIMENT FAILED after {elapsed}")
    runner.log("="*80)


if __name__ == "__main__":
    main()
