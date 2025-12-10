#!/usr/bin/env python3
"""
Real-data federated tests for time-series models with exogenous factors.
Covers ARMAXModel, DynamicLinearModel, KalmanFilterModel, StructuralTimeSeriesModel,
MarkovSwitchingRegressionModel with FedAvg and LVP aggregation.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_utils import (
    build_clients_from_mcc,
    load_mcc_series,
    load_news_exogenous,
    load_moex_series,
    build_clients_from_moex,
    train_test_split_series,
)
from aggregators import (
    corrupt_params,
    fedavg_aggregate,
    lvp_aggregate,
    sanitize_params,
)

# Import model classes from local v2 directories
import sys
v2_path = Path(__file__).parent
sys.path.insert(0, str(v2_path / "models"))
sys.path.insert(0, str(v2_path / "data_loaders"))
from arma_models import ARMAXModel
from state_space_models import DynamicLinearModel, KalmanFilterModel, StructuralTimeSeriesModel
from markov_switching_models import MarkovSwitchingRegressionModel
from reuters_loader import build_reuters_daily

MODEL_REGISTRY = {
    "ARMAXModel": ARMAXModel,
    "DynamicLinearModel": DynamicLinearModel,
    "KalmanFilterModel": KalmanFilterModel,
    "StructuralTimeSeriesModel": StructuralTimeSeriesModel,
    "MarkovSwitchingRegressionModel": MarkovSwitchingRegressionModel,
}

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def evaluate_model(model, test_df: pd.DataFrame) -> float:
    if test_df.empty:
        return float("inf")
    steps = min(len(test_df), 10)
    if steps < 1:
        return float("inf")

    exog_future = None
    exog_cols = [c for c in test_df.columns if c.startswith("exog")]
    if exog_cols:
        exog_future = np.asarray(test_df[exog_cols].values[:steps], dtype=float)
    try:
        forecast = model.predict(steps=steps, exog_future=exog_future, use_transform=False)
    except TypeError:
        try:
            forecast = model.predict(steps=steps, use_transform=False)
        except Exception:
            return float("inf")
    except Exception:
        return float("inf")

    forecast = np.asarray(forecast, dtype=float).flatten()
    target = np.asarray(test_df["amt"].values[: len(forecast)], dtype=float)
    if len(target) == 0:
        return float("inf")
    mse = float(np.mean((forecast - target) ** 2))
    return mse


def train_local_model(
    ModelClass,
    df: pd.DataFrame,
    local_epochs: int,
    global_params: Dict[str, np.ndarray],
) -> Tuple[Dict[str, np.ndarray], float, int]:
    train_df, test_df = train_test_split_series(df, test_ratio=0.2)
    model = ModelClass()
    if global_params:
        model.set_initial_params(global_params)

    fit_kwargs = {"target_col": "amt", "use_transform": False}
    exog_cols = [c for c in train_df.columns if c.startswith("exog")]
    
    # Only ARMAXModel supports exog_cols parameter
    model_name = ModelClass.__name__
    if exog_cols and model_name == "ARMAXModel":
        fit_kwargs["exog_cols"] = exog_cols

    for _ in range(max(1, local_epochs)):
        model.fit(train_df, **fit_kwargs)

    mse = evaluate_model(model, test_df)
    params = model.get_params()
    return params, mse, len(test_df)


def run_one_model(
    model_name: str,
    ModelClass,
    clients: List[pd.DataFrame],
    aggregator: str,
    rounds: int,
    local_epochs: int,
    malicious_frac: float,
    seed: int,
    attack_strategy: str = "label_flip",
    attack_scale: float = 2.5,
) -> Dict:
    rng = random.Random(seed)
    n_clients = len(clients)
    n_mal = int(round(n_clients * malicious_frac))
    malicious_ids = set(rng.sample(range(n_clients), n_mal)) if n_mal > 0 else set()

    global_params: Dict[str, np.ndarray] = {}
    history: List[Dict] = []

    for r in range(rounds):
        client_params: Dict[str, Dict[str, np.ndarray]] = {}
        client_metrics: Dict[str, float] = {}
        client_sizes: Dict[str, int] = {}
        quality: Dict[str, float] = {}

        for idx, df in enumerate(clients):
            cid = f"client_{idx}"
            params, mse, test_len = train_local_model(ModelClass, df, local_epochs, global_params)
            if idx in malicious_ids:
                params = corrupt_params(params, scale=attack_scale, strategy=attack_strategy)
                mse = mse * rng.uniform(2.5, 5.0)
            client_params[cid] = params
            client_metrics[cid] = mse
            client_sizes[cid] = max(test_len, 1)
            # Softer discount so median can still leverage weights; avoids collapsing malicious weights to ~0
            denom = 1.0 + (mse if np.isfinite(mse) else 1e6)
            quality[cid] = 1.0 / np.sqrt(denom)

        filtered, used_keys = sanitize_params(client_params)
        aggregated: Dict[str, np.ndarray] = {}
        if filtered:
            if aggregator == "fedavg":
                aggregated = fedavg_aggregate(filtered, client_sizes)
            else:
                aggregated = lvp_aggregate(filtered, quality)
            global_params = aggregated

        # Evaluate global model on ALL clients (for convergence curve)
        global_mses = []
        for idx, df in enumerate(clients):
            try:
                global_model = ModelClass()
                if global_params:
                    global_model.set_initial_params(global_params)
                train_df, test_df = train_test_split_series(df, test_ratio=0.2)
                
                fit_kwargs = {"target_col": "amt", "use_transform": False}
                exog_cols = [c for c in train_df.columns if c.startswith("exog")]
                if exog_cols and ModelClass.__name__ == "ARMAXModel":
                    fit_kwargs["exog_cols"] = exog_cols
                
                global_model.fit(train_df, **fit_kwargs)
                global_mse = evaluate_model(global_model, test_df)
                global_mses.append(global_mse if np.isfinite(global_mse) else 1e6)
            except Exception:
                global_mses.append(1e6)
        
        server_mse = float(np.mean(global_mses)) if global_mses else 1e6

        history.append(
            {
                "round": r + 1,
                "client_mse": client_metrics,
                "server_mse": server_mse,
                "used_param_keys": used_keys,
                "aggregated": {k: v.tolist() for k, v in aggregated.items()},
                "malicious_ids": sorted(list(malicious_ids)),
            }
        )

    return {
        "model": model_name,
        "aggregator": aggregator,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "malicious_frac": malicious_frac,
        "history": history,
    }


def _build_exogenous(base_path: Path, mcc_df: pd.DataFrame, use_reuters: bool) -> Optional[pd.DataFrame]:
    """Combine Fontanka sentiment/counts with Reuters sentiment if requested."""
    exog_cols = {}
    try:
        news = load_news_exogenous(base_path)
        if news is not None:
            exog_cols["exog_news"] = news.values
    except Exception:
        news = None

    if use_reuters:
        try:
            real_data_path = Path(__file__).parents[1] / "real_data_integration"
            reuters_df = build_reuters_daily(real_data_path, mcc_df["date"])
            exog_cols["exog_reuters"] = reuters_df["reuters_sentiment"].values
        except Exception:
            # fall back silently if Reuters not available
            pass

    if not exog_cols:
        return None
    exog_frame = pd.DataFrame()
    target_len = len(mcc_df)

    def _pad(values):
        arr = np.asarray(values, dtype=float)
        if len(arr) >= target_len:
            return arr[:target_len]
        out = np.zeros(target_len, dtype=float)
        out[: len(arr)] = arr
        if len(arr) > 0:
            out[len(arr) :] = arr[-1]
        return out

    for key, vals in exog_cols.items():
        exog_frame[key] = _pad(vals)

    return exog_frame


def run_grid(
    base_path: Path,
    model_names: List[str],
    n_clients_list: List[int],
    malicious_fracs: List[float],
    aggregators: List[str],
    rounds_list: List[int],
    local_epochs_list: List[int],
    seed: int,
    max_combos: int = None,
    use_reuters: bool = True,
    attack_strategy: str = "label_flip",
    attack_scale: float = 2.5,
    output_path: Path = None,
    data_source: str = "mcc",  # "mcc", "moex", or "news"
) -> Dict:
    """
    Run federated learning experiments.
    
    Args:
        data_source: "mcc" (transactions), "moex" (stocks), or "news" (news frequency)
    """
    
    # Load data based on source
    if data_source == "moex":
        print("📊 Loading MOEX stock data...")
        try:
            moex_df = load_moex_series(base_path)
            print(f"✓ MOEX data loaded: {moex_df.shape}")
        except Exception as e:
            print(f"❌ Failed to load MOEX: {e}")
            raise
        exog = None  # MOEX doesn't need exogenous factors
    elif data_source == "mcc":
        print("💳 Loading MCC transaction data...")
        mcc_df = load_mcc_series(base_path)
        exog = _build_exogenous(base_path, mcc_df, use_reuters)
    elif data_source == "news":
        print("📰 Loading news data...")
        try:
            news_df = load_news_exogenous(base_path)
            # Convert to DataFrame format
            mcc_df = pd.DataFrame({
                "date": news_df.index,
                "news_count": news_df.values
            })
            exog = None
        except Exception as e:
            print(f"❌ Failed to load news: {e}")
            raise
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    results: List[Dict] = []
    combo_counter = 0
    
    # Load existing results if output_path exists (for resuming)
    if output_path and output_path.exists():
        try:
            existing = json.loads(output_path.read_text())
            results = existing.get("results", [])
            print(f"Resuming from {len(results)} existing results")
        except Exception as e:
            print(f"Could not load existing results: {e}")
    
    # Pre-calculate total combinations for progress bar
    total_combos = len(model_names) * len(n_clients_list) * len(malicious_fracs) * len(aggregators) * len(rounds_list) * len(local_epochs_list)
    if max_combos:
        total_combos = min(total_combos, max_combos)
    
    pbar = tqdm(total=total_combos, desc="Overall Progress", unit="exp")

    for n_clients in n_clients_list:
        # Build clients based on data source
        if data_source == "moex":
            clients = build_clients_from_moex(moex_df, n_clients=n_clients)
        else:
            clients = build_clients_from_mcc(mcc_df, exog, n_clients=n_clients)
        
        if len(clients) < n_clients:
            clients = clients[: max(1, len(clients))]
        
        for model_name in model_names:
            ModelClass = MODEL_REGISTRY[model_name]
            print(f"\n{'='*60}\nTesting model: {model_name} (n_clients={len(clients)})\n{'='*60}")
            
            model_results: List[Dict] = []
            model_start = time.time()
            
            try:
                for mal_frac in malicious_fracs:
                    for agg in aggregators:
                        for rounds in rounds_list:
                            for local_epochs in local_epochs_list:
                                combo_counter += 1
                                if max_combos and combo_counter > max_combos:
                                    pbar.close()
                                    payload = {"results": results + model_results, "truncated": True, "combos": combo_counter - 1}
                                    if output_path:
                                        output_path.parent.mkdir(parents=True, exist_ok=True)
                                        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                                    return payload
                                
                                exp = run_one_model(
                                    model_name,
                                    ModelClass,
                                    clients,
                                    aggregator=agg,
                                    rounds=rounds,
                                    local_epochs=local_epochs,
                                    malicious_frac=mal_frac,
                                    seed=seed + combo_counter,
                                    attack_strategy=attack_strategy,
                                    attack_scale=attack_scale,
                                )
                                exp["n_clients"] = len(clients)
                                model_results.append(exp)
                                
                                # Print progress for this experiment
                                final_mse = np.mean(list(exp["history"][-1]["client_mse"].values()))
                                pbar.update(1)
                                pbar.set_postfix({
                                    "model": model_name,
                                    "mal": mal_frac,
                                    "agg": agg,
                                    "mse": f"{final_mse:.2e}"
                                })
            
            except Exception as e:
                print(f"  ✗ ERROR in {model_name}: {e}")
                import traceback
                traceback.print_exc()
                # Save what we have so far and continue to next model
                results.extend(model_results)
                if output_path:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    payload = {"results": results, "truncated": False, "combos": combo_counter}
                    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    print(f"  Saved partial results ({len(results)} total) to {output_path}")
                continue
            
            # Save results after each model completes successfully
            model_time = time.time() - model_start
            results.extend(model_results)
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {"results": results, "truncated": False, "combos": combo_counter}
                output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                print(f"  ✓ {model_name} done in {model_time:.1f}s. Checkpoint: {len(results)} total to {output_path}")
    
    pbar.close()
    return {"results": results, "truncated": False, "combos": combo_counter}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Federated real-data tests with exogenous factors")
    parser.add_argument("--base-path", type=str, default=str(Path(__file__).parents[1]), help="Path to repo root (expects 01_data_transactions and 02_data_fontanka)")
    parser.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()), help="Subset of models to run")
    parser.add_argument("--n-clients", nargs="+", type=int, default=[3, 5, 10], help="Client counts")
    parser.add_argument("--malicious-fracs", nargs="+", type=float, default=[0.0, 0.2, 0.4], help="Malicious client fractions")
    parser.add_argument("--aggregators", nargs="+", default=["fedavg", "lvp"], help="Aggregation methods")
    parser.add_argument("--rounds", nargs="+", type=int, default=[5, 20], help="Rounds per experiment")
    parser.add_argument("--local-epochs", nargs="+", type=int, default=[1, 3, 5], help="Local epochs")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--max-combos", type=int, default=None, help="Limit number of combinations for quick smoke runs")
    parser.add_argument("--output", type=str, default=str(ARTIFACTS / "real_federated_results.json"), help="Output JSON path")
    parser.add_argument("--no-reuters", action="store_true", help="Disable Reuters exogenous features")
    parser.add_argument("--attack-strategy", type=str, default="label_flip", choices=["label_flip", "noise", "random"], help="Byzantine attack strategy")
    parser.add_argument("--attack-scale", type=float, default=2.5, help="Attack intensity (noise scale)")
    parser.add_argument("--data-source", type=str, default="mcc", choices=["mcc", "moex", "news"], help="Data source: MCC transactions, MOEX stocks, or news frequency")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_path).resolve()
    out_path = Path(args.output)

    payload = run_grid(
        base_path=base_path,
        model_names=[m for m in args.models if m in MODEL_REGISTRY],
        n_clients_list=args.n_clients,
        malicious_fracs=args.malicious_fracs,
        aggregators=args.aggregators,
        rounds_list=args.rounds,
        local_epochs_list=args.local_epochs,
        seed=args.seed,
        max_combos=args.max_combos,
        use_reuters=not args.no_reuters,
        attack_strategy=args.attack_strategy,
        attack_scale=args.attack_scale,
        output_path=out_path,
        data_source=args.data_source,
    )

    # Final save (redundant but confirms completion)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n✓ All done. Final results: {out_path} ({len(payload['results'])} experiments)")
    if payload.get("truncated"):
        print("  WARNING: combinations truncated by max-combos")


if __name__ == "__main__":
    main()
