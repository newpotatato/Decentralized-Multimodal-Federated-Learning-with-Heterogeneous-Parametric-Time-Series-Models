#!/usr/bin/env python3
"""
Real-data federated forecasting aligned with the manuscript:
- Centralized FedAvg baseline.
- Decentralized Local Voting Protocol (LVP) sync on a Jaccard-based graph.
- Additional baselines: Krum, weighted median after L2 norm (server), SCAFFOLD-style
  control variates on parameter deltas, robust_median (alias of weighted median).

Primary metric: mean absolute error (MAE) on a fixed forecast horizon K (default 10).
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

# federated_learning/{core,data_loaders}
_FL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_FL_ROOT / "core"))
sys.path.insert(0, str(_FL_ROOT / "data_loaders"))

from data_utils import (
    build_client_information_profiles,
    build_clients_from_mcc,
    build_clients_from_moex,
    load_mcc_series,
    load_moex_series,
    load_news_exogenous,
    train_test_split_series,
)
from aggregators import (
    corrupt_params,
    fedavg_aggregate,
    krum_aggregate,
    sanitize_params,
    weighted_median_norm_aggregate,
)
from param_dict_ops import param_dict_add, param_dict_mean, param_dict_sub
from decentralized_consensus import (
    decentralized_balance_synchronize,
    balance_push_sum_synchronize,
    decentralized_fedavg_synchronize,
    defta_synchronize,
    push_sum_synchronize,
)
from decentralized_lvp import (
    build_neighbor_graph,
    decentralized_lvp_synchronize,
    default_alpha_from_graph,
)
from arma_models import ARMAXModel
from markov_switching_models import MarkovSwitchingRegressionModel
from state_space_models import (
    DynamicLinearModel,
    KalmanFilterModel,
    StructuralTimeSeriesModel,
)
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

# Repo root (parent of federated_learning/)
REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[2]


def _krum_f_effective(n_clients: int, malicious_frac: float, krum_f: int) -> int:
    """Byzantine upper bound f for Krum; if krum_f<0, use round(malicious_frac * n)."""
    cap = max(0, (n_clients - 3) // 2)
    if krum_f >= 0:
        return min(int(krum_f), cap)
    f_est = int(round(malicious_frac * n_clients))
    return min(max(f_est, 0), cap)


def _reuters_integration_dirs(repo_root: Path) -> List[Path]:
    return [
        _FL_ROOT / "real_data_integration",
        repo_root / "real_data_integration",
        repo_root / "federated_learning" / "real_data_integration",
        repo_root / "data_LVP" / "archive" / "08_federated_learning_old" / "real_data_integration",
        repo_root / "archive" / "08_federated_learning_old" / "real_data_integration",
    ]


def _attach_fit_budget(
    ModelClass: type,
    fit_kwargs: Dict,
    max_iterations: Optional[int],
) -> Dict:
    """Pass max_iterations into model.fit for statsmodels-backed models (FL variant 1)."""
    if max_iterations is None:
        return fit_kwargs
    name = ModelClass.__name__
    if name not in (
        "DynamicLinearModel",
        "KalmanFilterModel",
        "StructuralTimeSeriesModel",
        "ARMAXModel",
    ):
        return fit_kwargs
    return {**fit_kwargs, "max_iterations": int(max_iterations)}


def _model_uses_transform(ModelClass: type) -> bool:
    """Use constrained parameterization for statsmodels-backed models by default."""
    return ModelClass.__name__ in (
        "DynamicLinearModel",
        "KalmanFilterModel",
        "StructuralTimeSeriesModel",
        "ARMAXModel",
    )


def stabilize_numeric_param_dict(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Replace NaN/inf in numeric parameter tensors and clip variance-like keys
    so federated averaging does not explode sigma2 and downstream fits stay finite.
    Non-numeric dict entries are skipped (unchanged models may omit them).
    """
    if not params:
        return {}
    out: Dict[str, Any] = {}
    for k, v in params.items():
        try:
            arr = np.atleast_1d(np.asarray(v, dtype=float))
        except (TypeError, ValueError):
            continue
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e10, neginf=-1e10)
        lk = str(k).lower()
        if "sigma" in lk:
            arr = np.clip(arr, 1e-8, 1e10)
        out[k] = arr
    return out


def _stabilize_armax_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Apply tighter bounds for ARMAX parameters to prevent federated drift."""
    if not params:
        return {}
    out: Dict[str, Any] = {}
    for k, v in params.items():
        try:
            arr = np.atleast_1d(np.asarray(v, dtype=float))
        except (TypeError, ValueError):
            continue
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
        lk = str(k).lower()
        if lk.startswith("ar.") or lk.startswith("ma."):
            # Keep AR/MA coefficients away from unstable regions after averaging.
            arr = np.clip(arr, -0.98, 0.98)
        elif lk == "const":
            # Large intercept magnitudes were the main ARMAX failure mode.
            arr = np.clip(arr, -1e6, 1e6)
        elif "sigma" in lk:
            arr = np.clip(arr, 1e-8, 1e8)
        out[k] = arr
    return out


def _param_dict_l2_norm(params: Optional[Dict[str, Any]]) -> float:
    if not params:
        return 0.0
    sq_sum = 0.0
    for v in params.values():
        try:
            arr = np.asarray(v, dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        if arr.size == 0:
            continue
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        sq_sum += float(np.dot(arr, arr))
    return float(np.sqrt(max(sq_sum, 0.0)))


def _clip_param_dict_l2(params: Dict[str, Any], max_norm: Optional[float]) -> Dict[str, Any]:
    if not params or max_norm is None:
        return params
    limit = float(max_norm)
    if limit <= 0.0:
        return params
    norm = _param_dict_l2_norm(params)
    if not np.isfinite(norm) or norm <= limit or norm == 0.0:
        return params
    scale = limit / norm
    clipped: Dict[str, Any] = {}
    for k, v in params.items():
        try:
            arr = np.asarray(v, dtype=float)
        except (TypeError, ValueError):
            continue
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        clipped[k] = arr * scale
    return clipped


def _components_undirected(neighbors: List[List[int]]) -> List[List[int]]:
    n = len(neighbors)
    if n == 0:
        return []
    undirected = [set() for _ in range(n)]
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if 0 <= j < n and j != i:
                undirected[i].add(int(j))
                undirected[int(j)].add(i)
    seen = [False] * n
    components: List[List[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: List[int] = []
        while stack:
            node = stack.pop()
            comp.append(node)
            for nxt in undirected[node]:
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(nxt)
        components.append(comp)
    return components


def _flatten_numeric_param_dict(params: Dict[str, Any]) -> np.ndarray:
    if not params:
        return np.asarray([], dtype=float)
    parts: List[np.ndarray] = []
    for key in sorted(params.keys()):
        try:
            arr = np.asarray(params[key], dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        if arr.size:
            parts.append(arr)
    if not parts:
        return np.asarray([], dtype=float)
    return np.concatenate(parts)


def _param_l2_distance(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    if not a or not b:
        return 0.0
    keys = sorted(set(a.keys()) & set(b.keys()))
    if not keys:
        return 0.0
    diffs: List[np.ndarray] = []
    for key in keys:
        try:
            left = np.asarray(a[key], dtype=float).ravel()
            right = np.asarray(b[key], dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        if left.shape != right.shape:
            continue
        if left.size:
            diffs.append(left - right)
    if not diffs:
        return 0.0
    vec = np.concatenate(diffs)
    return float(np.linalg.norm(vec, ord=2))


def _component_sync_delta(
    current_states: List[Dict[str, np.ndarray]],
    previous_states: List[Dict[str, np.ndarray]],
    components: List[List[int]],
) -> float:
    if not current_states or not previous_states or not components:
        return 0.0
    comp_scores: List[float] = []
    for comp in components:
        deltas: List[float] = []
        for idx in comp:
            if idx < 0 or idx >= len(current_states) or idx >= len(previous_states):
                continue
            deltas.append(_param_l2_distance(current_states[idx], previous_states[idx]))
        if deltas:
            comp_scores.append(float(np.mean(deltas)))
    if not comp_scores:
        return 0.0
    return float(np.mean(comp_scores))


def _component_consensus_metrics(
    local_states: List[Dict[str, np.ndarray]],
    components: List[List[int]],
) -> Tuple[float, float]:
    """
    Return two consensus-style metrics averaged over connected components:
    - centroid_l2_mean: mean L2 distance from client states to component centroid.
    - pairwise_l2_mean: mean pairwise L2 distance inside each component.
    """
    if not local_states or not components:
        return 0.0, 0.0

    centroid_scores: List[float] = []
    pairwise_scores: List[float] = []

    for comp in components:
        valid_ids = [
            idx
            for idx in comp
            if 0 <= idx < len(local_states) and isinstance(local_states[idx], dict)
        ]
        if not valid_ids:
            continue

        comp_states = [local_states[idx] for idx in valid_ids]
        centroid = param_dict_mean(comp_states)

        d_centroid = [_param_l2_distance(local_states[idx], centroid) for idx in valid_ids]
        centroid_scores.append(float(np.mean(d_centroid)) if d_centroid else 0.0)

        if len(valid_ids) <= 1:
            pairwise_scores.append(0.0)
            continue
        pairwise: List[float] = []
        for i in range(len(valid_ids)):
            for j in range(i + 1, len(valid_ids)):
                pairwise.append(
                    _param_l2_distance(
                        local_states[valid_ids[i]],
                        local_states[valid_ids[j]],
                    )
                )
        pairwise_scores.append(float(np.mean(pairwise)) if pairwise else 0.0)

    centroid_mean = float(np.mean(centroid_scores)) if centroid_scores else 0.0
    pairwise_mean = float(np.mean(pairwise_scores)) if pairwise_scores else 0.0
    return centroid_mean, pairwise_mean


def _select_malicious_ids(
    n_clients: int,
    n_mal: int,
    selection: str,
    neighbors: Optional[List[List[int]]],
    rng: random.Random,
) -> set:
    if n_clients <= 0 or n_mal <= 0:
        return set()
    n_pick = min(n_mal, n_clients)
    mode = (selection or "random").strip().lower()
    if mode == "hub_targeted" and neighbors is not None:
        degree_order = sorted(
            range(n_clients),
            key=lambda i: len(neighbors[i]) if i < len(neighbors) else 0,
            reverse=True,
        )
        return set(degree_order[:n_pick])
    return set(rng.sample(range(n_clients), n_pick))


def _structural_neighbor_graph(n_clients: int, topology_mode: str) -> Tuple[List[List[int]], np.ndarray]:
    mode = (topology_mode or "").strip().lower()
    if n_clients <= 0:
        return [], np.zeros((0, 0), dtype=float)

    neighbors: List[List[int]] = []
    if mode == "complete":
        neighbors = [[j for j in range(n_clients) if j != i] for i in range(n_clients)]
    elif mode == "ring":
        if n_clients == 1:
            neighbors = [[]]
        else:
            neighbors = [[(i - 1) % n_clients, (i + 1) % n_clients] for i in range(n_clients)]
    elif mode == "line":
        for i in range(n_clients):
            row: List[int] = []
            if i - 1 >= 0:
                row.append(i - 1)
            if i + 1 < n_clients:
                row.append(i + 1)
            neighbors.append(row)
    elif mode == "star":
        if n_clients == 1:
            neighbors = [[]]
        else:
            neighbors = []
            for i in range(n_clients):
                if i == 0:
                    neighbors.append([j for j in range(1, n_clients)])
                else:
                    neighbors.append([0])
    else:
        raise ValueError(f"Unknown structural topology mode: {topology_mode!r}")

    kappa = np.ones((n_clients, n_clients), dtype=float)
    np.fill_diagonal(kappa, 0.0)
    return neighbors, kappa


def _apply_block_missing(df: pd.DataFrame, frac: float, seed: int, client_idx: int) -> pd.DataFrame:
    if frac <= 0.0 or df.empty:
        return df
    n = len(df)
    block = int(round(n * float(np.clip(frac, 0.0, 0.9))))
    if block < 2:
        return df
    local_rng = np.random.default_rng(int(seed + 97 * (client_idx + 1)))
    start_max = max(n - block, 0)
    start = int(local_rng.integers(0, start_max + 1)) if start_max > 0 else 0
    end = min(start + block, n)
    out = df.copy(deep=True)
    exog_cols = [c for c in out.columns if c.startswith("exog")]
    if exog_cols:
        out.loc[out.index[start:end], exog_cols] = 0.0
        out["exog_missing_block"] = 0.0
        out.loc[out.index[start:end], "exog_missing_block"] = 1.0
    if "amt" in out.columns and n >= 3:
        amt = out["amt"].to_numpy(dtype=float, copy=True)
        fill_val = amt[start - 1] if start > 0 else amt[start]
        amt[start:end] = fill_val
        out["amt"] = amt
    return out


def _apply_round_regime_cascade(
    df: pd.DataFrame,
    client_idx: int,
    round_idx: int,
    rounds: int,
    drift_scale: float,
) -> pd.DataFrame:
    if drift_scale <= 0.0 or df.empty or "amt" not in df.columns:
        return df
    group = int(client_idx % 3)
    if group == 0:
        onset = max(1, rounds // 4)
    elif group == 1:
        onset = max(1, rounds // 2)
    else:
        onset = max(1, (3 * rounds) // 4)
    if round_idx < onset:
        return df
    progress = float(round_idx - onset + 1) / float(max(1, rounds - onset + 1))
    amp = float(np.clip(drift_scale, 0.0, 2.0)) * progress
    out = df.copy(deep=True)
    y = out["amt"].to_numpy(dtype=float, copy=True)
    std = float(np.std(y))
    shift = amp * (std if std > 1e-12 else max(1e-6, float(np.mean(np.abs(y)) + 1e-6)))
    regime_sign = 1.0 if (group % 2 == 0) else -1.0
    out["amt"] = y + regime_sign * shift
    return out


def _forecast_and_target(
    model, test_df: pd.DataFrame, horizon: int = 10
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Aligned forecast vs test `amt` for the first `horizon` test steps."""
    if test_df.empty:
        return None
    steps = min(len(test_df), horizon)
    if steps < 1:
        return None

    exog_future = None
    exog_cols = [c for c in test_df.columns if c.startswith("exog")]
    if exog_cols:
        exog_future = np.asarray(test_df[exog_cols].values[:steps], dtype=float)

    forecast = None
    attempts = [
        (True, True),
        (True, False),
        (False, True),
        (False, False),
    ]
    for use_transform, with_exog in attempts:
        try:
            if with_exog:
                forecast = model.predict(steps=steps, exog_future=exog_future, use_transform=use_transform)
            else:
                forecast = model.predict(steps=steps, use_transform=use_transform)
            break
        except TypeError:
            try:
                forecast = model.predict(steps=steps)
                break
            except Exception:
                continue
        except Exception:
            continue

    if forecast is None:
        return None

    forecast = np.asarray(forecast, dtype=float).flatten()
    forecast = np.nan_to_num(forecast, nan=0.0, posinf=1e30, neginf=-1e30)
    target = np.asarray(test_df["amt"].values[: len(forecast)], dtype=float)
    target = np.nan_to_num(target, nan=0.0, posinf=1e30, neginf=-1e30)
    if len(target) == 0:
        return None
    return forecast, target


def evaluate_model(
    model, test_df: pd.DataFrame, horizon: int = 10
) -> float:
    """Mean absolute error (MAE) on the first `horizon` test steps (manuscript)."""
    pair = _forecast_and_target(model, test_df, horizon)
    if pair is None:
        return float("inf")
    forecast, target = pair
    err = np.abs(forecast - target)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return float("inf")
    return float(np.mean(err))


def evaluate_model_smape(
    model,
    test_df: pd.DataFrame,
    horizon: int = 10,
    eps: float = 1e-8,
    as_percent: bool = True,
) -> float:
    """
    Symmetric MAPE: mean_t  2|y - ŷ| / (|y| + |ŷ| + eps).
    If as_percent, return value in [0, 200] (conventional percentage scale).
    """
    pair = _forecast_and_target(model, test_df, horizon)
    if pair is None:
        return float("inf")
    forecast, target = pair
    denom = np.abs(target) + np.abs(forecast) + eps
    ratio = 2.0 * np.abs(target - forecast) / denom
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return float("inf")
    smape = np.mean(ratio)
    smape = float(np.clip(smape, 0.0, 2.0))
    return smape * 100.0 if as_percent else smape


def train_local_model(
    ModelClass,
    df: pd.DataFrame,
    local_epochs: int,
    local_params: Dict[str, np.ndarray],
    local_fit_maxiter: Optional[int] = 10,
    forecast_horizon: int = 10,
    strict_errors: bool = False,
) -> Tuple[Dict[str, np.ndarray], float, float, int]:
    train_df, test_df = train_test_split_series(df, test_ratio=0.2)
    model = ModelClass()
    if local_params:
        model.set_initial_params(local_params)

    fit_kwargs = {"target_col": "amt", "use_transform": _model_uses_transform(ModelClass)}
    exog_cols = [c for c in train_df.columns if c.startswith("exog")]

    model_name = ModelClass.__name__
    if exog_cols and model_name == "ARMAXModel":
        fit_kwargs["exog_cols"] = exog_cols

    fit_kwargs = _attach_fit_budget(ModelClass, fit_kwargs, local_fit_maxiter)

    for _ in range(max(1, local_epochs)):
        try:
            model.fit(train_df, **fit_kwargs)
        except Exception:
            if strict_errors:
                raise
            fallback_kwargs = dict(fit_kwargs)
            fallback_kwargs["use_transform"] = False
            model.fit(train_df, **fallback_kwargs)

    mae = evaluate_model(model, test_df, horizon=forecast_horizon)
    smape = evaluate_model_smape(model, test_df, horizon=forecast_horizon)
    params = stabilize_numeric_param_dict(model.get_params())
    if ModelClass.__name__ == "ARMAXModel":
        params = _stabilize_armax_params(params)
    if not np.isfinite(mae):
        mae = 1e12
    if not np.isfinite(smape):
        smape = 200.0
    return params, mae, smape, len(train_df)


def run_one_model(
    model_name: str,
    ModelClass,
    clients: List[pd.DataFrame],
    profiles: List[frozenset],
    aggregator: str,
    rounds: int,
    local_epochs: int,
    malicious_frac: float,
    seed: int,
    attack_strategy: str = "label_flip",
    attack_scale: float = 2.5,
    similarity_tau: float = 0.15,
    similarity_mode: str = "jaccard",
    lambda_jaccard: float = 0.5,
    tau_cos_min: float = -1.0,
    topology_mode: str = "similarity",
    lvp_alpha: Optional[float] = None,
    lvp_self_weight: float = 0.0,
    krum_f: int = -1,
    local_fit_maxiter: Optional[int] = 10,
    eval_fit_maxiter: Optional[int] = 0,
    forecast_horizon: int = 10,
    transmitted_param_norm_clip: Optional[float] = None,
    state_param_norm_clip: Optional[float] = None,
    strict_errors: bool = False,
    network_eval_mode: str = "refit",
    skip_train_rounds: int = 0,
    random_init: bool = False,
    malicious_selection: str = "random",
    regime_drift_scale: float = 0.0,
    async_stale_frac: float = 0.0,
    block_missing_frac: float = 0.0,
    num_workers: int = 1,
) -> Dict:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    n_clients = len(clients)
    n_mal = int(round(n_clients * malicious_frac))

    local_states: List[Dict[str, np.ndarray]] = [{} for _ in range(n_clients)]
    client_views: List[pd.DataFrame] = []
    for i, df in enumerate(clients):
        client_views.append(_apply_block_missing(df, block_missing_frac, seed, i))
    
    # If random_init, initialize with random parameters by one quick forward pass per client
    if random_init:
        for idx in range(n_clients):
            try:
                # Get model structure by calling with 0 epochs = just initialize
                params, _, _, _ = train_local_model(
                    ModelClass,
                    clients[idx],
                    0,  # 0 epochs → just get model structure with random init
                    {},
                    local_fit_maxiter=0,  # No optimization
                    forecast_horizon=forecast_horizon,
                    strict_errors=False,
                )
                # Add small random noise to make different
                params = {k: v + np_rng.standard_normal(v.shape) * 0.01 for k, v in (params or {}).items()}
                local_states[idx] = params
            except Exception:
                pass  # If fails, keep empty dict
    
    history: List[Dict] = []

    agg_mode = aggregator
    f_krum = _krum_f_effective(n_clients, malicious_frac, krum_f)

    topo_mode = (topology_mode or "similarity").strip().lower()
    if topo_mode in ("ring", "star", "complete", "line"):
        neighbors, kappa = _structural_neighbor_graph(n_clients, topo_mode)
    else:
        neighbors, kappa = build_neighbor_graph(profiles, similarity_tau)
    malicious_ids = _select_malicious_ids(
        n_clients,
        n_mal,
        malicious_selection,
        neighbors,
        rng,
    )
    alpha_lvp = lvp_alpha if (lvp_alpha is not None and lvp_alpha > 0) else default_alpha_from_graph(neighbors)

    # For hybrid similarity, recompute graph each round from parameter deltas.
    use_dynamic_graph = (similarity_mode or "jaccard").strip().lower() in (
        "jaccard_cosine_hybrid",
        "hybrid",
    )
    prev_round_states: Optional[List[Dict[str, np.ndarray]]] = None

    scaffold_server_w: Dict[str, np.ndarray] = {}
    scaffold_c_global: Dict[str, np.ndarray] = {}
    scaffold_c_clients: List[Dict[str, np.ndarray]] = [{} for _ in range(n_clients)]
    push_sum_x: Optional[List[Dict[str, np.ndarray]]] = None
    push_sum_w: Optional[np.ndarray] = None
    last_transmitted: List[Dict[str, np.ndarray]] = [deepcopy(s) if isinstance(s, dict) else {} for s in local_states]

    for r in range(rounds):
        prev_sync_states = [deepcopy(d) if isinstance(d, dict) else {} for d in local_states]
        raw_params: List[Dict[str, np.ndarray]] = []
        client_mae: Dict[str, float] = {}
        client_smape: Dict[str, float] = {}
        client_sizes: Dict[str, int] = {}
        quality: Dict[str, float] = {}
        scaffold_starts: List[Dict[str, np.ndarray]] = []

        theta_old = deepcopy(scaffold_server_w)

        stale_ids: set = set()
        if async_stale_frac > 0.0 and n_clients > 0:
            n_stale = int(round(float(np.clip(async_stale_frac, 0.0, 0.95)) * n_clients))
            if n_stale > 0:
                stale_ids = set(rng.sample(range(n_clients), min(n_stale, n_clients)))

        init_params_by_idx: List[Dict[str, np.ndarray]] = []
        for idx in range(n_clients):
            if agg_mode == "scaffold":
                start_i = param_dict_add(
                    theta_old,
                    param_dict_sub(scaffold_c_clients[idx], scaffold_c_global),
                )
                scaffold_starts.append(start_i)
                init_params_by_idx.append(start_i)
            else:
                scaffold_starts.append({})
                init_params_by_idx.append(local_states[idx])

        def _train_one_client(idx: int) -> Tuple[Dict[str, np.ndarray], float, float, int]:
            train_df_source = _apply_round_regime_cascade(
                client_views[idx],
                client_idx=idx,
                round_idx=r,
                rounds=rounds,
                drift_scale=regime_drift_scale,
            )

            if r < skip_train_rounds:
                return deepcopy(init_params_by_idx[idx]), 1e12, 200.0, 1

            try:
                return train_local_model(
                    ModelClass,
                    train_df_source,
                    local_epochs,
                    init_params_by_idx[idx],
                    local_fit_maxiter=local_fit_maxiter,
                    forecast_horizon=forecast_horizon,
                    strict_errors=strict_errors,
                )
            except Exception:
                if strict_errors:
                    raise
                return {}, 1e12, 200.0, 1

        worker_count = max(1, int(num_workers))
        if worker_count == 1 or n_clients <= 1:
            per_client = [_train_one_client(idx) for idx in range(n_clients)]
        else:
            with ThreadPoolExecutor(max_workers=min(worker_count, n_clients)) as pool:
                per_client = list(pool.map(_train_one_client, range(n_clients)))

        for idx, (params, mae, smape_loc, train_len) in enumerate(per_client):
            cid = f"client_{idx}"
            raw_params.append(params)
            client_mae[cid] = mae
            client_smape[cid] = smape_loc
            # FedAvg should be weighted by train sample count, not test split size.
            client_sizes[cid] = max(train_len, 1)
            denom = 1.0 + (mae if np.isfinite(mae) else 1e6)
            quality[cid] = 1.0 / np.sqrt(denom)

        transmitted: List[Dict[str, np.ndarray]] = []
        for idx, p in enumerate(raw_params):
            base_payload = p
            if idx in stale_ids and idx < len(last_transmitted) and isinstance(last_transmitted[idx], dict) and last_transmitted[idx]:
                base_payload = deepcopy(last_transmitted[idx])
            if idx in malicious_ids:
                corrupted = corrupt_params(
                    base_payload,
                    scale=attack_scale,
                    strategy=attack_strategy,
                    rng=np_rng,
                )
                transmitted.append(corrupted)
            else:
                transmitted.append(deepcopy(base_payload))
        if transmitted_param_norm_clip is not None:
            transmitted = [
                _clip_param_dict_l2(d, transmitted_param_norm_clip) if isinstance(d, dict) else d
                for d in transmitted
            ]
        last_transmitted = [deepcopy(d) if isinstance(d, dict) else {} for d in transmitted]

        used_keys: List[str] = []

        if agg_mode == "fedavg":
            client_params_dict = {f"client_{i}": transmitted[i] for i in range(n_clients)}
            filtered, used_keys = sanitize_params(client_params_dict)
            aggregated: Dict[str, np.ndarray] = {}
            if filtered:
                aggregated = fedavg_aggregate(filtered, client_sizes)
            local_states = [deepcopy(aggregated) for _ in range(n_clients)]

        elif agg_mode == "lvp":
            if topo_mode in ("ring", "star", "complete", "line"):
                pass
            elif use_dynamic_graph:
                neighbors, kappa = build_neighbor_graph(
                    profiles,
                    similarity_tau,
                    similarity_mode=similarity_mode,
                    theta_local=raw_params,
                    theta_prev=prev_round_states,
                    lambda_jaccard=lambda_jaccard,
                    tau_cos_min=tau_cos_min,
                )
            else:
                neighbors, kappa = build_neighbor_graph(
                    profiles,
                    similarity_tau,
                    similarity_mode="jaccard",
                )
            alpha_lvp = (
                lvp_alpha
                if (lvp_alpha is not None and lvp_alpha > 0)
                else default_alpha_from_graph(neighbors)
            )
            local_states = decentralized_lvp_synchronize(
                raw_params,
                transmitted,
                neighbors,
                kappa,
                alpha_lvp,
                self_weight=lvp_self_weight,
            )
            used_keys = sorted(
                set().union(*(set(d.keys()) for d in local_states if d))
            )

        elif agg_mode == "decentralized_fedavg":
            local_states = decentralized_fedavg_synchronize(
                raw_params,
                transmitted,
                neighbors,
                kappa,
            )
            used_keys = sorted(
                set().union(*(set(d.keys()) for d in local_states if d))
            )

        elif agg_mode == "defta":
            local_states = defta_synchronize(
                raw_params,
                transmitted,
                neighbors,
                kappa,
                temperature=3.0,
            )
            used_keys = sorted(
                set().union(*(set(d.keys()) for d in local_states if d))
            )

        elif agg_mode == "balance":
            local_states = decentralized_balance_synchronize(
                raw_params,
                transmitted,
                neighbors,
                kappa,
                temperature=4.0,
            )
            used_keys = sorted(
                set().union(*(set(d.keys()) for d in local_states if d))
            )

        elif agg_mode in ("push_sum", "balance_push_sum"):
            sync_fn = push_sum_synchronize if agg_mode == "push_sum" else balance_push_sum_synchronize
            local_states, push_sum_x, push_sum_w = sync_fn(
                raw_params,
                transmitted,
                neighbors,
                kappa,
                x_state=push_sum_x,
                w_state=push_sum_w,
            )
            used_keys = sorted(
                set().union(*(set(d.keys()) for d in local_states if d))
            )

        elif agg_mode in ("robust_median", "weighted_median"):
            client_params_dict = {f"client_{i}": transmitted[i] for i in range(n_clients)}
            filtered, used_keys = sanitize_params(client_params_dict)
            aggregated = {}
            if filtered:
                aggregated = weighted_median_norm_aggregate(filtered, quality)
            local_states = [deepcopy(aggregated) for _ in range(n_clients)]
            used_keys = list(aggregated.keys()) if aggregated else used_keys

        elif agg_mode == "krum":
            client_params_dict = {f"client_{i}": transmitted[i] for i in range(n_clients)}
            filtered, used_keys = sanitize_params(client_params_dict)
            aggregated = {}
            if filtered:
                aggregated = krum_aggregate(filtered, f_krum)
            local_states = [deepcopy(aggregated) for _ in range(n_clients)]
            used_keys = list(aggregated.keys()) if aggregated else used_keys

        elif agg_mode == "scaffold":
            delta_by_cid: Dict[str, Dict[str, np.ndarray]] = {}
            for i in range(n_clients):
                delta_by_cid[f"client_{i}"] = param_dict_sub(
                    transmitted[i], scaffold_starts[i]
                )
            filtered_d, used_keys = sanitize_params(delta_by_cid)
            delta_bar: Dict[str, np.ndarray] = {}
            if filtered_d:
                delta_bar = fedavg_aggregate(filtered_d, client_sizes)
            scaffold_server_w = param_dict_add(theta_old, delta_bar)
            for i in range(n_clients):
                # Use transmitted[i] (server-observed vectors), not raw_params[i]:
                # mixing honest local fits with corrupted uploads desynchronizes
                # control variates from the aggregated delta and causes blow-ups.
                scaffold_c_clients[i] = param_dict_add(
                    param_dict_sub(scaffold_c_clients[i], scaffold_c_global),
                    param_dict_sub(theta_old, transmitted[i]),
                )
            scaffold_c_global = param_dict_mean(scaffold_c_clients)
            local_states = [deepcopy(scaffold_server_w) for _ in range(n_clients)]

        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")

        local_states = [
            stabilize_numeric_param_dict(d) if isinstance(d, dict) and d else d
            for d in local_states
        ]
        if state_param_norm_clip is not None:
            local_states = [
                _clip_param_dict_l2(d, state_param_norm_clip) if isinstance(d, dict) and d else d
                for d in local_states
            ]
        sync_components = _components_undirected(neighbors)
        component_sync_l2_mean = _component_sync_delta(local_states, prev_sync_states, sync_components)
        component_sync_l2_max = 0.0
        component_sync_l2_by_component: List[float] = []
        component_sizes: List[int] = []
        if sync_components:
            all_deltas: List[float] = []
            for comp in sync_components:
                component_sizes.append(len(comp))
                comp_deltas: List[float] = []
                for idx in comp:
                    if idx < len(local_states) and idx < len(prev_sync_states):
                        d = _param_l2_distance(local_states[idx], prev_sync_states[idx])
                        all_deltas.append(d)
                        comp_deltas.append(d)
                component_sync_l2_by_component.append(float(np.mean(comp_deltas)) if comp_deltas else 0.0)
            component_sync_l2_max = float(max(all_deltas)) if all_deltas else 0.0
        component_centroid_l2_mean, component_pairwise_l2_mean = _component_consensus_metrics(
            local_states,
            sync_components,
        )
        prev_round_states = [deepcopy(d) if isinstance(d, dict) else {} for d in raw_params]

        # Network-level MAE / sMAPE after synchronization.
        if (network_eval_mode or "refit").strip().lower() == "proxy":
            network_maes = [
                float(v) if np.isfinite(v) else 1e6 for v in client_mae.values()
            ]
            network_smapes = [
                float(v) if np.isfinite(v) else 200.0 for v in client_smape.values()
            ]
        else:
            network_maes: List[float] = []
            network_smapes: List[float] = []
            for idx, df in enumerate(client_views):
                try:
                    m = ModelClass()
                    if local_states[idx]:
                        m.set_initial_params(local_states[idx])
                    train_df, test_df = train_test_split_series(df, test_ratio=0.2)
                    fit_kwargs = {
                        "target_col": "amt",
                        "use_transform": _model_uses_transform(ModelClass),
                    }
                    exog_cols = [c for c in train_df.columns if c.startswith("exog")]
                    if exog_cols and ModelClass.__name__ == "ARMAXModel":
                        fit_kwargs["exog_cols"] = exog_cols
                    fit_kwargs = _attach_fit_budget(
                        ModelClass, fit_kwargs, eval_fit_maxiter
                    )
                    try:
                        m.fit(train_df, **fit_kwargs)
                    except Exception:
                        if strict_errors:
                            raise
                        fallback_kwargs = dict(fit_kwargs)
                        fallback_kwargs["use_transform"] = False
                        m.fit(train_df, **fallback_kwargs)
                    nm = evaluate_model(m, test_df, horizon=forecast_horizon)
                    ns = evaluate_model_smape(m, test_df, horizon=forecast_horizon)
                    network_maes.append(nm if np.isfinite(nm) else 1e6)
                    network_smapes.append(ns if np.isfinite(ns) else 200.0)
                except Exception:
                    if strict_errors:
                        raise
                    network_maes.append(1e6)
                    network_smapes.append(200.0)

        network_mae = float(np.mean(network_maes)) if network_maes else 1e6
        network_smape = float(np.mean(network_smapes)) if network_smapes else 200.0

        hist_row = {
            "round": r + 1,
            "client_mae": client_mae,
            "client_smape": client_smape,
            "network_mae": network_mae,
            "network_smape": network_smape,
            "lvp_alpha": alpha_lvp if agg_mode == "lvp" else None,
            "lvp_self_weight": float(lvp_self_weight) if agg_mode == "lvp" else None,
            "krum_f": f_krum if agg_mode == "krum" else None,
            "similarity_tau": similarity_tau,
            "similarity_mode": similarity_mode,
            "lambda_jaccard": lambda_jaccard if agg_mode == "lvp" else None,
            "tau_cos_min": tau_cos_min if agg_mode == "lvp" else None,
            "topology_mode": topo_mode,
            "network_eval_mode": network_eval_mode,
            "sync_component_count": len(sync_components),
            "sync_component_l2_mean": component_sync_l2_mean,
            "sync_component_l2_max": component_sync_l2_max,
            "sync_component_centroid_l2_mean": component_centroid_l2_mean,
            "sync_component_pairwise_l2_mean": component_pairwise_l2_mean,
            "sync_component_sizes": component_sizes,
            "sync_component_l2_by_component": component_sync_l2_by_component,
            "sync_components": [list(comp) for comp in sync_components],
            "used_param_keys": used_keys,
            "aggregated": (
                {k: v.tolist() for k, v in local_states[0].items()}
                if local_states and local_states[0]
                else {}
            ),
            "malicious_ids": sorted(list(malicious_ids)),
            "async_stale_ids": sorted(list(stale_ids)),
            # Backward compatibility for plotting scripts expecting old field names
            "client_mse": client_mae,
            "server_mse": network_mae,
            "server_smape": network_smape,
        }
        history.append(hist_row)

    return {
        "model": model_name,
        "aggregator": aggregator,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "local_fit_maxiter": local_fit_maxiter,
        "eval_fit_maxiter": eval_fit_maxiter,
        "forecast_horizon": int(forecast_horizon),
        "transmitted_param_norm_clip": transmitted_param_norm_clip,
        "state_param_norm_clip": state_param_norm_clip,
        "malicious_frac": malicious_frac,
        "attack_strategy": attack_strategy,
        "attack_scale": attack_scale,
        "strict_errors": strict_errors,
        "network_eval_mode": network_eval_mode,
        "malicious_selection": malicious_selection,
        "regime_drift_scale": float(regime_drift_scale),
        "async_stale_frac": float(async_stale_frac),
        "block_missing_frac": float(block_missing_frac),
        "history": history,
        "metric": "mae",
    }


def _build_exogenous(
    repo_root: Path,
    mcc_df: pd.DataFrame,
    use_reuters: bool,
    strict_errors: bool = False,
) -> Optional[pd.DataFrame]:
    """Fontanka + optional Reuters daily sentiment (Reuters-21578 under real_data_integration)."""
    exog_cols = {}
    target_dates = pd.to_datetime(mcc_df["date"], errors="coerce").dt.normalize()
    try:
        news = load_news_exogenous(repo_root)
        if news is not None:
            news_series = pd.Series(news.values, index=pd.to_datetime(news.index, errors="coerce")).dropna()
            news_series.index = news_series.index.normalize()
            aligned_news = (
                news_series.reindex(target_dates)
                .ffill()
                .bfill()
                .fillna(0.0)
            )
            exog_cols["exog_news"] = aligned_news.values
    except Exception:
        if strict_errors:
            raise

    if use_reuters:
        for base in _reuters_integration_dirs(repo_root):
            try:
                corpus_root = base / "reuters" / "reuters"
                if corpus_root.exists():
                    reuters_df = build_reuters_daily(base, target_dates)
                    exog_cols["exog_reuters"] = reuters_df["reuters_sentiment"].values
                    break
            except Exception:
                if strict_errors:
                    raise
                continue

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
    data_source: str = "mcc",
    similarity_tau: float = 0.15,
    similarity_mode: str = "jaccard",
    lambda_jaccard: float = 0.5,
    tau_cos_min: float = -1.0,
    lvp_alpha: Optional[float] = None,
    lvp_self_weight: float = 0.0,
    krum_f: int = -1,
    local_fit_maxiter: Optional[int] = 10,
    eval_fit_maxiter: Optional[int] = 0,
    strict_errors: bool = False,
    network_eval_mode: str = "refit",
    malicious_selection: str = "random",
    regime_drift_scale: float = 0.0,
    async_stale_frac: float = 0.0,
    block_missing_frac: float = 0.0,
) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    if data_source == "moex":
        print("Loading MOEX stock data...")
        moex_df = load_moex_series(base_path)
        print(f"MOEX data loaded: {moex_df.shape}")
        exog = None
        mcc_df = None
    elif data_source == "mcc":
        print("Loading MCC transaction data...")
        mcc_df = load_mcc_series(base_path)
        exog = _build_exogenous(
            base_path,
            mcc_df,
            use_reuters,
            strict_errors=strict_errors,
        )
    elif data_source == "news":
        print("Loading news data...")
        news_df = load_news_exogenous(base_path)
        mcc_df = pd.DataFrame(
            {"date": news_df.index, "news_count": news_df.values}
        )
        exog = None
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    results: List[Dict] = []
    combo_counter = 0

    if output_path and output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            results = existing.get("results", [])
            print(f"Resuming from {len(results)} existing results")
        except Exception as e:
            print(f"Could not load existing results: {e}")

    total_combos = (
        len(model_names)
        * len(n_clients_list)
        * len(malicious_fracs)
        * len(aggregators)
        * len(rounds_list)
        * len(local_epochs_list)
    )
    if max_combos:
        total_combos = min(total_combos, max_combos)

    pbar = tqdm(total=total_combos, desc="Overall Progress", unit="exp")

    for n_clients in n_clients_list:
        if data_source == "moex":
            clients = build_clients_from_moex(moex_df, n_clients=n_clients)
        else:
            clients = build_clients_from_mcc(mcc_df, exog, n_clients=n_clients)

        if len(clients) < n_clients:
            clients = clients[: max(1, len(clients))]

        profiles = build_client_information_profiles(clients, data_source)

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
                                    payload = {
                                        "results": results + model_results,
                                        "truncated": True,
                                        "combos": combo_counter - 1,
                                    }
                                    if output_path:
                                        output_path.parent.mkdir(
                                            parents=True, exist_ok=True
                                        )
                                        output_path.write_text(
                                            json.dumps(payload, indent=2),
                                            encoding="utf-8",
                                        )
                                    return payload

                                exp = run_one_model(
                                    model_name,
                                    ModelClass,
                                    clients,
                                    profiles,
                                    aggregator=agg,
                                    rounds=rounds,
                                    local_epochs=local_epochs,
                                    malicious_frac=mal_frac,
                                    seed=seed + combo_counter,
                                    attack_strategy=attack_strategy,
                                    attack_scale=attack_scale,
                                    similarity_tau=similarity_tau,
                                    similarity_mode=similarity_mode,
                                    lambda_jaccard=lambda_jaccard,
                                    tau_cos_min=tau_cos_min,
                                    lvp_alpha=lvp_alpha,
                                    lvp_self_weight=lvp_self_weight,
                                    krum_f=krum_f,
                                    local_fit_maxiter=local_fit_maxiter,
                                    eval_fit_maxiter=eval_fit_maxiter,
                                    strict_errors=strict_errors,
                                    network_eval_mode=network_eval_mode,
                                    malicious_selection=malicious_selection,
                                    regime_drift_scale=regime_drift_scale,
                                    async_stale_frac=async_stale_frac,
                                    block_missing_frac=block_missing_frac,
                                )
                                exp["n_clients"] = len(clients)
                                model_results.append(exp)

                                final_m = np.mean(
                                    list(exp["history"][-1]["client_mae"].values())
                                )
                                pbar.update(1)
                                pbar.set_postfix(
                                    {
                                        "model": model_name,
                                        "mal": mal_frac,
                                        "agg": agg,
                                        "mae": f"{final_m:.4f}",
                                    }
                                )

            except Exception as e:
                print(f"  ERROR in {model_name}: {e}")
                import traceback

                traceback.print_exc()
                results.extend(model_results)
                if output_path:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    payload = {
                        "results": results,
                        "truncated": False,
                        "combos": combo_counter,
                    }
                    output_path.write_text(
                        json.dumps(payload, indent=2), encoding="utf-8"
                    )
                    print(
                        f"  Saved partial results ({len(results)} total) to {output_path}"
                    )
                continue

            model_time = time.time() - model_start
            results.extend(model_results)
            if output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "results": results,
                    "truncated": False,
                    "combos": combo_counter,
                }
                output_path.write_text(
                    json.dumps(payload, indent=2), encoding="utf-8"
                )
                print(
                    f"  {model_name} done in {model_time:.1f}s. Checkpoint: {len(results)} total to {output_path}"
                )

    pbar.close()
    return {"results": results, "truncated": False, "combos": combo_counter}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Federated multimodal time-series experiments (manuscript-aligned)."
    )
    p.add_argument(
        "--base-path",
        type=str,
        default=str(REPO_ROOT_DEFAULT),
        help="Repository root (expects 01_data_transactions, 02_data_fontanka)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(MODEL_REGISTRY.keys()),
        help="Model class names (e.g. ARMAXModel)",
    )
    p.add_argument(
        "--n-clients",
        nargs="+",
        type=int,
        default=[20],
        help="Client counts (manuscript uses N=20 for real-data grid)",
    )
    p.add_argument(
        "--malicious-fracs",
        nargs="+",
        type=float,
        default=[0.0, 0.2, 0.4],
        help="Fraction of clients with corrupted outbound parameters",
    )
    p.add_argument(
        "--aggregators",
        nargs="+",
        default=["fedavg", "lvp"],
        help=(
            "fedavg | lvp | decentralized_fedavg | defta | balance | push_sum | balance_push_sum (legacy alias) | krum | weighted_median | robust_median (same as weighted_median) | scaffold"
        ),
    )
    p.add_argument(
        "--rounds",
        nargs="+",
        type=int,
        default=[5, 20],
        help="Communication rounds R",
    )
    p.add_argument(
        "--local-epochs",
        nargs="+",
        type=int,
        default=[1, 3, 5],
        help="Local epochs per round",
    )
    p.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    p.add_argument(
        "--max-combos",
        type=int,
        default=None,
        help="Stop after this many experiment combinations (smoke tests)",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(ARTIFACTS / "real_federated_results.json"),
        help="Output JSON path",
    )
    p.add_argument(
        "--no-reuters", action="store_true", help="Disable Reuters exogenous channel"
    )
    p.add_argument(
        "--attack-strategy",
        type=str,
        default="label_flip",
        choices=[
            "label_flip",
            "noise",
            "noise_colluded",
            "noise_heavy_tail",
            "noise_colluded_heavy_tail",
            "random",
        ],
        help="Byzantine corruption of transmitted parameters",
    )
    p.add_argument(
        "--attack-scale", type=float, default=2.5, help="Attack scaling / noise level"
    )
    p.add_argument(
        "--data-source",
        type=str,
        default="mcc",
        choices=["mcc", "moex", "news"],
        help="Primary data modality",
    )
    p.add_argument(
        "--similarity-tau",
        type=float,
        default=0.15,
        help="Jaccard threshold τ for decentralized graph (edge if κ_ij >= τ)",
    )
    p.add_argument(
        "--similarity-mode",
        type=str,
        default="jaccard",
        choices=["jaccard", "jaccard_cosine_hybrid"],
        help="Similarity mode for LVP graph: pure Jaccard or Jaccard+cosine hybrid",
    )
    p.add_argument(
        "--lambda-jaccard",
        type=float,
        default=0.5,
        help="Hybrid mix coefficient λ in s = λ*Jaccard + (1-λ)*cos01",
    )
    p.add_argument(
        "--tau-cos-min",
        type=float,
        default=-1.0,
        help="Optional cosine gate for hybrid mode (edge only if cosine >= tau_cos_min)",
    )
    p.add_argument(
        "--lvp-alpha",
        type=float,
        default=0.0,
        help="LVP step α; if <=0, use stability heuristic from max degree",
    )
    p.add_argument(
        "--lvp-self-weight",
        type=float,
        default=0.0,
        help="LVP self-weight γ in [0,1] for damping (higher = smoother, less reactive)",
    )
    p.add_argument(
        "--krum-f",
        type=int,
        default=-1,
        help="Krum Byzantine bound f (>=0); if <0, use round(malicious_frac * n_clients)",
    )
    p.add_argument(
        "--local-fit-maxiter",
        type=int,
        default=10,
        help="Statsmodels maxiter per local fit (FL variant 1: limited optimization).",
    )
    p.add_argument(
        "--eval-fit-maxiter",
        type=int,
        default=0,
        help="Maxiter when fitting for network_mae after sync (0 ≈ evaluate at synced params).",
    )
    p.add_argument(
        "--strict-errors",
        action="store_true",
        help="Raise on local/eval model errors instead of replacing with penalty values.",
    )
    p.add_argument(
        "--network-eval-mode",
        type=str,
        default="refit",
        choices=["refit", "proxy"],
        help="'refit' computes post-sync network metrics by fitting each client model; 'proxy' uses local metrics.",
    )
    p.add_argument(
        "--malicious-selection",
        type=str,
        default="random",
        choices=["random", "hub_targeted"],
        help="How to choose malicious clients: random or highest-degree hubs in similarity graph.",
    )
    p.add_argument(
        "--regime-drift-scale",
        type=float,
        default=0.0,
        help="Cascading per-round regime drift amplitude added to client targets (0 disables).",
    )
    p.add_argument(
        "--async-stale-frac",
        type=float,
        default=0.0,
        help="Fraction of clients sending stale (previous-round) updates each round.",
    )
    p.add_argument(
        "--block-missing-frac",
        type=float,
        default=0.0,
        help="Fraction of each client series replaced by a contiguous missing/frozen block.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_path = Path(args.base_path).resolve()
    out_path = Path(args.output)
    lvp_alpha = args.lvp_alpha if args.lvp_alpha > 0 else None

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
        similarity_tau=args.similarity_tau,
        similarity_mode=args.similarity_mode,
        lambda_jaccard=args.lambda_jaccard,
        tau_cos_min=args.tau_cos_min,
        lvp_alpha=lvp_alpha,
        lvp_self_weight=float(np.clip(args.lvp_self_weight, 0.0, 1.0)),
        krum_f=args.krum_f,
        local_fit_maxiter=args.local_fit_maxiter,
        eval_fit_maxiter=args.eval_fit_maxiter,
        strict_errors=args.strict_errors,
        network_eval_mode=args.network_eval_mode,
        malicious_selection=args.malicious_selection,
        regime_drift_scale=args.regime_drift_scale,
        async_stale_frac=args.async_stale_frac,
        block_missing_frac=args.block_missing_frac,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nDone. Results: {out_path} ({len(payload['results'])} experiments)")
    if payload.get("truncated"):
        print("  WARNING: combinations truncated by max-combos")


if __name__ == "__main__":
    main()
