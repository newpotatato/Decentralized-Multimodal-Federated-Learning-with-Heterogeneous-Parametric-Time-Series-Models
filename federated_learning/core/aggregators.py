from typing import Dict, List, Tuple
import numpy as np


def sanitize_params(raw_params: Dict[str, dict]) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    filtered: Dict[str, Dict[str, np.ndarray]] = {}
    shared_keys: List[str] = []

    for cid, params in raw_params.items():
        if not isinstance(params, dict):
            continue
        numeric: Dict[str, np.ndarray] = {}
        for key, val in params.items():
            try:
                arr = np.atleast_1d(np.array(val, dtype=float))
                numeric[key] = arr
            except Exception:
                continue
        if numeric:
            filtered[cid] = numeric
            if not shared_keys:
                shared_keys = list(numeric.keys())
            else:
                shared_keys = list(set(shared_keys).intersection(numeric.keys()))

    if not shared_keys:
        return {}, []

    valid_keys: List[str] = []
    for key in sorted(shared_keys):
        shapes = [filtered[cid][key].shape for cid in filtered if key in filtered[cid]]
        if shapes and all(shape == shapes[0] for shape in shapes):
            valid_keys.append(key)

    if not valid_keys:
        return {}, []

    trimmed = {cid: {k: filtered[cid][k] for k in valid_keys} for cid in filtered}
    return trimmed, valid_keys


def fedavg_aggregate(params: Dict[str, Dict[str, np.ndarray]], weights: Dict[str, float]) -> Dict[str, np.ndarray]:
    if not params:
        return {}
    total = sum(weights.values()) or 1.0
    w = {cid: weights.get(cid, 0.0) / total for cid in params}
    keys = list(next(iter(params.values())).keys())
    agg: Dict[str, np.ndarray] = {}
    for key in keys:
        stacked = np.stack([params[cid][key] for cid in params], axis=0)
        ws = np.array([w.get(cid, 0.0) for cid in params])
        if ws.sum() == 0:
            ws = np.ones_like(ws) / len(ws)
        agg[key] = np.average(stacked, axis=0, weights=ws)
    return agg


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    vals = np.asarray(values, dtype=float).flatten()
    wts = np.asarray(weights, dtype=float).flatten()
    sorter = np.argsort(vals)
    vals_sorted = vals[sorter]
    wts_sorted = wts[sorter]
    cum = np.cumsum(wts_sorted)
    cutoff = 0.5 * wts_sorted.sum()
    idx = np.searchsorted(cum, cutoff)
    idx = min(idx, len(vals_sorted) - 1)
    return float(vals_sorted[idx])


def lvp_aggregate(params: Dict[str, Dict[str, np.ndarray]], qualities: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    Robust aggregation using quality-weighted medians (Levantine Plate).
    Normalizes parameters before aggregation to handle different scales.
    """
    if not params:
        return {}
    total = sum(qualities.values()) or 1.0
    q = {cid: max(qualities.get(cid, 0.0), 0.0) / total for cid in params}
    keys = list(next(iter(params.values())).keys())
    agg: Dict[str, np.ndarray] = {}
    
    for key in keys:
        stacked = np.stack([params[cid][key] for cid in params], axis=0)
        ws = np.array([q.get(cid, 0.0) for cid in params], dtype=float)
        
        if ws.sum() == 0:
            ws = np.ones_like(ws) / len(ws)
        
        flat = stacked.reshape(stacked.shape[0], -1)
        
        # Normalize each parameter vector before aggregation
        # This prevents large-scale parameters from dominating the median
        norms = np.linalg.norm(flat, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)  # Avoid division by zero
        flat_norm = flat / norms
        
        # Compute weighted median on normalized values
        med_flat = np.array([weighted_median(flat_norm[:, j], ws) for j in range(flat_norm.shape[1])])
        
        # Denormalize back using the median norm
        median_norm = np.median(norms.flatten())
        agg[key] = (med_flat * median_norm).reshape(stacked.shape[1:])
    
    return agg


def corrupt_params(params: Dict[str, np.ndarray], scale: float = 2.0, strategy: str = "label_flip") -> Dict[str, np.ndarray]:
    """
    Corrupt parameters via different Byzantine attack strategies.
    Only corrupts numeric (float/int) parameters; string/object params left unchanged.
    
    Args:
        params: Original parameters from local training
        scale: Attack intensity (2.0 = 2x noise std)
        strategy: 'label_flip' (invert all), 'noise' (add Gaussian noise), 'random' (random values)
    
    Returns:
        Corrupted parameters
    """
    noisy: Dict[str, np.ndarray] = {}
    
    if strategy == "label_flip":
        # Classic label flipping: invert numeric parameters only
        for key, val in params.items():
            try:
                arr = np.array(val, dtype=float)
                noise = np.random.normal(0, np.std(arr) * scale + 1e-6, size=arr.shape)
                noisy[key] = -(arr + noise)
            except (ValueError, TypeError):
                # Skip non-numeric values (strings, objects, etc.)
                noisy[key] = val
    
    elif strategy == "noise":
        # Gaussian noise only (no inversion) for numeric params
        for key, val in params.items():
            try:
                arr = np.array(val, dtype=float)
                noise = np.random.normal(0, np.std(arr) * scale + 1e-6, size=arr.shape)
                noisy[key] = arr + noise
            except (ValueError, TypeError):
                noisy[key] = val
    
    elif strategy == "random":
        # Replace numeric params with random values from large distribution
        for key, val in params.items():
            try:
                arr = np.array(val, dtype=float)
                min_val = np.min(arr) - 3 * (np.std(arr) + 1e-6)
                max_val = np.max(arr) + 3 * (np.std(arr) + 1e-6)
                noisy[key] = np.random.uniform(min_val, max_val, size=arr.shape)
            except (ValueError, TypeError):
                noisy[key] = val
    
    else:
        # Default: same as original
        noisy = params
    
    return noisy
