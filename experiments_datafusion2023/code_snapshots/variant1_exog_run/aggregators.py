from copy import deepcopy
import hashlib
from typing import Dict, List, Optional, Tuple

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


def weighted_median_norm_aggregate(
    params: Dict[str, Dict[str, np.ndarray]], qualities: Dict[str, float]
) -> Dict[str, np.ndarray]:
    """Coordinate-wise quality-weighted median after per-client L2 normalization of stacked flats."""
    return lvp_aggregate(params, qualities)


def krum_aggregate(
    params: Dict[str, Dict[str, np.ndarray]], f: int
) -> Dict[str, np.ndarray]:
    """
    Single-Krum (Blanchard et al.): pick one client whose parameter vector is closest
    to its m nearest neighbors in Euclidean distance, m = n - f_eff - 2.
    f is an upper bound on Byzantine workers; clamped so that n > 2*f_eff + 2 when possible.
    """
    if not params:
        return {}
    cids = list(params.keys())
    n = len(cids)
    if n == 1:
        return deepcopy(params[cids[0]])

    f_eff = int(max(0, min(f, max(0, (n - 3) // 2))))
    m = n - f_eff - 2
    if m < 1:
        m = 1
    m = min(m, n - 1)

    keys = list(next(iter(params.values())).keys())
    vecs: List[np.ndarray] = []
    for cid in cids:
        parts = [np.asarray(params[cid][k], dtype=float).ravel() for k in keys]
        vecs.append(np.concatenate(parts))

    dist_mat = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_mat[i, j] = np.linalg.norm(vecs[i] - vecs[j])

    scores = np.zeros(n, dtype=float)
    for i in range(n):
        d_sorted = np.sort(dist_mat[i])
        m_eff = min(m, max(0, n - 1))
        scores[i] = float(np.sum(d_sorted[1 : 1 + m_eff]))

    best = int(np.argmin(scores))
    return deepcopy(params[cids[best]])


def lvp_aggregate(params: Dict[str, Dict[str, np.ndarray]], qualities: Dict[str, float]) -> Dict[str, np.ndarray]:
    """
    Legacy robust server aggregation: coordinate-wise quality-weighted median after per-client
    L2 normalization of flattened tensors. Not the decentralized Local Voting Protocol from
    Amelina et al. (2015); kept for backward compatibility (aggregator name: robust_median).
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


def corrupt_params(
    params: Dict[str, np.ndarray],
    scale: float = 2.0,
    strategy: str = "label_flip",
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Corrupt parameters via different Byzantine attack strategies.
    Only corrupts numeric (float/int) parameters; string/object params left unchanged.
    
    Args:
        params: Original parameters from local training
        scale: Attack intensity (2.0 = 2x noise std)
        strategy: 'label_flip', 'noise' (zero-mean Gaussian), 'noise_colluded' (noise +
            colluded bias per key), 'random'
    
    Returns:
        Corrupted parameters
    """
    noisy: Dict[str, np.ndarray] = {}
    local_rng = rng if rng is not None else np.random.default_rng()
    
    if strategy == "label_flip":
        # Classic label flipping: invert numeric parameters only
        for key, val in params.items():
            try:
                arr = np.array(val, dtype=float)
                noise = local_rng.normal(0, np.std(arr) * scale + 1e-6, size=arr.shape)
                noisy[key] = -(arr + noise)
            except (ValueError, TypeError):
                # Skip non-numeric values (strings, objects, etc.)
                noisy[key] = val
    
    elif strategy == "noise":
        # Gaussian noise only (no inversion) for numeric params
        for key, val in params.items():
            try:
                arr = np.array(val, dtype=float)
                noise = local_rng.normal(0, np.std(arr) * scale + 1e-6, size=arr.shape)
                noisy[key] = arr + noise
            except (ValueError, TypeError):
                noisy[key] = val

    elif strategy == "noise_colluded":
        # Same zero-mean Gaussian as `noise`, plus a bias vector identical for every
        # malicious client on a given key (collusion via deterministic RNG from key).
        # Independent noise averages partly under FedAvg; the shared bias does not.
        for key, val in params.items():
            try:
                arr = np.array(val, dtype=float)
                s = float(np.std(arr) * scale + 1e-6)
                noise = local_rng.normal(0, s, size=arr.shape)
                h = hashlib.sha256(str(key).encode("utf-8")).digest()
                seed = int.from_bytes(h[:4], "little") & 0x7FFFFFFF
                key_rng = np.random.default_rng(seed)
                u = key_rng.standard_normal(size=arr.shape)
                u = u / (np.linalg.norm(u.ravel()) + 1e-12)
                bias_vec = u * (s * 0.75)
                noisy[key] = arr + noise + bias_vec
            except (ValueError, TypeError):
                noisy[key] = val

    elif strategy == "noise_heavy_tail":
        # Heavy-tailed corruption (Student-t) to generate occasional extreme outliers.
        for key, val in params.items():
            try:
                arr = np.array(val, dtype=float)
                s = float(np.std(arr) * scale + 1e-6)
                noise = local_rng.standard_t(df=2.0, size=arr.shape) * (s / np.sqrt(2.0))
                noisy[key] = arr + noise
            except (ValueError, TypeError):
                noisy[key] = val

    elif strategy == "noise_colluded_heavy_tail":
        # Colluded heavy-tailed attack: shared deterministic direction + heavy-tail shocks.
        for key, val in params.items():
            try:
                arr = np.array(val, dtype=float)
                s = float(np.std(arr) * scale + 1e-6)
                noise = local_rng.standard_t(df=2.0, size=arr.shape) * (s / np.sqrt(2.0))
                h = hashlib.sha256(str(key).encode("utf-8")).digest()
                seed = int.from_bytes(h[:4], "little") & 0x7FFFFFFF
                key_rng = np.random.default_rng(seed)
                u = key_rng.standard_normal(size=arr.shape)
                u = u / (np.linalg.norm(u.ravel()) + 1e-12)
                bias_vec = u * (s * 1.25)
                noisy[key] = arr + noise + bias_vec
            except (ValueError, TypeError):
                noisy[key] = val

    elif strategy == "random":
        # Replace numeric params with random values from large distribution
        for key, val in params.items():
            try:
                arr = np.array(val, dtype=float)
                min_val = np.min(arr) - 3 * (np.std(arr) + 1e-6)
                max_val = np.max(arr) + 3 * (np.std(arr) + 1e-6)
                noisy[key] = local_rng.uniform(min_val, max_val, size=arr.shape)
            except (ValueError, TypeError):
                noisy[key] = val
    
    else:
        # Default: same as original
        noisy = params
    
    return noisy
