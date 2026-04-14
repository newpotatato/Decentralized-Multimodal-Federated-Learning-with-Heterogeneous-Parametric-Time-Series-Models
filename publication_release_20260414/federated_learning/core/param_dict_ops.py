"""Element-wise arithmetic on parameter dicts (str -> ndarray) for SCAFFOLD-style updates."""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np


def _keys_union(*dicts: Dict[str, np.ndarray]) -> Set[str]:
    s: Set[str] = set()
    for d in dicts:
        s.update(d.keys())
    return s


def param_dict_add(
    a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k in _keys_union(a, b):
        va = a.get(k)
        vb = b.get(k)
        if va is None:
            out[k] = np.asarray(vb, dtype=float).copy()
        elif vb is None:
            out[k] = np.asarray(va, dtype=float).copy()
        else:
            out[k] = np.asarray(va, dtype=float) + np.asarray(vb, dtype=float)
    return out


def param_dict_sub(
    a: Dict[str, np.ndarray], b: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """a - b; missing keys treated as zero tensors matching the other side."""
    out: Dict[str, np.ndarray] = {}
    for k in _keys_union(a, b):
        if k in a and k in b:
            out[k] = np.asarray(a[k], dtype=float) - np.asarray(b[k], dtype=float)
        elif k in a:
            out[k] = np.asarray(a[k], dtype=float).copy()
        else:
            out[k] = -np.asarray(b[k], dtype=float)
    return out


def param_dict_mean(dicts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    if not dicts:
        return {}
    keys: Set[str] = set()
    for d in dicts:
        keys.update(d.keys())
    out: Dict[str, np.ndarray] = {}
    n = len(dicts)
    for k in sorted(keys):
        ref = None
        for d in dicts:
            if k in d:
                ref = np.asarray(d[k], dtype=float)
                break
        if ref is None:
            continue
        acc = np.zeros_like(ref, dtype=float)
        for d in dicts:
            if k in d:
                acc += np.asarray(d[k], dtype=float)
        out[k] = acc / max(n, 1)
    return out
