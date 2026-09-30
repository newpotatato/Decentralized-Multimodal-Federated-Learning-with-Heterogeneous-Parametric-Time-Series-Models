"""
Decentralized Local Voting Protocol (LVP) synchronization per Amelina et al. (2015),
as used in the manuscript: peer-to-peer pulls toward neighbors' transmitted parameters.

θ_i^{t+1} = (1-γ) * [ θ_i^t + α * Σ_{j∈N_i(t)} b_ij^t ( θ_j^{sent,t} - θ_i^t ) ]
            + γ * θ_i^t

where γ in [0,1] is an explicit self-weight (damping) term.

Trust weights: ˜b_ij = κ_ij 1{j∈N_i},  b_ij = ˜b_ij / max(1, Σ_k ˜b_ik),  κ_ij = Jaccard(S_i, S_j).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def jaccard_similarity(profile_i: frozenset, profile_j: frozenset) -> float:
    if not profile_i and not profile_j:
        return 1.0
    inter = len(profile_i & profile_j)
    uni = len(profile_i | profile_j)
    return float(inter) / float(uni) if uni else 0.0


def build_neighbor_graph(
    profiles: List[frozenset],
    tau: float,
    similarity_mode: str = "jaccard",
    theta_local: Optional[List[Dict[str, np.ndarray]]] = None,
    theta_prev: Optional[List[Dict[str, np.ndarray]]] = None,
    lambda_jaccard: float = 0.5,
    tau_cos_min: float = -1.0,
    theta_sent: Optional[List[Dict[str, np.ndarray]]] = None,
    theta_sent_prev: Optional[List[Dict[str, np.ndarray]]] = None,
) -> Tuple[List[List[int]], np.ndarray]:
    """
    (i,j) in G iff κ_ij >= τ, i != j.
    Returns adjacency lists and full κ matrix.

    In the hybrid mode row i is the receiver: it compares its own increment with the
    increment of the vectors actually transmitted by j (theta_sent), when those are given.
    """
    n = len(profiles)
    mode = (similarity_mode or "jaccard").strip().lower()

    jacc = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            jacc[i, j] = jaccard_similarity(profiles[i], profiles[j])

    if mode == "jaccard":
        sim = jacc
        cos = None
    elif mode in ("jaccard_cosine_hybrid", "hybrid"):
        lam = float(np.clip(lambda_jaccard, 0.0, 1.0))
        cos = _cosine_similarity_from_deltas(theta_local, theta_prev, theta_sent, theta_sent_prev)
        # Map cosine [-1, 1] to [0, 1] for convex mixing with Jaccard.
        cos01 = 0.5 * (cos + 1.0)
        sim = lam * jacc + (1.0 - lam) * cos01
    elif mode in ("jaccard_paramcos_hybrid", "hybrid_param"):
        # Receiver-side consistency: cosine between the vector received from j and the
        # receiver's own locally trained parameters (both observable by client i).
        lam = float(np.clip(lambda_jaccard, 0.0, 1.0))
        cos = _cosine_received_vs_own(theta_local, theta_sent if theta_sent is not None else theta_local)
        sim = lam * jacc + (1.0 - lam) * 0.5 * (cos + 1.0)
    else:
        raise ValueError(
            f"Unknown similarity_mode={similarity_mode!r}; use 'jaccard', 'jaccard_cosine_hybrid' "
            "or 'jaccard_paramcos_hybrid'"
        )

    neighbors: List[List[int]] = []
    for i in range(n):
        n_i: List[int] = []
        for j in range(n):
            if i == j:
                continue
            if sim[i, j] < tau:
                continue
            if cos is not None and cos[i, j] < tau_cos_min:
                continue
            n_i.append(j)
        neighbors.append(n_i)
    return neighbors, sim


def _cosine_received_vs_own(
    theta_own: Optional[List[Dict[str, np.ndarray]]],
    theta_received: Optional[List[Dict[str, np.ndarray]]],
) -> np.ndarray:
    """cos[i, j] = cosine(own parameters of i, vector received from j); 1 when undefined."""
    n = len(theta_own or theta_received or [])
    cos = np.ones((n, n), dtype=float)
    np.fill_diagonal(cos, 0.0)
    if not theta_own or not theta_received:
        return cos
    own = [_flatten_numeric_param_dict(p) for p in theta_own]
    rec = [_flatten_numeric_param_dict(p) for p in theta_received]
    for i in range(n):
        for j in range(n):
            if i == j or own[i].size == 0 or rec[j].size == 0:
                continue
            m = min(own[i].size, rec[j].size)
            a, b = own[i][:m], rec[j][:m]
            na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
            if na > 1e-12 and nb > 1e-12:
                cos[i, j] = float(np.dot(a, b) / (na * nb))
    return np.clip(cos, -1.0, 1.0)


def _flatten_numeric_param_dict(params: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
    if not params:
        return np.array([], dtype=float)
    parts: List[np.ndarray] = []
    for key in sorted(params.keys()):
        try:
            arr = np.asarray(params[key], dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        if arr.size:
            parts.append(arr)
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def _param_deltas(
    theta_now: List[Dict[str, np.ndarray]],
    theta_before: List[Dict[str, np.ndarray]],
) -> List[np.ndarray]:
    deltas: List[np.ndarray] = []
    for now, before in zip(theta_now, theta_before):
        a = _flatten_numeric_param_dict(now)
        b = _flatten_numeric_param_dict(before)
        if a.size == 0 or b.size == 0:
            deltas.append(np.array([], dtype=float))
            continue
        m = min(a.size, b.size)
        deltas.append(a[:m] - b[:m])
    return deltas


def _cosine_similarity_from_deltas(
    theta_local: Optional[List[Dict[str, np.ndarray]]],
    theta_prev: Optional[List[Dict[str, np.ndarray]]],
    theta_sent: Optional[List[Dict[str, np.ndarray]]] = None,
    theta_sent_prev: Optional[List[Dict[str, np.ndarray]]] = None,
) -> np.ndarray:
    """
    Build cosine similarity matrix between client parameter deltas.
    If deltas are unavailable (e.g., first round), return matrix of ones off-diagonal,
    which effectively falls back to Jaccard-only thresholding for that round.
    """
    if not theta_local or not theta_prev or len(theta_local) != len(theta_prev):
        n = len(theta_local or theta_prev or [])
        out = np.ones((n, n), dtype=float)
        np.fill_diagonal(out, 0.0)
        return out

    n = len(theta_local)
    deltas = _param_deltas(theta_local, theta_prev)
    # Increments of the neighbors as seen by a receiver: from transmitted vectors if available.
    if theta_sent is not None and theta_sent_prev is not None and len(theta_sent) == n == len(theta_sent_prev):
        sent_deltas = _param_deltas(theta_sent, theta_sent_prev)
    else:
        sent_deltas = deltas

    cos = np.ones((n, n), dtype=float)
    np.fill_diagonal(cos, 0.0)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            di = deltas[i]
            dj = sent_deltas[j]
            if di.size == 0 or dj.size == 0:
                cos[i, j] = 1.0
                continue
            m = min(di.size, dj.size)
            if m == 0:
                cos[i, j] = 1.0
                continue
            ui = di[:m]
            uj = dj[:m]
            ni = float(np.linalg.norm(ui))
            nj = float(np.linalg.norm(uj))
            if ni <= 1e-12 or nj <= 1e-12:
                cos[i, j] = 1.0
                continue
            cos[i, j] = float(np.dot(ui, uj) / (ni * nj))
    return np.clip(cos, -1.0, 1.0)


def _trust_weights_for_neighbors(
    i: int, neighbor_js: List[int], kappa: np.ndarray
) -> Dict[int, float]:
    """b_ij = ˜b_ij / max(1, Σ_k ˜b_ik) with ˜b_ij = κ_ij on edges."""
    if not neighbor_js:
        return {}
    raw = [float(kappa[i, j]) for j in neighbor_js]
    s = max(1.0, sum(raw))
    return {j: r / s for j, r in zip(neighbor_js, raw)}


def decentralized_lvp_synchronize(
    theta_local: List[Dict[str, np.ndarray]],
    theta_sent: List[Dict[str, np.ndarray]],
    neighbors: List[List[int]],
    kappa: np.ndarray,
    alpha: float,
    self_weight: float = 0.0,
) -> List[Dict[str, np.ndarray]]:
    """
    One round of decentralized LVP. theta_local[i] = post-local-training θ_i;
    theta_sent[j] = vector actually broadcast by j (may be Byzantine-corrupted).
    """
    n = len(theta_local)
    gamma = float(np.clip(self_weight, 0.0, 1.0))
    out: List[Dict[str, np.ndarray]] = []

    for i in range(n):
        theta_i = theta_local[i]
        n_i = neighbors[i]
        if not n_i:
            out.append({k: v.copy() for k, v in theta_i.items()})
            continue

        b_map = _trust_weights_for_neighbors(i, n_i, kappa)
        new_theta: Dict[str, np.ndarray] = {}

        for key, arr_i in theta_i.items():
            acc = np.zeros_like(arr_i, dtype=float)
            wsum = 0.0
            for j in n_i:
                if key not in theta_sent[j]:
                    continue
                b_ij = b_map.get(j, 0.0)
                if b_ij <= 0:
                    continue
                acc += b_ij * (theta_sent[j][key] - arr_i)
                wsum += b_ij
            if wsum > 0:
                candidate = arr_i + alpha * acc
                # Blend with self-state to damp oscillations under noisy/colluded rounds.
                new_theta[key] = (1.0 - gamma) * candidate + gamma * arr_i
            else:
                new_theta[key] = arr_i.copy()

        out.append(new_theta)

    return out


def default_alpha_from_graph(neighbors: List[List[int]]) -> float:
    """Stability heuristic: α < 1/Δ_max (use slightly smaller)."""
    degrees = [len(n) for n in neighbors] or [0]
    d_max = max(degrees) if degrees else 0
    if d_max <= 0:
        return 0.5
    return min(0.45, 0.95 / float(d_max))
