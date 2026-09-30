"""Operational decentralized consensus baselines for federated comparison.

These implementations are intentionally lightweight and reuse the same client
neighbor graph as LVP so they can be compared under the same topology.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _cosine_similarity_param_dicts(
    a: Dict[str, np.ndarray],
    b: Dict[str, np.ndarray],
    eps: float = 1e-12,
) -> float:
    """Cosine similarity over common numeric parameter keys."""
    keys = sorted(set(a.keys()) & set(b.keys()))
    if not keys:
        return 0.0
    avec: List[np.ndarray] = []
    bvec: List[np.ndarray] = []
    for key in keys:
        aa = np.asarray(a[key], dtype=float).ravel()
        bb = np.asarray(b[key], dtype=float).ravel()
        if aa.size == 0 or bb.size == 0:
            continue
        if aa.shape != bb.shape:
            m = min(aa.size, bb.size)
            if m == 0:
                continue
            aa = aa[:m]
            bb = bb[:m]
        avec.append(aa)
        bvec.append(bb)
    if not avec:
        return 0.0
    av = np.concatenate(avec)
    bv = np.concatenate(bvec)
    denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
    if denom <= eps:
        return 0.0
    return float(np.dot(av, bv) / denom)


def _copy_param_dict(params: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=float).copy() for k, v in params.items()}


def _scale_param_dict(params: Dict[str, np.ndarray], scale: float) -> Dict[str, np.ndarray]:
    return {k: np.asarray(v, dtype=float) * float(scale) for k, v in params.items()}


def _add_scaled_param_dict(
    dest: Dict[str, np.ndarray],
    src: Dict[str, np.ndarray],
    scale: float,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {k: np.asarray(v, dtype=float).copy() for k, v in dest.items()}
    s = float(scale)
    for key, value in src.items():
        arr = np.asarray(value, dtype=float)
        if key in out:
            out[key] = out[key] + s * arr
        else:
            out[key] = s * arr.copy()
    return out


def _blend_param_dicts(
    params: Sequence[Dict[str, np.ndarray]],
    weights: Sequence[float],
) -> Dict[str, np.ndarray]:
    if not params:
        return {}
    ws = np.asarray(weights, dtype=float).flatten()
    if ws.size != len(params):
        raise ValueError("weights length must match params length")
    ws = np.clip(ws, 0.0, None)
    total = float(ws.sum())
    if total <= 0:
        ws = np.ones(len(params), dtype=float) / max(len(params), 1)
    else:
        ws = ws / total

    keys = sorted({key for p in params for key in p.keys()})
    out: Dict[str, np.ndarray] = {}
    for key in keys:
        ref = None
        for p in params:
            if key in p:
                ref = np.asarray(p[key], dtype=float)
                break
        if ref is None:
            continue
        acc = np.zeros_like(ref, dtype=float)
        for p, w in zip(params, ws):
            if key in p:
                acc += float(w) * np.asarray(p[key], dtype=float)
        out[key] = acc
    return out


def _sender_distribution(
    sender_idx: int,
    neighbors: List[List[int]],
    kappa: np.ndarray,
    mode: str,
    temperature: float,
) -> Tuple[List[int], np.ndarray]:
    outgoing = [sender_idx] + [j for j in neighbors[sender_idx] if j != sender_idx]
    if not outgoing:
        return [sender_idx], np.array([1.0], dtype=float)

    if mode == "uniform":
        raw = np.ones(len(outgoing), dtype=float)
    elif mode == "trust":
        raw_vals = [1.0]
        raw_vals.extend(max(float(kappa[sender_idx, j]), 0.0) for j in outgoing[1:])
        raw = np.asarray(raw_vals, dtype=float)
        temp = float(max(temperature, 1e-6))
        raw = np.power(np.maximum(raw, 1e-12), temp)
    else:
        raise ValueError(f"Unknown mixing mode: {mode}")

    total = float(raw.sum())
    if total <= 0:
        probs = np.ones(len(outgoing), dtype=float) / len(outgoing)
    else:
        probs = raw / total
    return outgoing, probs


def _build_send_maps(
    neighbors: List[List[int]],
    kappa: np.ndarray,
    mode: str,
    temperature: float,
) -> List[Dict[int, float]]:
    send_maps: List[Dict[int, float]] = []
    for sender_idx in range(len(neighbors)):
        outgoing, probs = _sender_distribution(sender_idx, neighbors, kappa, mode, temperature)
        send_maps.append({receiver_idx: float(prob) for receiver_idx, prob in zip(outgoing, probs)})
    return send_maps


def _mix_from_senders(
    theta_local: List[Dict[str, np.ndarray]],
    theta_sent: List[Dict[str, np.ndarray]],
    send_maps: List[Dict[int, float]],
) -> List[Dict[str, np.ndarray]]:
    n = len(theta_local)
    out: List[Dict[str, np.ndarray]] = []

    for receiver_idx in range(n):
        sender_dicts: List[Dict[str, np.ndarray]] = []
        sender_weights: List[float] = []
        for sender_idx, mapping in enumerate(send_maps):
            weight = mapping.get(receiver_idx)
            if weight is None or weight <= 0:
                continue
            sender_dicts.append(theta_local[sender_idx] if sender_idx == receiver_idx else theta_sent[sender_idx])
            sender_weights.append(float(weight))
        if not sender_dicts:
            out.append(_copy_param_dict(theta_local[receiver_idx]))
            continue
        out.append(_blend_param_dicts(sender_dicts, sender_weights))
    return out


def decentralized_fedavg_synchronize(
    theta_local: List[Dict[str, np.ndarray]],
    theta_sent: List[Dict[str, np.ndarray]],
    neighbors: List[List[int]],
    kappa: np.ndarray,
) -> List[Dict[str, np.ndarray]]:
    """Simple decentralized FedAvg: uniform averaging over self + neighbors."""
    send_maps = _build_send_maps(neighbors, kappa, mode="uniform", temperature=1.0)
    return _mix_from_senders(theta_local, theta_sent, send_maps)


def defta_synchronize(
    theta_local: List[Dict[str, np.ndarray]],
    theta_sent: List[Dict[str, np.ndarray]],
    neighbors: List[List[int]],
    kappa: np.ndarray,
    temperature: float = 3.0,
) -> List[Dict[str, np.ndarray]]:
    """Topology-adaptive decentralized averaging using similarity-weighted consensus."""
    send_maps = _build_send_maps(neighbors, kappa, mode="trust", temperature=temperature)
    return _mix_from_senders(theta_local, theta_sent, send_maps)


def decentralized_balance_synchronize(
    theta_local: List[Dict[str, np.ndarray]],
    theta_sent: List[Dict[str, np.ndarray]],
    neighbors: List[List[int]],
    kappa: np.ndarray,
    temperature: float = 4.0,
) -> List[Dict[str, np.ndarray]]:
    """BALANCE-style decentralized robust averaging.

    Each receiver uses its own local model as the reference and reweights
    incoming neighbor models by cosine similarity to this reference.
    """
    n = len(theta_local)
    send_maps = _build_send_maps(neighbors, kappa, mode="uniform", temperature=1.0)
    out: List[Dict[str, np.ndarray]] = []

    for receiver_idx in range(n):
        ref = theta_local[receiver_idx]
        sender_dicts: List[Dict[str, np.ndarray]] = [ref]
        sender_weights: List[float] = [1.0]

        for sender_idx, mapping in enumerate(send_maps):
            if sender_idx == receiver_idx:
                continue
            base_w = float(mapping.get(receiver_idx, 0.0))
            if base_w <= 0:
                continue
            incoming = theta_sent[sender_idx]
            cos = _cosine_similarity_param_dicts(ref, incoming)
            cos01 = float(np.clip((cos + 1.0) * 0.5, 0.0, 1.0))
            robust_w = base_w * max(cos01, 1e-12) ** float(max(temperature, 1e-6))
            sender_dicts.append(incoming)
            sender_weights.append(robust_w)

        out.append(_blend_param_dicts(sender_dicts, sender_weights))
    return out


def push_sum_synchronize(
    theta_local: List[Dict[str, np.ndarray]],
    theta_sent: List[Dict[str, np.ndarray]],
    neighbors: List[List[int]],
    kappa: np.ndarray,
    x_state: Optional[List[Dict[str, np.ndarray]]] = None,
    w_state: Optional[np.ndarray] = None,
    temperature: float = 1.0,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]], np.ndarray]:
    """Push-Sum ratio consensus over the same neighbor graph.

    We keep a scalar mass per client and update the latent x/w states before the
    consensus step. This gives an operational push-sum baseline without changing
    the surrounding FL training loop.
    """
    n = len(theta_local)
    if x_state is None or len(x_state) != n:
        x_state = [_copy_param_dict(p) for p in theta_local]
    if w_state is None or len(w_state) != n:
        w_state = np.ones(n, dtype=float)
    else:
        w_state = np.asarray(w_state, dtype=float).copy()

    # Inject the new local models into the current mass before the push step.
    # A client keeps its own local model; neighbors receive the transmitted one
    # (which is the attacked payload for Byzantine senders).
    prepared_own = [_scale_param_dict(theta_local[i], float(w_state[i])) for i in range(n)]
    prepared_sent = [_scale_param_dict(theta_sent[i], float(w_state[i])) for i in range(n)]
    send_maps = _build_send_maps(neighbors, kappa, mode="uniform", temperature=temperature)

    new_x: List[Dict[str, np.ndarray]] = [{} for _ in range(n)]
    new_w = np.zeros(n, dtype=float)

    for sender_idx, mapping in enumerate(send_maps):
        for receiver_idx, weight in mapping.items():
            if weight <= 0:
                continue
            payload = prepared_own[sender_idx] if receiver_idx == sender_idx else prepared_sent[sender_idx]
            if not new_x[receiver_idx]:
                new_x[receiver_idx] = _scale_param_dict(payload, float(weight))
            else:
                new_x[receiver_idx] = _add_scaled_param_dict(
                    new_x[receiver_idx], payload, float(weight)
                )
            new_w[receiver_idx] += float(weight) * float(w_state[sender_idx])

    theta_next: List[Dict[str, np.ndarray]] = []
    eps = 1e-12
    for idx in range(n):
        denom = float(new_w[idx])
        if denom <= eps or not new_x[idx]:
            theta_next.append(_copy_param_dict(theta_local[idx]))
            continue
        theta_next.append({key: np.asarray(value, dtype=float) / denom for key, value in new_x[idx].items()})

    return theta_next, new_x, new_w


def balance_push_sum_synchronize(
    theta_local: List[Dict[str, np.ndarray]],
    theta_sent: List[Dict[str, np.ndarray]],
    neighbors: List[List[int]],
    kappa: np.ndarray,
    x_state: Optional[List[Dict[str, np.ndarray]]] = None,
    w_state: Optional[np.ndarray] = None,
    temperature: float = 1.0,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]], np.ndarray]:
    """Backward-compatible alias for legacy experiment names."""
    return push_sum_synchronize(
        theta_local,
        theta_sent,
        neighbors,
        kappa,
        x_state=x_state,
        w_state=w_state,
        temperature=temperature,
    )