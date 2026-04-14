#!/usr/bin/env python3
"""Round-1 topology diagnostic for hybrid similarity (Jaccard + cosine).

Builds client profiles and first-round local model parameters, then constructs a
hybrid similarity graph and reports connected components. Optionally searches
for a tau value that yields a more balanced component-size distribution.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "core"))
sys.path.insert(0, str(_FL / "data_loaders"))

from data_utils import (  # noqa: E402
    build_client_information_profiles,
    build_clients_from_mcc,
    load_mcc_series,
)
from decentralized_lvp import build_neighbor_graph  # noqa: E402
from run_real_experiments import MODEL_REGISTRY, _build_exogenous, train_local_model  # noqa: E402


def _flatten_params(params: Dict[str, np.ndarray]) -> np.ndarray:
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


def _cosine_matrix(params_list: Sequence[Dict[str, np.ndarray]]) -> np.ndarray:
    n = len(params_list)
    vecs = [_flatten_params(p) for p in params_list]
    cos = np.ones((n, n), dtype=float)
    np.fill_diagonal(cos, 0.0)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = vecs[i]
            b = vecs[j]
            if a.size == 0 or b.size == 0:
                cos[i, j] = 1.0
                continue
            m = min(a.size, b.size)
            if m == 0:
                cos[i, j] = 1.0
                continue
            a = a[:m]
            b = b[:m]
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na <= 1e-12 or nb <= 1e-12:
                cos[i, j] = 1.0
                continue
            cos[i, j] = float(np.dot(a, b) / (na * nb))
    return np.clip(cos, -1.0, 1.0)


def _neighbors_from_hybrid(
    sim: np.ndarray,
    cos: np.ndarray,
    tau: float,
    tau_cos_min: float,
) -> List[List[int]]:
    n = sim.shape[0]
    out: List[List[int]] = []
    for i in range(n):
        nbrs: List[int] = []
        for j in range(n):
            if i == j:
                continue
            if sim[i, j] < tau:
                continue
            if cos[i, j] < tau_cos_min:
                continue
            nbrs.append(j)
        out.append(nbrs)
    return out


def _components_undirected(neighbors: List[List[int]]) -> List[List[int]]:
    n = len(neighbors)
    und = [set(row) for row in neighbors]
    for i in range(n):
        for j in neighbors[i]:
            und[j].add(i)

    seen = [False] * n
    comps: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp: List[int] = []
        while stack:
            v = stack.pop()
            comp.append(v)
            for u in und[v]:
                if not seen[u]:
                    seen[u] = True
                    stack.append(u)
        comps.append(sorted(comp))
    return sorted(comps, key=len, reverse=True)


def _balance_score(
    components: List[List[int]],
    target_components: int,
    n_clients: int,
) -> float:
    sizes = np.asarray([len(c) for c in components], dtype=float)
    if sizes.size == 0:
        return 1e9
    ideal = float(n_clients) / max(float(target_components), 1.0)
    spread = float(np.std(sizes))
    comp_gap = abs(len(components) - target_components)
    singletons = int(np.sum(sizes == 1.0))
    return spread + 3.0 * comp_gap + 2.0 * singletons + 0.5 * abs(float(np.mean(sizes)) - ideal)


def _plot_topology(
    neighbors: List[List[int]],
    components: List[List[int]],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    n = len(neighbors)
    pos: Dict[int, Tuple[float, float]] = {}
    n_comp = max(len(components), 1)
    outer_r = 5.0

    for c_idx, comp in enumerate(components):
        ang = 2.0 * math.pi * c_idx / n_comp
        cx = outer_r * math.cos(ang)
        cy = outer_r * math.sin(ang)
        inner_r = 0.8 + 0.2 * math.sqrt(max(len(comp), 1))
        for k, node in enumerate(comp):
            a = 2.0 * math.pi * k / max(len(comp), 1)
            pos[node] = (cx + inner_r * math.cos(a), cy + inner_r * math.sin(a))

    fig, ax = plt.subplots(figsize=(9, 7))
    for i in range(n):
        xi, yi = pos[i]
        for j in neighbors[i]:
            if i < j:
                xj, yj = pos[j]
                ax.plot([xi, xj], [yi, yj], color="#9aa4b2", alpha=0.55, linewidth=1.0)

    comp_id = {}
    for idx, comp in enumerate(components):
        for node in comp:
            comp_id[node] = idx
    colors = [comp_id.get(i, -1) for i in range(n)]

    xs = [pos[i][0] for i in range(n)]
    ys = [pos[i][1] for i in range(n)]
    ax.scatter(xs, ys, c=colors, cmap="tab10", s=150, edgecolors="black", linewidths=0.6)
    for i in range(n):
        ax.text(pos[i][0], pos[i][1], str(i), ha="center", va="center", fontsize=8, color="white")

    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_similarity_heatmap(sim: np.ndarray, components: List[List[int]], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    order = [idx for comp in components for idx in comp]
    mat = sim[np.ix_(order, order)] if order else sim
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Clients (grouped by component)")
    ax.set_ylabel("Clients (grouped by component)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Hybrid similarity")
    plt.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Round-1 hybrid topology diagnostic")
    p.add_argument("--base-path", type=str, default=str(_FL.parent))
    p.add_argument("--out-dir", type=str, default=str(_FL / "artifacts" / "topology_round1_hybrid"))
    p.add_argument("--model", type=str, default="DynamicLinearModel")
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--column-partition", type=str, default="strided", choices=["contiguous", "strided"])
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--local-fit-maxiter", type=int, default=10)
    p.add_argument("--similarity-tau", type=float, default=0.6)
    p.add_argument("--lambda-jaccard", type=float, default=0.6)
    p.add_argument("--tau-cos-min", type=float, default=0.15)
    p.add_argument("--sync-topic-groups", type=int, default=4)
    p.add_argument("--target-components", type=int, default=4)
    p.add_argument("--auto-balance-tau", action="store_true")
    p.add_argument("--tau-grid-min", type=float, default=0.45)
    p.add_argument("--tau-grid-max", type=float, default=0.85)
    p.add_argument("--tau-grid-steps", type=int, default=41)
    args = p.parse_args()

    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}")

    base = Path(args.base_path).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=True)
    clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=args.n_clients,
        column_partition=args.column_partition,
    )
    profiles = build_client_information_profiles(clients, "mcc")
    if args.sync_topic_groups > 0:
        profiles = [
            frozenset(pf) | {f"sync_topic_{idx % args.sync_topic_groups}"}
            for idx, pf in enumerate(profiles)
        ]

    ModelClass = MODEL_REGISTRY[args.model]
    round1_params: List[Dict[str, np.ndarray]] = []
    for df in clients:
        params, _, _, _ = train_local_model(
            ModelClass,
            df,
            local_epochs=args.local_epochs,
            local_params={},
            local_fit_maxiter=args.local_fit_maxiter,
            strict_errors=True,
        )
        round1_params.append(params)

    _, jacc = build_neighbor_graph(profiles, tau=0.0, similarity_mode="jaccard")
    cos = _cosine_matrix(round1_params)
    cos01 = 0.5 * (cos + 1.0)
    lam = float(np.clip(args.lambda_jaccard, 0.0, 1.0))
    hybrid = lam * jacc + (1.0 - lam) * cos01

    tau = float(args.similarity_tau)
    best = None
    if args.auto_balance_tau:
        for t in np.linspace(args.tau_grid_min, args.tau_grid_max, args.tau_grid_steps):
            neighbors_t = _neighbors_from_hybrid(hybrid, cos, float(t), float(args.tau_cos_min))
            comps_t = _components_undirected(neighbors_t)
            score = _balance_score(comps_t, args.target_components, len(clients))
            if best is None or score < best[0]:
                best = (score, float(t), neighbors_t, comps_t)
        assert best is not None
        tau = best[1]
        neighbors = best[2]
        components = best[3]
    else:
        neighbors = _neighbors_from_hybrid(hybrid, cos, tau, float(args.tau_cos_min))
        components = _components_undirected(neighbors)

    degrees = [len(nbrs) for nbrs in neighbors]
    component_sizes = [len(c) for c in components]

    topology_png = out_dir / "round1_hybrid_topology.png"
    heatmap_png = out_dir / "round1_hybrid_similarity_heatmap.png"
    _plot_topology(
        neighbors,
        components,
        topology_png,
        title=(
            f"Round-1 hybrid topology ({args.column_partition}) | "
            f"tau={tau:.3f}, lambda={lam:.2f}, tau_cos_min={args.tau_cos_min:.2f}"
        ),
    )
    _plot_similarity_heatmap(
        hybrid,
        components,
        heatmap_png,
        title="Round-1 hybrid similarity matrix",
    )

    payload = {
        "scenario": {
            "model": args.model,
            "n_clients": len(clients),
            "column_partition": args.column_partition,
            "local_epochs": args.local_epochs,
            "lambda_jaccard": lam,
            "tau_cos_min": float(args.tau_cos_min),
            "tau_selected": tau,
            "auto_balance_tau": bool(args.auto_balance_tau),
            "target_components": int(args.target_components),
            "sync_topic_groups": int(args.sync_topic_groups),
        },
        "graph": {
            "n_components": len(components),
            "component_sizes": component_sizes,
            "components": components,
            "degree_min": int(min(degrees)) if degrees else 0,
            "degree_max": int(max(degrees)) if degrees else 0,
            "degree_mean": float(np.mean(degrees)) if degrees else 0.0,
            "degree_std": float(np.std(degrees)) if degrees else 0.0,
        },
        "artifacts": {
            "topology_png": str(topology_png),
            "similarity_heatmap_png": str(heatmap_png),
        },
    }

    out_json = out_dir / "round1_hybrid_topology.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved: {topology_png}")
    print(f"Saved: {heatmap_png}")
    print(f"Saved: {out_json}")
    print(
        "Components:",
        payload["graph"]["n_components"],
        "sizes=",
        payload["graph"]["component_sizes"],
    )


if __name__ == "__main__":
    main()
