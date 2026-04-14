#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "core"))
sys.path.insert(0, str(_FL / "data_loaders"))

from data_utils import build_client_information_profiles, build_clients_from_mcc, load_mcc_series
from decentralized_lvp import build_neighbor_graph
from run_real_experiments import _build_exogenous, _components_undirected


def _plot_topology(neighbors: List[List[int]], components: List[List[int]], out_path: Path, title: str) -> None:
    n = len(neighbors)

    # Place components around a large circle; nodes of each component around a local circle.
    pos: Dict[int, Tuple[float, float]] = {}
    n_comp = max(len(components), 1)
    outer_r = 5.0

    for c_idx, comp in enumerate(components):
        ang = 2.0 * math.pi * c_idx / n_comp
        cx = outer_r * math.cos(ang)
        cy = outer_r * math.sin(ang)
        inner_r = 1.0 + 0.25 * math.sqrt(max(len(comp), 1))
        for k, node in enumerate(comp):
            a = 2.0 * math.pi * k / max(len(comp), 1)
            pos[node] = (cx + inner_r * math.cos(a), cy + inner_r * math.sin(a))

    fig, ax = plt.subplots(figsize=(10.5, 8.2))

    # Draw undirected edges once.
    for i in range(n):
        xi, yi = pos[i]
        for j in neighbors[i]:
            if i < j:
                xj, yj = pos[j]
                ax.plot([xi, xj], [yi, yj], color="#8b98a9", alpha=0.25, linewidth=0.8, zorder=1)

    comp_id = {}
    for idx, comp in enumerate(components):
        for node in comp:
            comp_id[node] = idx

    xs = [pos[i][0] for i in range(n)]
    ys = [pos[i][1] for i in range(n)]
    colors = [comp_id.get(i, -1) for i in range(n)]

    ax.scatter(xs, ys, c=colors, cmap="tab10", s=180, edgecolors="black", linewidths=0.7, zorder=2)
    for i in range(n):
        ax.text(pos[i][0], pos[i][1], str(i), ha="center", va="center", fontsize=8, color="white", zorder=3)

    ax.set_title(title)
    ax.set_aspect("equal", "box")
    ax.axis("off")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize client topology graph (Jaccard)")
    p.add_argument(
        "--pkg-dir",
        type=str,
        default=str(_FL / "artifacts" / "article_package_current_run_20260410" / "experiment_with_more_rounds"),
    )
    p.add_argument("--tau", type=float, default=None)
    p.add_argument("--n-clients", type=int, default=20)
    args = p.parse_args()

    pkg = Path(args.pkg_dir)
    cfg_path = pkg / "raw" / "seeds" / "seed42" / "fig3_decentralized_methods.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    scenario = cfg.get("scenario", {})
    tau = float(args.tau) if args.tau is not None else float(cfg.get("similarity_tau", 0.35))

    base = _FL.parent
    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=True)
    clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=int(args.n_clients),
        column_partition=str(scenario.get("column_partition", "contiguous")),
    )
    profiles = build_client_information_profiles(clients, "mcc")

    neighbors, _ = build_neighbor_graph(profiles, tau=tau, similarity_mode="jaccard")
    components = _components_undirected(neighbors)

    # Undirected edge count for subtitle
    edge_set = set()
    for i, row in enumerate(neighbors):
        for j in row:
            edge_set.add(tuple(sorted((i, j))))

    title = (
        f"Client topology graph (Jaccard, tau={tau:.2f}) | "
        f"nodes={len(clients)}, edges={len(edge_set)}, components={len(components)}"
    )

    out_dir = pkg / "plots" / "coherence" / "components"
    fig_path = out_dir / f"topology_graph_tau_{tau:.2f}.png"
    _plot_topology(neighbors, components, fig_path, title)

    summary = {
        "tau": tau,
        "n_nodes": len(clients),
        "n_edges": len(edge_set),
        "n_components": len(components),
        "component_sizes": [len(c) for c in components],
        "figure": str(fig_path),
    }
    (out_dir / f"topology_graph_tau_{tau:.2f}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
