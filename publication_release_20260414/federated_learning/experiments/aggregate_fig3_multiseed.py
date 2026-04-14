#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate fig3_decentralized_methods outputs over multiple seeds")
    p.add_argument("--input-dirs", nargs="+", required=True)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    in_dirs = [Path(x) for x in args.input_dirs]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads = []
    for d in in_dirs:
        payloads.append(_load_json(d / "fig3_decentralized_methods.json"))

    base = payloads[0]
    best_model = base.get("best_model") or base.get("model") or "UnknownModel"
    scenario = base["scenario"]

    aggs = ["lvp", "decentralized_fedavg", "defta", "balance", "push_sum"]
    label_map = {
        "lvp": "LVP",
        "decentralized_fedavg": "Decentralized FedAvg",
        "defta": "DeFTA",
        "balance": "BALANCE",
        "push_sum": "Push-Sum",
    }

    mean_curves: Dict[str, List[float]] = {}
    std_curves: Dict[str, List[float]] = {}
    final_stats: Dict[str, Dict[str, float]] = {}

    for agg in aggs:
        curves = []
        finals = []
        for p in payloads:
            exp = next(x for x in p["results"] if x["aggregator"] == agg)
            ys = [float(r["network_mae"]) for r in exp["history"]]
            curves.append(ys)
            finals.append(float(exp["history"][-1]["network_mae"]))
        arr = np.asarray(curves, dtype=float)
        mean_curves[agg] = arr.mean(axis=0).tolist()
        std_curves[agg] = arr.std(axis=0, ddof=0).tolist()
        final_stats[agg] = {
            "mean": float(np.mean(finals)),
            "std": float(np.std(finals, ddof=0)),
        }

    rounds = list(range(1, len(next(iter(mean_curves.values()))) + 1))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for agg in aggs:
        ys = np.asarray(mean_curves[agg], dtype=float)
        sd = np.asarray(std_curves[agg], dtype=float)
        lw = 3.0 if agg == "lvp" else 1.8
        alpha = 1.0 if agg == "lvp" else 0.9
        ax.plot(rounds, ys, "o-", linewidth=lw, markersize=4.5, alpha=alpha, label=label_map[agg])
        ax.fill_between(rounds, ys - sd, ys + sd, alpha=0.12)

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Network MAE (absolute, amt units)")
    ax.set_title(
        f"Network MAE vs round — decentralized methods ({best_model})"
        f" (mean±std over {len(payloads)} seeds)"
    )
    if len(rounds) <= 25:
        ax.set_xticks(rounds)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    fig_path = out_dir / "fig3_decentralized_methods_multiseed_network_mae.png"
    fig.savefig(fig_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    ordered = sorted(final_stats.items(), key=lambda kv: kv[1]["mean"])

    summary = {
        "n_seeds": len(payloads),
        "seed_values": [int(p["scenario"]["seed"]) for p in payloads],
        "best_model": best_model,
        "scenario": scenario,
        "final_network_mae_mean_std": final_stats,
        "final_network_mae_ranked_by_mean": [
            {"aggregator": name, "mean": vals["mean"], "std": vals["std"]}
            for name, vals in ordered
        ],
        "figure": str(fig_path),
    }
    (out_dir / "fig3_decentralized_methods_multiseed.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Figure 3 Multi-seed Aggregate",
        "",
        f"Best model: {best_model}",
        f"Seeds: {', '.join(str(s) for s in summary['seed_values'])}",
        f"Scenario: partition={scenario['column_partition']}, mal={scenario['malicious_frac']:.2f}, attack={scenario['attack_strategy']}, scale={scenario['attack_scale']}",
        "",
        "## Final MAE by method (mean ± std)",
    ]
    for name, vals in ordered:
        lines.append(f"- {name}: {vals['mean']:.6f} ± {vals['std']:.6f}")
    (out_dir / "fig3_decentralized_methods_multiseed_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    print(f"Saved: {fig_path}")
    print(f"Saved: {out_dir / 'fig3_decentralized_methods_multiseed.json'}")
    print(f"Saved: {out_dir / 'fig3_decentralized_methods_multiseed_report.md'}")


if __name__ == "__main__":
    main()
