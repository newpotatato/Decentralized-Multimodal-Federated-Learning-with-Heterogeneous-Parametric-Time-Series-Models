#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

AGG_ORDER = ["lvp", "decentralized_fedavg", "defta", "balance", "push_sum", "balance_push_sum"]
LABEL_MAP = {
    "lvp": "LVP",
    "decentralized_fedavg": "Decentralized FedAvg",
    "defta": "DeFTA",
    "balance": "BALANCE",
    "push_sum": "Push-Sum",
    "balance_push_sum": "BALANCE Push-Sum",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build boxplots for fig3 scenario: small vs large rounds")
    p.add_argument("--small-dirs", nargs="+", required=True)
    p.add_argument("--large-dirs", nargs="+", required=True)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def _load_finals(dirs: List[str]) -> Dict[str, List[float]]:
    by_agg: Dict[str, List[float]] = {}
    for d in dirs:
        payload = json.loads((Path(d) / "fig3_decentralized_methods.json").read_text(encoding="utf-8"))
        for exp in payload["results"]:
            agg = str(exp["aggregator"])
            hist = exp.get("history", [])
            if not hist:
                continue
            value = float(hist[-1].get("network_mae", np.inf))
            by_agg.setdefault(agg, []).append(value)
    return by_agg


def _ordered_aggs(by_agg_small: Dict[str, List[float]], by_agg_large: Dict[str, List[float]]) -> List[str]:
    present = set(by_agg_small.keys()) | set(by_agg_large.keys())
    ordered = [a for a in AGG_ORDER if a in present]
    tail = sorted([a for a in present if a not in ordered])
    return ordered + tail


def _summary(by_agg: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for agg, vals in by_agg.items():
        arr = np.asarray(vals, dtype=float)
        out[agg] = {
            "n": int(arr.size),
            "mean": float(np.mean(arr)) if arr.size else float("inf"),
            "median": float(np.median(arr)) if arr.size else float("inf"),
            "std": float(np.std(arr, ddof=0)) if arr.size else float("inf"),
            "min": float(np.min(arr)) if arr.size else float("inf"),
            "max": float(np.max(arr)) if arr.size else float("inf"),
        }
    return out


def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    small = _load_finals(args.small_dirs)
    large = _load_finals(args.large_dirs)
    aggs = _ordered_aggs(small, large)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=True)

    def draw_panel(ax, by_agg: Dict[str, List[float]], title: str) -> None:
        data = [by_agg.get(a, []) for a in aggs]
        ax.boxplot(data, tick_labels=[LABEL_MAP.get(a, a) for a in aggs], patch_artist=True)
        ax.set_title(title)
        ax.set_ylabel("Final network MAE")
        ax.tick_params(axis="x", labelrotation=20)
        ax.grid(True, axis="y", alpha=0.3)

    draw_panel(axes[0], small, "Small rounds scenario")
    draw_panel(axes[1], large, "Large rounds scenario")
    plt.tight_layout()

    fig_path = out_dir / "fig3_boxplot_small_vs_large_rounds.png"
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    small_stats = _summary(small)
    large_stats = _summary(large)

    payload = {
        "small_dirs": args.small_dirs,
        "large_dirs": args.large_dirs,
        "aggregators": aggs,
        "small_rounds_summary": small_stats,
        "large_rounds_summary": large_stats,
        "figure": str(fig_path),
    }
    json_path = out_dir / "fig3_boxplot_small_vs_large_rounds.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Figure 3 Boxplot: Small vs Large Rounds",
        "",
        "## Small rounds (final MAE)",
    ]
    for agg in aggs:
        s = small_stats.get(agg)
        if not s:
            continue
        lines.append(f"- {LABEL_MAP.get(agg, agg)}: n={s['n']}, mean={s['mean']:.6f}, median={s['median']:.6f}, std={s['std']:.6f}")

    lines.append("")
    lines.append("## Large rounds (final MAE)")
    for agg in aggs:
        s = large_stats.get(agg)
        if not s:
            continue
        lines.append(f"- {LABEL_MAP.get(agg, agg)}: n={s['n']}, mean={s['mean']:.6f}, median={s['median']:.6f}, std={s['std']:.6f}")

    md_path = out_dir / "fig3_boxplot_small_vs_large_rounds_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {fig_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {md_path}")


if __name__ == "__main__":
    main()
