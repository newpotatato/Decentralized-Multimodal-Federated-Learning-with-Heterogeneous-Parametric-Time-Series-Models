#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


ORDER = ["lvp", "decentralized_fedavg", "defta", "balance", "push_sum"]
LABEL = {
    "lvp": "LVP",
    "decentralized_fedavg": "Decentralized FedAvg",
    "defta": "DeFTA",
    "balance": "BALANCE",
    "push_sum": "Push-Sum",
}


def _load_seed_rows(input_dirs: List[str]) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for d in input_dirs:
        path = Path(d) / "fig3_decentralized_methods.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        row: Dict[str, float] = {}
        for exp in payload.get("results", []):
            agg = str(exp.get("aggregator", ""))
            hist = exp.get("history") or []
            if not hist:
                continue
            val = float(hist[-1].get("network_mae", np.nan))
            if np.isfinite(val):
                row[agg] = val
        if row:
            rows.append(row)
    return rows


def _panel_aggs(seed_rows: List[Dict[str, float]], include_fedavg: bool, excluded_aggs: List[str]) -> List[str]:
    present = set()
    for row in seed_rows:
        present.update(row.keys())
    return [
        a
        for a in ORDER
        if a in present and (include_fedavg or a != "decentralized_fedavg") and a not in excluded_aggs
    ]


def _normalize_seed_values(values: np.ndarray, mode: str, lvp_idx: int) -> np.ndarray:
    out = values.astype(float).copy()
    if mode == "none":
        return out
    mask = np.isfinite(out)
    if not np.any(mask):
        return out
    if mode == "minmax_per_seed":
        lo = float(np.min(out[mask]))
        hi = float(np.max(out[mask]))
        span = hi - lo
        if span <= 1e-15:
            out[mask] = 0.0
            return out
        out[mask] = (out[mask] - lo) / span
        return out
    if mode == "ratio_to_lvp_per_seed":
        if lvp_idx < 0 or lvp_idx >= out.size or not np.isfinite(out[lvp_idx]) or abs(out[lvp_idx]) <= 1e-15:
            out[:] = np.nan
            return out
        out[mask] = out[mask] / float(out[lvp_idx])
        return out
    if mode == "rank_per_seed":
        valid_idx = np.where(mask)[0]
        if valid_idx.size == 0:
            return out
        vals = out[valid_idx]
        order = np.argsort(vals)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(order.size, dtype=float)
        denom = max(order.size - 1, 1)
        out[:] = np.nan
        out[valid_idx] = ranks / float(denom)
        return out
    raise ValueError(f"Unknown normalization mode: {mode}")


def _build_panel_data(
    seed_rows: List[Dict[str, float]],
    include_fedavg: bool,
    excluded_aggs: List[str],
    normalize: str,
) -> Tuple[List[str], List[List[float]]]:
    aggs = _panel_aggs(seed_rows, include_fedavg, excluded_aggs)
    if not aggs:
        return [], []
    lvp_idx = aggs.index("lvp") if "lvp" in aggs else -1
    per_agg: Dict[str, List[float]] = {a: [] for a in aggs}

    for row in seed_rows:
        vals = np.asarray([float(row.get(a, np.nan)) for a in aggs], dtype=float)
        vals = _normalize_seed_values(vals, normalize, lvp_idx)
        for idx, agg in enumerate(aggs):
            if np.isfinite(vals[idx]):
                per_agg[agg].append(float(vals[idx]))

    return aggs, [per_agg[a] for a in aggs]


def _panel(
    ax: plt.Axes,
    seed_rows: List[Dict[str, float]],
    include_fedavg: bool,
    title: str,
    excluded_aggs: List[str],
    show_zoom_inset: bool,
    normalize: str,
    show_seed_points: bool,
) -> None:
    aggs, data = _build_panel_data(seed_rows, include_fedavg, excluded_aggs, normalize)
    if not aggs:
        raise RuntimeError("No data available for selected panel configuration")
    labels = [LABEL.get(a, a) for a in aggs]
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
    palette = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#76b7b2"]
    for idx, patch in enumerate(bp.get("boxes", [])):
        patch.set_facecolor(palette[idx % len(palette)])
        patch.set_alpha(0.45)
    for median in bp.get("medians", []):
        median.set_color("#1a1a1a")
        median.set_linewidth(2.0)
    for mean in bp.get("means", []):
        mean.set_marker("D")
        mean.set_markerfacecolor("black")
        mean.set_markeredgecolor("black")
        mean.set_markersize(4)
    for line in bp.get("whiskers", []) + bp.get("caps", []):
        line.set_linewidth(1.15)

    if show_seed_points:
        rng = np.random.default_rng(12345)
        for pos, vals in enumerate(data, start=1):
            arr = np.asarray(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            x = pos + rng.uniform(-0.08, 0.08, size=arr.size)
            ax.scatter(x, arr, s=28, c="#111111", alpha=0.75, zorder=3, linewidths=0)

    ax.set_title(title)
    y_label = "Final network MAE"
    if normalize == "minmax_per_seed":
        y_label = "Normalized MAE (min-max per seed, [0,1])"
    elif normalize == "ratio_to_lvp_per_seed":
        y_label = "Normalized MAE (ratio to LVP per seed)"
    elif normalize == "rank_per_seed":
        y_label = "Normalized MAE rank (best=0, worst=1 per seed)"
    ax.set_ylabel(y_label)
    ax.tick_params(axis="x", labelrotation=18)
    ax.grid(True, axis="y", alpha=0.3)

    flat = np.asarray([v for grp in data for v in grp], dtype=float)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return

    lo = float(np.min(flat))
    hi = float(np.max(flat))
    pad = max((hi - lo) * 0.08, 1.0)
    ax.set_ylim(lo - pad, hi + pad)

    if show_zoom_inset:
        inset = inset_axes(ax, width="38%", height="38%", loc="upper right", borderpad=1.1)
        inset.boxplot(data, tick_labels=labels, patch_artist=True, showmeans=True)
        inset.set_xlim(0.5, len(data) + 0.5)
        inset.set_ylim(lo - pad * 0.25, hi + pad * 0.25)
        inset.tick_params(axis="x", labelsize=7, rotation=15)
        inset.tick_params(axis="y", labelsize=7)
        inset.grid(True, axis="y", alpha=0.2)
        inset.set_title("Zoom", fontsize=8)


def main() -> None:
    p = argparse.ArgumentParser(description="Boxplot with vs without FedAvg over multi-seed final MAE")
    p.add_argument("--input-dirs", nargs="+", required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument(
        "--exclude-aggregators",
        type=str,
        default="",
        help="Comma-separated aggregator ids to exclude (e.g. push_sum,decentralized_fedavg)",
    )
    p.add_argument(
        "--single-panel",
        action="store_true",
        help="Generate one boxplot panel instead of with/without FedAvg pair",
    )
    p.add_argument(
        "--show-zoom-inset",
        action="store_true",
        help="Add a zoom inset to the single-panel plot for readability",
    )
    p.add_argument(
        "--show-seed-points",
        action="store_true",
        help="Overlay individual seed values as jittered points on top of boxplots",
    )
    p.add_argument(
        "--normalize",
        type=str,
        default="none",
        choices=["none", "minmax_per_seed", "ratio_to_lvp_per_seed", "rank_per_seed"],
        help="Optional normalization before boxplot: none, minmax_per_seed, ratio_to_lvp_per_seed, rank_per_seed",
    )
    p.add_argument(
        "--method-labels",
        type=str,
        default="",
        help="Comma-separated key=value pairs to rename method labels (e.g., 'lvp=LVPFL,defta=DeFTA-v2')",
    )
    p.add_argument(
        "--panel-title",
        type=str,
        default="",
        help="Custom title for single-panel plot",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply custom method labels if provided
    custom_labels = {}
    if args.method_labels:
        for pair in args.method_labels.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                custom_labels[k.strip()] = v.strip()
    if custom_labels:
        LABEL.update(custom_labels)

    seed_rows = _load_seed_rows(args.input_dirs)
    if not seed_rows:
        raise RuntimeError("No final MAE values found in input dirs")

    excluded = [x.strip() for x in args.exclude_aggregators.split(",") if x.strip()]

    if args.single_panel:
        fig, ax = plt.subplots(1, 1, figsize=(8.6, 5.8), sharey=False)
        single_panel_title = args.panel_title.strip() if args.panel_title else "Final MAE across seeds"
        _panel(
            ax,
            seed_rows,
            include_fedavg=True,
            title=single_panel_title,
            excluded_aggs=excluded,
            show_zoom_inset=args.show_zoom_inset,
            normalize=args.normalize,
            show_seed_points=args.show_seed_points,
        )
        fig_path = out_dir / "fig3_boxplot_single_multiseed.png"
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), sharey=False)
        _panel(
            axes[0],
            seed_rows,
            include_fedavg=True,
            title="With FedAvg (final MAE across seeds)",
            excluded_aggs=excluded,
            show_zoom_inset=False,
            normalize=args.normalize,
            show_seed_points=args.show_seed_points,
        )
        _panel(
            axes[1],
            seed_rows,
            include_fedavg=False,
            title="Without FedAvg (final MAE across seeds)",
            excluded_aggs=excluded,
            show_zoom_inset=False,
            normalize=args.normalize,
            show_seed_points=args.show_seed_points,
        )
        fig_path = out_dir / "fig3_boxplot_with_vs_without_fedavg_multiseed.png"
    if args.single_panel and args.show_zoom_inset:
        fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.14)
    else:
        plt.tight_layout()
    fig.savefig(fig_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    report = out_dir / "fig3_boxplot_with_vs_without_fedavg_multiseed_report.md"
    lines = [
        "# Boxplot With vs Without FedAvg (Multiseed)",
        "",
        f"Seeds used: {len(args.input_dirs)} runs",
        f"Excluded aggregators: {', '.join(excluded) if excluded else '(none)'}",
        f"Single panel: {bool(args.single_panel)}",
        f"Normalization: {args.normalize}",
        f"Show seed points: {bool(args.show_seed_points)}",
        "",
        "## Final MAE mean +- std (raw)",
    ]
    raw_by_agg: Dict[str, List[float]] = {a: [] for a in ORDER}
    for row in seed_rows:
        for agg in ORDER:
            val = row.get(agg)
            if val is not None and np.isfinite(float(val)):
                raw_by_agg[agg].append(float(val))
    for agg in ORDER:
        if agg in excluded:
            continue
        vals = raw_by_agg.get(agg)
        if not vals:
            continue
        arr = np.asarray(vals, dtype=float)
        lines.append(f"- {LABEL.get(agg, agg)}: {float(np.mean(arr)):.6f} +- {float(np.std(arr, ddof=0)):.6f}")

    norm_aggs, norm_data = _build_panel_data(seed_rows, include_fedavg=True, excluded_aggs=excluded, normalize=args.normalize)
    if norm_aggs and args.normalize != "none":
        lines.extend(["", "## Mean +- std after normalization (panel data)"])
        for agg, vals in zip(norm_aggs, norm_data):
            arr = np.asarray(vals, dtype=float)
            if arr.size == 0:
                continue
            lines.append(f"- {LABEL.get(agg, agg)}: {float(np.mean(arr)):.6f} +- {float(np.std(arr, ddof=0)):.6f}")
    report.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {fig_path}")
    print(f"Saved: {report}")


if __name__ == "__main__":
    main()