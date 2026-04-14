#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np

ORDER = ["lvp", "decentralized_fedavg", "defta", "balance", "push_sum"]
LABEL_MAP = {
    "lvp": "LVP",
    "decentralized_fedavg": "Decentralized FedAvg",
    "defta": "DeFTA",
    "balance": "BALANCE",
    "push_sum": "Push-Sum",
}


def _extract_series(payload: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    for rec in payload.get("results", []):
        agg = str(rec.get("aggregator", ""))
        hist = rec.get("history") or []
        rounds = np.asarray([int(r.get("round", 0)) for r in hist], dtype=float)
        shift = np.asarray([float(r.get("sync_component_l2_mean", 0.0)) for r in hist], dtype=float)
        centroid = np.asarray(
            [float(r.get("sync_component_centroid_l2_mean", 0.0)) for r in hist],
            dtype=float,
        )
        pairwise = np.asarray(
            [float(r.get("sync_component_pairwise_l2_mean", 0.0)) for r in hist],
            dtype=float,
        )
        components = np.asarray([float(r.get("sync_component_count", 0.0)) for r in hist], dtype=float)
        if rounds.size:
            out[agg] = {
                "rounds": rounds,
                "shift": shift,
                "centroid": centroid,
                "pairwise": pairwise,
                "components": components,
            }
    return out


def _aggregate(seed_payloads: List[Dict[str, Any]]) -> Dict[str, Dict[str, np.ndarray]]:
    agg_series: Dict[str, Dict[str, np.ndarray]] = {}
    for agg in ORDER:
        curves_shift: List[np.ndarray] = []
        curves_centroid: List[np.ndarray] = []
        curves_pairwise: List[np.ndarray] = []
        curves_k: List[np.ndarray] = []
        rounds_ref: np.ndarray | None = None
        for payload in seed_payloads:
            s = _extract_series(payload)
            if agg not in s:
                continue
            rounds = s[agg]["rounds"]
            shift = s[agg]["shift"]
            centroid = s[agg]["centroid"]
            pairwise = s[agg]["pairwise"]
            comp = s[agg]["components"]
            if rounds_ref is None:
                rounds_ref = rounds
            min_len = min(len(rounds_ref), len(rounds))
            rounds_ref = rounds_ref[:min_len]
            curves_shift = [c[:min_len] for c in curves_shift]
            curves_centroid = [c[:min_len] for c in curves_centroid]
            curves_pairwise = [c[:min_len] for c in curves_pairwise]
            curves_k = [k[:min_len] for k in curves_k]
            curves_shift.append(shift[:min_len])
            curves_centroid.append(centroid[:min_len])
            curves_pairwise.append(pairwise[:min_len])
            curves_k.append(comp[:min_len])
        if not curves_shift or rounds_ref is None:
            continue
        arr_shift = np.asarray(curves_shift, dtype=float)
        arr_centroid = np.asarray(curves_centroid, dtype=float)
        arr_pairwise = np.asarray(curves_pairwise, dtype=float)
        arr_k = np.asarray(curves_k, dtype=float)
        agg_series[agg] = {
            "rounds": rounds_ref,
            "shift_mean": np.mean(arr_shift, axis=0),
            "shift_std": np.std(arr_shift, axis=0, ddof=0),
            "centroid_mean": np.mean(arr_centroid, axis=0),
            "centroid_std": np.std(arr_centroid, axis=0, ddof=0),
            "pairwise_mean": np.mean(arr_pairwise, axis=0),
            "pairwise_std": np.std(arr_pairwise, axis=0, ddof=0),
            "components_mean": np.mean(arr_k, axis=0),
            "components_std": np.std(arr_k, axis=0, ddof=0),
        }
    return agg_series


def _plot_metric(
    agg_series: Dict[str, Dict[str, np.ndarray]],
    out_path: Path,
    key_mean: str,
    key_std: str,
    y_label: str,
    title: str,
) -> None:
    fig, ax1 = plt.subplots(figsize=(10.5, 5.8))
    rounds_for_ticks: np.ndarray | None = None

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, agg in enumerate(ORDER):
        if agg not in agg_series:
            continue
        rec = agg_series[agg]
        x = rec["rounds"]
        if rounds_for_ticks is None:
            rounds_for_ticks = np.asarray(x, dtype=float)
        y = rec[key_mean]
        sd = rec[key_std]
        color = colors[idx % len(colors)] if colors else None

        lw = 2.8 if agg == "lvp" else 1.7
        ax1.plot(x, y, marker="o", linewidth=lw, markersize=4, color=color, label=LABEL_MAP.get(agg, agg))
        ax1.fill_between(x, y - sd, y + sd, alpha=0.15, color=color)

    ax1.set_xlabel("Communication round")
    ax1.set_ylabel(y_label)
    ax1.set_title(title)
    if rounds_for_ticks is not None and rounds_for_ticks.size:
        # For short horizons, show every round explicitly to avoid misleading sparse tick labels.
        if rounds_for_ticks.size <= 25:
            ax1.set_xticks(rounds_for_ticks)
    ax1.grid(True, alpha=0.3)

    h, l = ax1.get_legend_handles_labels()
    ax1.legend(h, l, loc="best", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot multiseed within-component shift")
    p.add_argument("--input-dirs", nargs="+", required=True, help="Seed directories containing fig3_decentralized_methods.json")
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payloads: List[Dict[str, Any]] = []
    seeds: List[int] = []
    for d in args.input_dirs:
        pth = Path(d) / "fig3_decentralized_methods.json"
        payload = json.loads(pth.read_text(encoding="utf-8"))
        payloads.append(payload)
        seeds.append(int(payload.get("scenario", {}).get("seed", -1)))

    agg_series = _aggregate(payloads)
    if not agg_series:
        raise RuntimeError("No coherence series found in provided inputs")

    fig_path_shift = out_dir / "fig3_within_component_shift_multiseed.png"
    _plot_metric(
        agg_series,
        fig_path_shift,
        key_mean="shift_mean",
        key_std="shift_std",
        y_label="Mean within-component shift (L2)",
        title="Within-component shift by round (mean+-std across seeds)",
    )

    fig_path_centroid = out_dir / "fig3_within_component_centroid_multiseed.png"
    _plot_metric(
        agg_series,
        fig_path_centroid,
        key_mean="centroid_mean",
        key_std="centroid_std",
        y_label="Mean distance to component centroid (L2)",
        title="Within-component centroid distance by round (mean+-std across seeds)",
    )

    fig_path_pairwise = out_dir / "fig3_within_component_pairwise_multiseed.png"
    _plot_metric(
        agg_series,
        fig_path_pairwise,
        key_mean="pairwise_mean",
        key_std="pairwise_std",
        y_label="Mean within-component pairwise distance (L2)",
        title="Within-component pairwise distance by round (mean+-std across seeds)",
    )

    lines = [
        "# Within-Component Consensus Metrics Multi-seed Report",
        "",
        f"Seeds: {', '.join(str(s) for s in seeds)}",
        "",
        "## Final round metrics (mean +- std)",
    ]
    ranked_shift = []
    ranked_centroid = []
    ranked_pairwise = []
    for agg, rec in agg_series.items():
        ranked_shift.append(
            (
                agg,
                float(rec["shift_mean"][-1]),
                float(rec["shift_std"][-1]),
                float(np.mean(rec["shift_mean"])),
            )
        )
        ranked_centroid.append(
            (
                agg,
                float(rec["centroid_mean"][-1]),
                float(rec["centroid_std"][-1]),
                float(np.mean(rec["centroid_mean"])),
            )
        )
        ranked_pairwise.append(
            (
                agg,
                float(rec["pairwise_mean"][-1]),
                float(rec["pairwise_std"][-1]),
                float(np.mean(rec["pairwise_mean"])),
            )
        )

    ranked_shift.sort(key=lambda x: x[1])
    ranked_centroid.sort(key=lambda x: x[1])
    ranked_pairwise.sort(key=lambda x: x[1])

    lines.append("### Shift (round-to-round)")
    for agg, last_mean, last_std, mean_all in ranked_shift:
        lines.append(f"- {agg}: last={last_mean:.6f} +- {last_std:.6f}, mean={mean_all:.6f}")

    lines.append("")
    lines.append("### Centroid distance (consensus)")
    for agg, last_mean, last_std, mean_all in ranked_centroid:
        lines.append(f"- {agg}: last={last_mean:.6f} +- {last_std:.6f}, mean={mean_all:.6f}")

    lines.append("")
    lines.append("### Pairwise distance (consensus)")
    for agg, last_mean, last_std, mean_all in ranked_pairwise:
        lines.append(f"- {agg}: last={last_mean:.6f} +- {last_std:.6f}, mean={mean_all:.6f}")

    report_path = out_dir / "fig3_within_component_shift_multiseed_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {fig_path_shift}")
    print(f"Saved: {fig_path_centroid}")
    print(f"Saved: {fig_path_pairwise}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
