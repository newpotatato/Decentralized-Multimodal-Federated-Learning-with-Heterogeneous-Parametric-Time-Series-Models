#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def rolling_mean(values: List[float], window: int = 3) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    out = np.zeros_like(arr)
    half = window // 2
    for i in range(len(arr)):
        l = max(0, i - half)
        r = min(len(arr), i + half + 1)
        out[i] = float(np.mean(arr[l:r]))
    return out


def relative_change(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    base = arr[0]
    if not np.isfinite(base) or abs(base) < 1e-12:
        return np.zeros_like(arr)
    return (arr / base - 1.0) * 100.0


def extract_lvp_model_series(results_json: Dict) -> Tuple[str, Dict[str, List[float]]]:
    best_model = results_json["best_model_selection"]["best_model"]
    series: Dict[str, List[float]] = {}
    for rec in results_json.get("lvp_results", []):
        if rec.get("aggregator") != "lvp":
            continue
        model_name = rec.get("model", "unknown")
        ys = [float(h["network_mae"]) for h in rec.get("history", [])]
        series[model_name] = ys
    return best_model, series


def extract_aggregator_series_for_best_model(results_json: Dict) -> Tuple[str, Dict[str, List[float]]]:
    best_model = results_json["best_model_selection"]["best_model"]
    series: Dict[str, List[float]] = {}
    for rec in results_json.get("aggregator_results", []):
        if rec.get("model") != best_model:
            continue
        agg = rec.get("aggregator", "unknown")
        ys = [float(h["network_mae"]) for h in rec.get("history", [])]
        series[agg] = ys
    return best_model, series


def top_lvp_jumps(ys: List[float], k: int = 6) -> List[Tuple[int, float]]:
    if len(ys) < 2:
        return []
    jumps = [(i + 2, ys[i + 1] - ys[i]) for i in range(len(ys) - 1)]
    jumps.sort(key=lambda x: abs(x[1]), reverse=True)
    return jumps[:k]


def extract_best_lvp_history(results_json: Dict, best_model: str) -> List[Dict[str, Any]]:
    for rec in results_json.get("lvp_results", []):
        if rec.get("model") == best_model and rec.get("aggregator") == "lvp":
            return list(rec.get("history", []))
    return []


def mean_client_mae_series(history: List[Dict[str, Any]]) -> np.ndarray:
    means: List[float] = []
    for row in history:
        client_mae = row.get("client_mae") or {}
        vals = [float(v) for v in client_mae.values() if np.isfinite(float(v))]
        means.append(float(np.mean(vals)) if vals else float("nan"))
    return np.asarray(means, dtype=float)


def comparison_series(report_json: Dict) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {}
    for agg, rec in (report_json.get("stage_b_comparison") or {}).items():
        out[agg] = {
            "mean_mae": list(rec.get("mean_mae") or []),
            "std_mae": list(rec.get("std_mae") or []),
        }
    return out


def smooth_for_trend(values: List[float], window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    w = max(3, int(window))
    med = np.zeros_like(arr)
    half = w // 2
    for i in range(arr.size):
        l = max(0, i - half)
        r = min(arr.size, i + half + 1)
        med[i] = float(np.median(arr[l:r]))
    return rolling_mean(med.tolist(), window=3)


def main() -> None:
    p = argparse.ArgumentParser(description="Create article-style 5-figure report (article dynamics + decentralized comparison).")
    p.add_argument("--input", type=str, required=True, help="Path to article_pipeline_one_seed_results.json")
    p.add_argument("--comparison-input", type=str, required=True, help="Path to honest comparison report JSON")
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--smooth-window", type=int, default=3)
    p.add_argument("--comparison-smooth-window", type=int, default=3)
    args = p.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = json.loads(in_path.read_text(encoding="utf-8"))
    comparison_payload = json.loads(Path(args.comparison_input).read_text(encoding="utf-8"))
    best_model, model_series_lvp = extract_lvp_model_series(payload)
    _, series = extract_aggregator_series_for_best_model(payload)
    if not model_series_lvp:
        raise RuntimeError("No LVP model series found in input JSON")
    if not series:
        raise RuntimeError("No aggregator comparison series found in input JSON")

    lvp_history = extract_best_lvp_history(payload, best_model)
    if not lvp_history:
        raise RuntimeError(f"No LVP history found for best model {best_model}")

    rounds = list(range(1, len(next(iter(model_series_lvp.values()))) + 1))

    # Figure 1: absolute MAE and relative change from round 1
    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(14, 5.6))
    for model_name, ys in sorted(model_series_lvp.items()):
        ys_arr = np.asarray(ys, dtype=float)
        ys_sm = rolling_mean(ys, window=max(1, int(args.smooth_window)))
        rel = relative_change(ys)
        lw = 3.0 if model_name == best_model else 1.7
        alpha = 1.0 if model_name == best_model else 0.7
        ax_abs.plot(rounds, ys_arr, "o-", linewidth=1.0, alpha=alpha, label=f"{model_name} raw")
        ax_abs.plot(rounds, ys_sm, "-", linewidth=lw, label=f"{model_name} smooth")
        ax_rel.plot(rounds, rel, "o-", linewidth=lw, markersize=4, alpha=alpha, label=model_name)

    ax_abs.set_xlabel("Communication round")
    ax_abs.set_ylabel("Network MAE")
    ax_abs.set_title(f"LVP model selection dynamics | best={best_model}")
    ax_abs.grid(True, alpha=0.25)
    ax_abs.legend(loc="best", fontsize=8, ncol=2)

    ax_rel.axhline(0.0, color="black", linewidth=0.9, linestyle="--", alpha=0.5)
    ax_rel.set_xlabel("Communication round")
    ax_rel.set_ylabel("Change vs round 1 (%)")
    ax_rel.set_title("Relative movement from round 1 (LVP only)")
    ax_rel.grid(True, alpha=0.25)
    ax_rel.legend(loc="best", fontsize=8)

    plt.tight_layout()
    fig_path = out_dir / "fig3_article_raw_vs_smoothed.png"
    fig.savefig(fig_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: mean local error over agents for the best LVP model
    local_mean = mean_client_mae_series(lvp_history)
    local_sm = rolling_mean(local_mean.tolist(), window=max(1, int(args.smooth_window)))
    fig2, ax2 = plt.subplots(figsize=(10, 5.0))
    ax2.plot(rounds, local_mean, "o-", linewidth=2.0, markersize=4, label="Mean local MAE")
    ax2.plot(rounds, local_sm, "-", linewidth=2.6, label="Smoothed mean local MAE")
    ax2.set_xlabel("Communication round")
    ax2.set_ylabel("Mean local MAE")
    ax2.set_title(f"Local agent error dynamics — {best_model} + LVP")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig2_path = out_dir / "fig3_article_local_mae_agents.png"
    fig2.savefig(fig2_path, dpi=180, bbox_inches="tight")
    plt.close(fig2)

    # Figure 3: synchronization dynamics by dynamic component graph
    sync_vals = np.asarray([float(row.get("sync_component_l2_mean", 0.0)) for row in lvp_history], dtype=float)
    sync_sm = rolling_mean(sync_vals.tolist(), window=max(1, int(args.smooth_window)))
    comp_counts = np.asarray([float(row.get("sync_component_count", 0.0)) for row in lvp_history], dtype=float)
    fig3, ax3 = plt.subplots(figsize=(10, 5.0))
    ax3.plot(rounds, sync_vals, "o-", linewidth=2.0, markersize=4, label="Mean component ΔL2")
    ax3.plot(rounds, sync_sm, "-", linewidth=2.6, label="Smoothed component ΔL2")
    ax3.set_xlabel("Communication round")
    ax3.set_ylabel("Mean component parameter change (L2)")
    ax3.set_title(f"Synchronization dynamics — {best_model} + LVP")
    ax3.grid(True, alpha=0.25)
    ax3b = ax3.twinx()
    ax3b.plot(rounds, comp_counts, "^--", color="tab:orange", linewidth=1.5, markersize=4, alpha=0.85, label="Connected components")
    ax3b.set_ylabel("Connected components")
    h3, l3 = ax3.get_legend_handles_labels()
    h3b, l3b = ax3b.get_legend_handles_labels()
    ax3.legend(h3 + h3b, l3 + l3b, loc="best", fontsize=8)
    plt.tight_layout()
    fig3_path = out_dir / "fig3_article_sync_dynamics.png"
    fig3.savefig(fig3_path, dpi=180, bbox_inches="tight")
    plt.close(fig3)

    comparison = comparison_series(comparison_payload)
    if not comparison:
        raise RuntimeError("No decentralized comparison series found in comparison report")

    # Figure 4: decentralized comparison on normal scale (raw + light smoothing)
    fig4, ax4 = plt.subplots(figsize=(10, 5.2))
    label_map = {
        "lvp": "LVP",
        "decentralized_fedavg": "Decentralized FedAvg",
        "defta": "DeFTA",
        "balance": "BALANCE",
        "push_sum": "Push-Sum",
    }
    for agg, rec in comparison.items():
        ys_raw = np.asarray(rec["mean_mae"], dtype=float)
        ys = smooth_for_trend(ys_raw.tolist(), window=max(3, int(args.comparison_smooth_window)))
        xs = np.arange(1, len(ys) + 1)
        lw = 2.8 if agg == "lvp" else 1.7
        alpha = 0.95 if agg == "lvp" else 0.75
        raw_alpha = 0.50 if agg == "lvp" else 0.35
        ax4.plot(xs, ys_raw, "--", linewidth=1.0, alpha=raw_alpha)
        ax4.plot(xs, ys, "o-", linewidth=lw, markersize=4, alpha=alpha, label=label_map.get(agg, agg))
    ax4.set_xlabel("Communication round")
    ax4.set_ylabel("Network MAE")
    ax4.set_title("Decentralized comparison on the tuned scenario (raw + smoothed)")
    ax4.grid(True, alpha=0.25)
    ax4.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig4_path = out_dir / "fig3_article_decentralized_comparison_linear.png"
    fig4.savefig(fig4_path, dpi=180, bbox_inches="tight")
    plt.close(fig4)

    # Figure 5: decentralized comparison on log scale (raw + light smoothing)
    fig5, ax5 = plt.subplots(figsize=(10, 5.2))
    for agg, rec in comparison.items():
        ys_raw = np.asarray(rec["mean_mae"], dtype=float)
        ys = smooth_for_trend(ys_raw.tolist(), window=max(3, int(args.comparison_smooth_window)))
        xs = np.arange(1, len(ys) + 1)
        lw = 2.8 if agg == "lvp" else 1.7
        alpha = 0.95 if agg == "lvp" else 0.75
        raw_alpha = 0.50 if agg == "lvp" else 0.35
        ax5.plot(xs, ys_raw, "--", linewidth=1.0, alpha=raw_alpha)
        ax5.plot(xs, ys, "o-", linewidth=lw, markersize=4, alpha=alpha, label=label_map.get(agg, agg))
    ax5.set_yscale("log")
    ax5.set_xlabel("Communication round")
    ax5.set_ylabel("Network MAE (log scale)")
    ax5.set_title("Decentralized comparison on the tuned scenario (log scale, raw + smoothed)")
    ax5.grid(True, alpha=0.25, which="both")
    ax5.legend(loc="best", fontsize=8)
    plt.tight_layout()
    fig5_path = out_dir / "fig3_article_decentralized_comparison_log.png"
    fig5.savefig(fig5_path, dpi=180, bbox_inches="tight")
    plt.close(fig5)

    top = top_lvp_jumps([float(x) for x in local_mean.tolist() if np.isfinite(x)], k=6)

    report_lines = [
        "# Figure 3 Article Diagnostics",
        "",
        f"Input: {in_path}",
        f"Comparison input: {args.comparison_input}",
        f"Best model: {best_model}",
        "",
        "## Final MAE by aggregator",
    ]
    finals = sorted(((agg, vals[-1]) for agg, vals in series.items() if vals), key=lambda x: x[1])
    for agg, v in finals:
        report_lines.append(f"- {agg}: {v:.6f}")

    report_lines.append("")
    report_lines.append("## Relative change by aggregator")
    for agg, vals in sorted(series.items()):
        rel = relative_change(vals)
        if len(rel) == 0:
            continue
        report_lines.append(f"- {agg}: round1={rel[0]:+.4f}%, round_last={rel[-1]:+.4f}%")

    report_lines.append("")
    report_lines.append("## Synchronization summary")
    if len(sync_vals) > 0:
        report_lines.append(f"- mean sync delta first round: {sync_vals[0]:.6f}")
        report_lines.append(f"- mean sync delta last round: {sync_vals[-1]:.6f}")
        report_lines.append(f"- average connected components: {float(np.mean(comp_counts)):.3f}")
    report_lines.append("")
    report_lines.append("## Top LVP jump rounds (absolute delta of local MAE)")
    for r, d in top:
        sign = "+" if d >= 0 else ""
        report_lines.append(f"- round {r}: {sign}{d:.6f}")

    report_lines.append("")
    report_lines.append("## Interpretation")
    report_lines.append("- Absolute MAE is large, so the raw lines can look nearly straight even when the model still moves.")
    report_lines.append("- Relative and delta views are the better article figures for showing convergence and spikes.")
    report_lines.append("- The synchronization metric measures round-to-round parameter movement inside each connected component.")
    report_lines.append("- Decentralized comparison plots (Figure 4 and 5) now show raw dashed curves and lightly smoothed trends.")
    report_lines.append("- This preserves readability without hiding real round-to-round movement in baselines.")

    report_path = out_dir / "fig3_article_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved: {fig_path}")
    print(f"Saved: {fig2_path}")
    print(f"Saved: {fig3_path}")
    print(f"Saved: {fig4_path}")
    print(f"Saved: {fig5_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
