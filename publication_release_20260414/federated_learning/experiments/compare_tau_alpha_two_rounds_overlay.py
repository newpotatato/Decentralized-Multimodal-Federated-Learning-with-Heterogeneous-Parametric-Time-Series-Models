#!/usr/bin/env python3
"""Overlay comparison plots for tau and alpha across two round budgets.

Input: the JSON produced by ablate_selected_scenario_tau_alpha_two_rounds.py.
Output:
- tau_overlay_rounds_10_30.png
- alpha_overlay_rounds_10_30.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay tau/alpha ablation curves for two round settings")
    p.add_argument(
        "--input-json",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "artifacts"
            / "selected_scenario_tau_alpha_ablation_two_rounds_run"
            / "selected_scenario_tau_alpha_ablation_two_rounds.json"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(
            Path(__file__).resolve().parent.parent
            / "artifacts"
            / "selected_scenario_tau_alpha_ablation_two_rounds_overlay"
        ),
    )
    return p.parse_args()


def _load(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _curve(summary_rows: List[Dict], rounds: int, key: str) -> List[Dict]:
    return sorted([r for r in summary_rows if int(r["rounds"]) == rounds], key=lambda r: float(r[key]))


def _plot_overlay_x(
    summary_rows: List[Dict],
    key: str,
    xlabel: str,
    title: str,
    out_path: Path,
    rounds_small: int = 10,
    rounds_large: int = 30,
) -> None:
    fig, ax = plt.subplots(figsize=(8.9, 5.0))
    styles = {
        rounds_small: dict(color="C0", marker="o", label=f"rounds={rounds_small}"),
        rounds_large: dict(color="C3", marker="s", label=f"rounds={rounds_large}"),
    }

    for rounds in [rounds_small, rounds_large]:
        sub = _curve(summary_rows, rounds, key)
        xs = [float(r[key]) for r in sub]
        ys = [float(r["mean_final_mae"]) for r in sub]
        es = [float(r["std_final_mae"]) for r in sub]
        ax.errorbar(
            xs,
            ys,
            yerr=es,
            fmt=f"{styles[rounds]['marker']}-",
            color=styles[rounds]["color"],
            linewidth=2.2,
            markersize=5,
            capsize=4,
            label=styles[rounds]["label"],
        )

        # annotate the minimum point for each rounds setting
        best = min(sub, key=lambda r: float(r["mean_final_mae"]))
        ax.scatter([float(best[key])], [float(best["mean_final_mae"])], color=styles[rounds]["color"], s=70, zorder=4)
        ax.annotate(
            f"best {rounds}",
            xy=(float(best[key]), float(best["mean_final_mae"])),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=8,
            color=styles[rounds]["color"],
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Final network MAE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = _load(input_json)
    summary_rows = payload["tau_summary"]
    alpha_rows = payload["alpha_summary"]

    _plot_overlay_x(
        summary_rows,
        key="similarity_tau",
        xlabel="similarity_tau",
        title="Selected scenario: error vs similarity_tau (overlay of rounds=10 and rounds=30)",
        out_path=out_dir / "tau_overlay_rounds_10_30.png",
    )
    _plot_overlay_x(
        alpha_rows,
        key="lvp_alpha",
        xlabel="lvp_alpha",
        title="Selected scenario: error vs lvp_alpha (overlay of rounds=10 and rounds=30)",
        out_path=out_dir / "alpha_overlay_rounds_10_30.png",
    )

    report = out_dir / "overlay_report.md"
    report.write_text(
        "\n".join(
            [
                "# Overlay comparison",
                "",
                f"Input JSON: {input_json}",
                "",
                "Outputs:",
                "- tau_overlay_rounds_10_30.png",
                "- alpha_overlay_rounds_10_30.png",
            ]
        ),
        encoding="utf-8",
    )

    print(f"Saved: {out_dir / 'tau_overlay_rounds_10_30.png'}")
    print(f"Saved: {out_dir / 'alpha_overlay_rounds_10_30.png'}")
    print(f"Saved: {report}")


if __name__ == "__main__":
    main()
