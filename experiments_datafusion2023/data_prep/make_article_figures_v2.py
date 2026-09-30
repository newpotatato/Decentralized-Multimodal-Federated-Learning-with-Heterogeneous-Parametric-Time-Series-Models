#!/usr/bin/env python3
"""Figures of the article from artifacts_pipeline_v2 (ablations and round-10 box plots)."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_v2 import B, M, SEEDS, final, load  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "img_for_article_v2"
LABELS = {
    "lvp": "LVP-FL", "decentralized_fedavg": "Dec. FedAvg", "defta": "DeFTA",
    "balance": "BALANCE", "push_sum": "Push-Sum", "local": "No exchange",
}
COLORS = {
    "lvp": "#4c72b0", "decentralized_fedavg": "#dd8452", "defta": "#55a868",
    "balance": "#c44e52", "push_sum": "#64b5cd", "local": "#8c8c8c",
}
LINE, BAND, BEST = "#2171b5", "#9ecae1", "#d62728"


def ablation_plot(rows, key, xlabel, title, path):
    xs = np.array([r[key] for r in rows])
    ys = np.array([r["mean"] for r in rows])
    sd = np.array([r["std"] for r in rows])
    best = int(np.argmin(ys))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    log_scale = ys.max() / ys.min() > 20
    lo = np.maximum(ys - sd, ys.min() * 0.5) if log_scale else ys - sd
    ax.fill_between(xs, lo, ys + sd, color=BAND, alpha=0.40, label=r"$\pm 1\,\sigma$ (across seeds)")
    ax.plot(xs, ys, "o-", color=LINE, linewidth=2.0, markersize=5, label="Mean final MAE")
    ax.axvline(xs[best], color=BEST, linewidth=1.4, linestyle="--", alpha=0.8)
    sym = r"\kappa_0" if key == "tau" else r"\alpha"
    ax.scatter([xs[best]], [ys[best]], s=110, marker="*", color=BEST, zorder=5,
               label=fr"Best ${sym} = {xs[best]:.2f}$")
    if log_scale:
        ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Final network MAE" + (" (log scale)" if log_scale else ""), fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.legend(loc="best", fontsize=10, framealpha=0.85)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def box_plot(scenario, title, path, ymax=None):
    X = load(scenario)
    data = [final(X, m) for m in M]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showmeans=True,
                    meanprops=dict(marker="D", markerfacecolor="black", markeredgecolor="black", markersize=5),
                    medianprops=dict(color="black", linewidth=1.8))
    for patch, m in zip(bp["boxes"], M):
        patch.set_facecolor(COLORS[m])
        patch.set_alpha(0.55)
    for i, vals in enumerate(data, 1):
        ax.scatter(np.full(len(vals), i) + np.linspace(-0.12, 0.12, len(vals)), vals, s=14, color="black", alpha=0.6, zorder=3)
    ax.set_xticks(range(1, len(M) + 1))
    ax.set_xticklabels([LABELS[m] for m in M], rotation=18, fontsize=10)
    ax.set_ylabel("Round-10 network MAE", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.set_ylim(0, ymax)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


for mode in ("jaccard", "hybrid"):
    d = json.load(open(B / f"ablation_{mode}" / "tau_alpha_ablation_rounds10_article_summary.json"))
    ablation_plot(d["tau_rows"], "tau", r"$\kappa_0$  (similarity threshold)",
                  r"$\kappa_0$ sweep (10 rounds, 5 seeds)", OUT / f"ablation_{mode}" / "tau_ablation_rounds10_article.png")
    ablation_plot(d["alpha_rows"], "alpha", r"$\alpha$  (LVP synchronization step)",
                  r"$\alpha$ sweep (10 rounds, 5 seeds)", OUT / f"ablation_{mode}" / "alpha_ablation_rounds10_article.png")

box_plot("scenario_B", "Sign-inversion attack, 40% Byzantine clients", OUT / "round10_sign_inversion.png", ymax=3.6)
box_plot("clean_shared", "No attack", OUT / "round10_no_attack.png", ymax=3.6)
print("figures written to", OUT)
