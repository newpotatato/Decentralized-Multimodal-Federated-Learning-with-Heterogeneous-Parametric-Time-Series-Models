#!/usr/bin/env python3
"""Print the numbers reported in the article from artifacts_pipeline_v2."""
import json
from pathlib import Path

import numpy as np

B = Path(__file__).resolve().parents[1] / "artifacts_pipeline_v2"
SEEDS = (42, 52, 62, 72, 82)
M = ["lvp", "decentralized_fedavg", "defta", "balance", "push_sum", "local"]


def load(sc):
    out = {m: {} for m in M}
    for s in SEEDS:
        d = json.load(open(B / sc / "raw" / "seeds" / f"seed{s}" / "fig3_decentralized_methods.json", encoding="utf-8"))
        for r in d["results"]:
            out[r["aggregator"]][s] = r["history"]
    return out


def final(X, m):
    return np.array([X[m][s][-1]["network_mae"] for s in SEEDS])


if __name__ == "__main__":
    for name in ("ablation_jaccard", "ablation_hybrid"):
        d = json.load(open(B / name / "tau_alpha_ablation_rounds10_article_summary.json"))
        print(name, "alpha swept at kappa0 =", d["config"]["fixed_tau_for_alpha"])
        print("  kappa0:", [(r["tau"], float(f"{r['mean']:.4g}"), float(f"{r['std']:.3g}")) for r in d["tau_rows"]])
        print("  alpha :", [(r["alpha"], float(f"{r['mean']:.4g}"), float(f"{r['std']:.3g}")) for r in d["alpha_rows"]])

    C = load("scenario_C")
    print("\nTable 2 (scenario C, 10 rounds x 5 seeds = 50 values, percentile method = higher)")
    for m in M:
        v = np.array([h["network_mae"] for s in SEEDS for h in C[m][s]])
        p = lambda q: np.percentile(v, q, method="higher")
        fin = final(C, m)
        print(f"{m:22s} min={v.min():.3g} med={p(50):.3g} p25={p(25):.3g} p95={p(95):.3g} | values>10: {np.mean(v > 10):.2f}"
              f" | seeds with round-10 MAE>10: {(fin > 10).sum()}/5 | round-10 median={np.median(fin):.3g}")

    print("\nRound-10 MAE, mean +- sd over seeds")
    for sc in ("clean_shared", "scenario_B", "scenario_A", "clean_native", "scenario_B_noexog"):
        X = load(sc)
        print(sc, {m: f"{final(X, m).mean():.2f}+-{final(X, m).std():.2f}" for m in M})
        print("   medians:", {m: round(float(np.median(final(X, m))), 2) for m in M})
        print("   LVP rank per seed:", [sorted(M, key=lambda m: final(X, m)[i]).index("lvp") + 1 for i in range(5)])

    X = load("scenario_B")
    print("\nB: lvp-local per seed", np.round(final(X, "lvp") - final(X, "local"), 3),
          "| lvp-balance", np.round(final(X, "lvp") - final(X, "balance"), 3),
          "| fedavg/lvp", np.round(final(X, "decentralized_fedavg") / final(X, "lvp"), 2))
    X = load("clean_shared")
    print("clean: lvp-local", np.round(final(X, "lvp") - final(X, "local"), 3),
          "| fedavg-lvp", np.round(final(X, "decentralized_fedavg") - final(X, "lvp"), 3),
          "| balance-lvp", np.round(final(X, "balance") - final(X, "lvp"), 3))
    for sc in ("clean_shared", "scenario_B"):
        X = load(sc)
        print(sc, "pairwise parameter disagreement, round 10:",
              {m: round(float(np.mean([X[m][s][-1]["sync_component_pairwise_l2_mean"] for s in SEEDS])), 3) for m in M})
