#!/usr/bin/env python3
"""
Fig.3-style experiments on uploaded-parameter attacks with enlarged FL budget (vs 8×20×1×10):

  1) noise_scaled — Gaussian noise, 25% malicious (same mal as small noise_attack).
  2) noise_scaled_high_mal — same large sim, 40% malicious, Gaussian noise.
  3) noise_colluded_scaled — same large sim, 25% malicious, noise_colluded (noise + colluded bias).

Same tau / alpha / best model as ablation_summary.json. Outputs under
artifacts/full_comparison/fig3_noise_scale/ by default.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "core"))
sys.path.insert(0, str(_FL / "data_loaders"))

from data_utils import (
    build_client_information_profiles,
    build_clients_from_mcc,
    load_mcc_series,
)
from full_comparison_lvp import (
    AGGREGATORS_COMPARE,
    DEFAULT_ABLATION_SUMMARY,
    load_best_hparams,
    plot_aggregators_network_mae,
)
from run_real_experiments import MODEL_REGISTRY, _build_exogenous, run_one_model

# Baseline noise_attack (fig3_scenario_studies): mal=0.25, scale=5.0, contiguous, 8×20×1×10
BASELINE_NOISE_MAL = 0.25
BASELINE_NOISE_SCALE = 5.0

# Enlarged simulation (defaults; override via CLI)
DEFAULT_LARGE_N_CLIENTS = 16
DEFAULT_LARGE_ROUNDS = 40
DEFAULT_LARGE_LOCAL_EPOCHS = 3
DEFAULT_LARGE_LOCAL_FIT = 15
DEFAULT_HIGH_MAL = 0.40


def _topic_profiles(clients: List, n_groups: int) -> List:
    profiles = build_client_information_profiles(clients, "mcc")
    return [
        frozenset(p) | {f"sync_topic_{idx % n_groups}"}
        for idx, p in enumerate(profiles)
    ]


def run_fig3_aggregators(
    *,
    best_model: str,
    clients: List,
    profiles: List,
    rounds: int,
    local_epochs: int,
    seed: int,
    similarity_tau: float,
    lvp_alpha: float | None,
    local_fit_maxiter: int,
    eval_fit_maxiter: int,
    malicious_frac: float,
    attack_strategy: str,
    attack_scale: float,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for i, agg in enumerate(AGGREGATORS_COMPARE, start=1):
        ModelClass = MODEL_REGISTRY[best_model]
        print(
            f"  [{i}/{len(AGGREGATORS_COMPARE)}] aggregator={agg} "
            f"mal={malicious_frac} strat={attack_strategy} scale={attack_scale}"
        )
        exp = run_one_model(
            best_model,
            ModelClass,
            clients,
            profiles,
            aggregator=agg,
            rounds=rounds,
            local_epochs=local_epochs,
            malicious_frac=malicious_frac,
            seed=seed,
            attack_strategy=attack_strategy,
            attack_scale=attack_scale,
            similarity_tau=similarity_tau,
            lvp_alpha=lvp_alpha if agg == "lvp" else None,
            krum_f=-1,
            local_fit_maxiter=local_fit_maxiter,
            eval_fit_maxiter=eval_fit_maxiter,
        )
        results.append(exp)
    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description="Large-scale Fig.3 experiments (noise / noise_colluded), optional --only filter."
    )
    p.add_argument("--base-path", type=str, default=str(_FL.parent))
    p.add_argument(
        "--ablation-summary",
        type=str,
        default=str(DEFAULT_ABLATION_SUMMARY),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_FL / "artifacts" / "full_comparison" / "fig3_noise_scale"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-reuters", action="store_true")
    p.add_argument("--eval-fit-maxiter", type=int, default=0)
    p.add_argument(
        "--large-n-clients",
        type=int,
        default=DEFAULT_LARGE_N_CLIENTS,
        help="Larger cohort (default 16 vs baseline 8).",
    )
    p.add_argument(
        "--large-rounds",
        type=int,
        default=DEFAULT_LARGE_ROUNDS,
        help="More communication rounds (default 40 vs 20).",
    )
    p.add_argument(
        "--large-local-epochs",
        type=int,
        default=DEFAULT_LARGE_LOCAL_EPOCHS,
        help="More local epochs per round (default 3 vs 1).",
    )
    p.add_argument(
        "--large-local-fit-maxiter",
        type=int,
        default=DEFAULT_LARGE_LOCAL_FIT,
        help="Higher local optimizer budget (default 15 vs 10).",
    )
    p.add_argument(
        "--high-malicious-frac",
        type=float,
        default=DEFAULT_HIGH_MAL,
        help="Malicious fraction for experiment 2 (default 0.4).",
    )
    p.add_argument(
        "--noise-scale",
        type=float,
        default=BASELINE_NOISE_SCALE,
        help="Gaussian noise scale on params (same as noise_attack fig3).",
    )
    p.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma-separated experiment ids to run (e.g. noise_colluded_scaled). Empty = all.",
    )
    args = p.parse_args()

    base = Path(args.base_path).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_model, tau_star, alpha_star = load_best_hparams(Path(args.ablation_summary))
    hparam_note = f" (tau={tau_star:.3g}, alpha={alpha_star!s} from ablation)"

    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=not args.no_reuters)

    n_clients = int(args.large_n_clients)
    rounds = int(args.large_rounds)
    local_epochs = int(args.large_local_epochs)
    local_fit = int(args.large_local_fit_maxiter)
    eval_fit = int(args.eval_fit_maxiter)
    noise_scale = float(args.noise_scale)
    high_mal = float(args.high_malicious_frac)

    topic_groups = max(4, min(8, max(1, n_clients // 2)))

    all_experiments: List[Dict[str, Any]] = [
        {
            "id": "noise_scaled",
            "title": (
                f"Noise attack, enlarged sim: {n_clients} clients, {rounds} rounds, "
                f"{local_epochs} local epochs, local_fit_maxiter={local_fit}; "
                f"mal={BASELINE_NOISE_MAL:.0%} (unchanged vs noise_attack), scale={noise_scale}"
            ),
            "malicious_frac": BASELINE_NOISE_MAL,
            "attack_strategy": "noise",
        },
        {
            "id": "noise_scaled_high_mal",
            "title": (
                f"Noise attack, same enlarged sim + higher mal={high_mal:.0%}, scale={noise_scale}"
            ),
            "malicious_frac": high_mal,
            "attack_strategy": "noise",
        },
        {
            "id": "noise_colluded_scaled",
            "title": (
                f"noise_colluded (Gaussian + colluded bias), enlarged sim: {n_clients} clients, "
                f"{rounds} rounds, {local_epochs} local epochs, local_fit_maxiter={local_fit}; "
                f"mal={BASELINE_NOISE_MAL:.0%}, scale={noise_scale}"
            ),
            "malicious_frac": BASELINE_NOISE_MAL,
            "attack_strategy": "noise_colluded",
        },
    ]

    only_raw = (args.only or "").strip()
    if only_raw:
        want = {x.strip() for x in only_raw.split(",") if x.strip()}
        experiments = [e for e in all_experiments if e["id"] in want]
        missing = want - {e["id"] for e in experiments}
        if missing:
            raise SystemExit(f"Unknown --only id(s): {sorted(missing)}")
        if not experiments:
            raise SystemExit("--only matched no experiments.")
    else:
        experiments = all_experiments

    summary: Dict[str, Any] = {
        "experiment_suite": "fig3_noise_scaled",
        "best_model": best_model,
        "similarity_tau": tau_star,
        "lvp_alpha": alpha_star,
        "shared_sim_params": {
            "n_clients": n_clients,
            "rounds": rounds,
            "local_epochs": local_epochs,
            "local_fit_maxiter": local_fit,
            "eval_fit_maxiter": eval_fit,
            "attack_scale": noise_scale,
            "column_partition": "contiguous",
            "topic_overlay_groups": topic_groups,
        },
        "reference_baseline_noise_attack": {
            "malicious_frac": BASELINE_NOISE_MAL,
            "attack_scale": BASELINE_NOISE_SCALE,
            "typical_small_grid": "8 clients, 20 rounds, 1 local epoch, 10 local_fit_maxiter",
        },
        "runs": [],
    }

    clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=n_clients,
        column_partition="contiguous",
    )
    if len(clients) < 2:
        raise RuntimeError(
            f"Need at least 2 clients; got {len(clients)}. "
            "Lower --large-n-clients or check MCC columns."
        )
    profiles = _topic_profiles(clients, topic_groups)

    for ex in experiments:
        mal = float(ex["malicious_frac"])
        strat = str(ex["attack_strategy"])
        print(f"\n=== {ex['id']} — {ex['title']} ===")
        results = run_fig3_aggregators(
            best_model=best_model,
            clients=clients,
            profiles=profiles,
            rounds=rounds,
            local_epochs=local_epochs,
            seed=args.seed,
            similarity_tau=tau_star,
            lvp_alpha=alpha_star,
            local_fit_maxiter=local_fit,
            eval_fit_maxiter=eval_fit,
            malicious_frac=mal,
            attack_strategy=strat,
            attack_scale=noise_scale,
        )

        png_path = out_dir / f"fig3_aggregators_{ex['id']}.png"
        title_suffix = f"\n{ex['title']}{hparam_note}"
        plot_aggregators_network_mae(results, best_model, png_path, title_suffix)

        merged_cfg = {
            **summary["shared_sim_params"],
            "id": ex["id"],
            "title": ex["title"],
            "malicious_frac": mal,
            "attack_strategy": strat,
        }
        slice_path = out_dir / f"fig3_{ex['id']}.json"
        with open(slice_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "config": merged_cfg,
                    "best_model": best_model,
                    "similarity_tau": tau_star,
                    "lvp_alpha": alpha_star,
                    "results": [
                        {
                            "model": r["model"],
                            "aggregator": r["aggregator"],
                            "rounds": r["rounds"],
                            "history": r["history"],
                            "attack_strategy": r.get("attack_strategy"),
                            "attack_scale": r.get("attack_scale"),
                            "malicious_frac": r.get("malicious_frac"),
                        }
                        for r in results
                    ],
                },
                f,
                indent=2,
                default=str,
            )
        print(f"Saved: {slice_path}")

        finals = {r["aggregator"]: float(r["history"][-1]["network_mae"]) for r in results}
        summary["runs"].append({"id": ex["id"], "network_mae_final_by_aggregator": finals})

    meta_path = out_dir / "fig3_noise_scale_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
