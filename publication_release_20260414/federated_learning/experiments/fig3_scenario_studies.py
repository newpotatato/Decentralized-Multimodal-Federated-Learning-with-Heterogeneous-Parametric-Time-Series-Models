#!/usr/bin/env python3
"""
Fig.3-style aggregator comparison under several honest, pre-registered scenarios:

  1) Stronger attack on the mean (higher malicious fraction and/or label_flip scale).
  2) Stronger non-IID (strided MCC column partition instead of contiguous blocks).
  3) Combined stress test.
  4) Alternative attack (Gaussian noise on params) — different bias than inversion.

Uses the same tau / alpha / best model as ablation_summary.json (no per-scenario retuning).
Outputs separate PNG + JSON slice per scenario under artifacts/full_comparison/fig3_scenarios/.
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


def _topic_profiles(clients: List, n_groups: int = 4) -> List:
    profiles = build_client_information_profiles(clients, "mcc")
    return [
        frozenset(p) | {f"sync_topic_{idx % n_groups}"}
        for idx, p in enumerate(profiles)
    ]


def run_fig3_for_scenario(
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
        exp["scenario_malicious_frac"] = malicious_frac
        exp["scenario_attack_strategy"] = attack_strategy
        exp["scenario_attack_scale"] = attack_scale
        results.append(exp)
    return results


def main() -> None:
    p = argparse.ArgumentParser(
        description="Fig.3 aggregator panels under attack / non-IID scenarios (real MCC)."
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
        default=str(_FL / "artifacts" / "full_comparison" / "fig3_scenarios"),
    )
    p.add_argument("--n-clients", type=int, default=8)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-reuters", action="store_true")
    p.add_argument("--local-fit-maxiter", type=int, default=10)
    p.add_argument("--eval-fit-maxiter", type=int, default=0)
    args = p.parse_args()

    base = Path(args.base_path).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_model, tau_star, alpha_star = load_best_hparams(Path(args.ablation_summary))
    hparam_note = f" (same tau={tau_star:.3g}, alpha={alpha_star!s} as ablation)"

    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=not args.no_reuters)

    scenarios: List[Dict[str, Any]] = [
        {
            "id": "attack_on_mean",
            "title": "Attack on mean: 40% malicious, label_flip scale 3.5",
            "column_partition": "contiguous",
            "malicious_frac": 0.4,
            "attack_strategy": "label_flip",
            "attack_scale": 3.5,
        },
        {
            "id": "noniid_strided",
            "title": "Stronger non-IID: strided MCC partition (20% malicious, baseline attack)",
            "column_partition": "strided",
            "malicious_frac": 0.2,
            "attack_strategy": "label_flip",
            "attack_scale": 2.5,
        },
        {
            "id": "attack_and_noniid",
            "title": "Combined: 35% malicious, scale 3.0 + strided partition",
            "column_partition": "strided",
            "malicious_frac": 0.35,
            "attack_strategy": "label_flip",
            "attack_scale": 3.0,
        },
        {
            "id": "noise_attack",
            "title": "Alternative attack: Gaussian noise on params (25% malicious, scale 5.0)",
            "column_partition": "contiguous",
            "malicious_frac": 0.25,
            "attack_strategy": "noise",
            "attack_scale": 5.0,
        },
        {
            "id": "noise_colluded_attack",
            "title": (
                "Noise + colluded directional bias (same shift per key for all Byzantine); "
                "25% malicious, scale 5.0 — FedAvg cannot average bias away"
            ),
            "column_partition": "contiguous",
            "malicious_frac": 0.25,
            "attack_strategy": "noise_colluded",
            "attack_scale": 5.0,
        },
    ]

    summary: Dict[str, Any] = {
        "best_model": best_model,
        "similarity_tau": tau_star,
        "lvp_alpha": alpha_star,
        "hparam_note": hparam_note.strip(),
        "rounds": args.rounds,
        "scenarios": [],
    }

    for sc in scenarios:
        print(f"\n=== Scenario: {sc['id']} — {sc['title']} ===")
        clients = build_clients_from_mcc(
            mcc_df,
            exog,
            n_clients=args.n_clients,
            column_partition=sc["column_partition"],
        )
        if len(clients) < 2:
            raise RuntimeError(f"Scenario {sc['id']}: need at least 2 clients.")
        profiles = _topic_profiles(clients)

        results = run_fig3_for_scenario(
            best_model=best_model,
            clients=clients,
            profiles=profiles,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            seed=args.seed,
            similarity_tau=tau_star,
            lvp_alpha=alpha_star,
            local_fit_maxiter=args.local_fit_maxiter,
            eval_fit_maxiter=args.eval_fit_maxiter,
            malicious_frac=float(sc["malicious_frac"]),
            attack_strategy=str(sc["attack_strategy"]),
            attack_scale=float(sc["attack_scale"]),
        )

        slug = sc["id"]
        png_path = out_dir / f"fig3_aggregators_{slug}.png"
        title_suffix = f"\n{sc['title']}{hparam_note}"
        plot_aggregators_network_mae(results, best_model, png_path, title_suffix)

        slice_path = out_dir / f"fig3_scenario_{slug}.json"
        with open(slice_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "scenario": sc,
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

        finals = []
        for r in results:
            last = r["history"][-1]["network_mae"]
            finals.append((r["aggregator"], float(last)))
        summary["scenarios"].append(
            {
                "id": slug,
                "config": sc,
                "network_mae_final_by_aggregator": dict(finals),
            }
        )

    meta_path = out_dir / "fig3_scenarios_summary.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved summary: {meta_path}")
    print("Done.")


if __name__ == "__main__":
    main()
