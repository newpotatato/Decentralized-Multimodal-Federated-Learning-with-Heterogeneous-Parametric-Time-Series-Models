#!/usr/bin/env python3
"""
LVP ablation: one-factor-at-a-time (OFAT).

- Sweep τ only (fixed α) → CSV + plot network MAE vs τ.
- Sweep α only (fixed τ) → CSV + plot network MAE vs effective α.

Outputs under federated_learning/artifacts/ablation_lvp/ by default.
Metric: absolute error only — network MAE (same units as column amt).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "core"))
sys.path.insert(0, str(_FL / "data_loaders"))

from data_utils import (
    build_client_information_profiles,
    build_clients_from_mcc,
    load_mcc_series,
)
from run_real_experiments import (  # noqa: E402
    MODEL_REGISTRY,
    _build_exogenous,
    run_one_model,
)


def apply_topic_overlay(
    profiles: List[frozenset], n_groups: int = 4
) -> List[frozenset]:
    return [
        frozenset(p) | {f"sync_topic_{idx % n_groups}"}
        for idx, p in enumerate(profiles)
    ]


def _lvp_alpha_arg(fixed: Optional[float]) -> Optional[float]:
    """None → auto heuristic; positive → fixed LVP step."""
    if fixed is None:
        return None
    if fixed <= 0:
        return None
    return float(fixed)


def _alpha_tag(alpha: Optional[float]) -> str:
    if alpha is None or alpha <= 0:
        return "auto"
    return f"{alpha:.4g}"


def _parse_fixed_alpha(s: str) -> Optional[float]:
    t = s.strip().lower()
    if t in ("auto", "none", ""):
        return None
    return float(t)


def _alpha_grid_full() -> List[Optional[float]]:
    """Dense OFAT grid: auto + numeric alpha from 0.05 to 0.55 step 0.02."""
    return [None] + [round(0.05 + i * 0.02, 4) for i in range(26)]


def _alpha_grid_quick() -> List[Optional[float]]:
    """Shorter grid for smoke tests."""
    return [None, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]


def _load_clients_and_profiles(
    base_path: Path,
    n_clients: int,
    use_reuters: bool,
    enrich_topics: bool,
    topic_groups: int,
) -> Tuple[List[pd.DataFrame], List[frozenset]]:
    mcc_df = load_mcc_series(base_path)
    exog = _build_exogenous(base_path, mcc_df, use_reuters=use_reuters)
    clients = build_clients_from_mcc(mcc_df, exog, n_clients=n_clients)
    if len(clients) < 2:
        raise RuntimeError("Need at least 2 clients for LVP ablation.")
    profiles = build_client_information_profiles(clients, "mcc")
    if enrich_topics:
        profiles = apply_topic_overlay(profiles, n_groups=topic_groups)
        print(
            f"Topic overlay: {topic_groups} groups (Jaccard varies with tau)."
        )
    return clients, profiles


def _run_one_lvp(
    model_name: str,
    ModelClass,
    clients: List[pd.DataFrame],
    profiles: List[frozenset],
    tau: float,
    lvp_alpha: Optional[float],
    rounds: int,
    local_epochs: int,
    malicious_frac: float,
    seed: int,
    local_fit_maxiter: Optional[int],
    eval_fit_maxiter: Optional[int],
) -> Dict:
    return run_one_model(
        model_name,
        ModelClass,
        clients,
        profiles,
        aggregator="lvp",
        rounds=rounds,
        local_epochs=local_epochs,
        malicious_frac=malicious_frac,
        seed=seed,
        attack_strategy="label_flip",
        attack_scale=2.5,
        similarity_tau=tau,
        lvp_alpha=lvp_alpha,
        krum_f=-1,
        local_fit_maxiter=local_fit_maxiter,
        eval_fit_maxiter=eval_fit_maxiter,
    )


def _row_from_exp(
    exp: Dict,
    tau: float,
    alpha_requested: Optional[float],
    alpha_tag: str,
    fixed_tau: Optional[float],
    fixed_alpha: Optional[float],
    sweep: str,
    model_name: str,
    malicious_frac: float,
) -> Dict:
    last = exp["history"][-1]
    first = exp["history"][0]
    mean_client_mae = float(np.mean(list(last["client_mae"].values())))
    network_mae = float(last["network_mae"])
    network_mae_r1 = float(first["network_mae"])
    alpha_eff = last.get("lvp_alpha")
    return {
        "sweep": sweep,
        "tau": tau,
        "fixed_tau": fixed_tau,
        "alpha_requested": alpha_requested,
        "alpha_config": alpha_tag,
        "fixed_alpha_for_tau_sweep": fixed_alpha,
        "lvp_alpha_effective": alpha_eff,
        "mean_client_mae_final": mean_client_mae,
        "network_mae_final": network_mae,
        "network_mae_round1": network_mae_r1,
        "network_mae_improvement": network_mae_r1 - network_mae,
        "model": model_name,
        "rounds": exp["rounds"],
        "malicious_frac": malicious_frac,
    }


def run_tau_sweep(
    base_path: Path,
    model_name: str,
    n_clients: int,
    rounds: int,
    local_epochs: int,
    malicious_frac: float,
    seed: int,
    taus: List[float],
    fixed_alpha: Optional[float],
    use_reuters: bool,
    out_dir: Path,
    enrich_topics: bool,
    topic_groups: int,
    local_fit_maxiter: Optional[int],
    eval_fit_maxiter: Optional[int],
) -> pd.DataFrame:
    print("Loading data (tau sweep)...")
    clients, profiles = _load_clients_and_profiles(
        base_path, n_clients, use_reuters, enrich_topics, topic_groups
    )
    ModelClass = MODEL_REGISTRY[model_name]
    lvp_a = _lvp_alpha_arg(fixed_alpha)
    tag = _alpha_tag(lvp_a)

    rows: List[Dict] = []
    for i, tau in enumerate(taus):
        print(f"[tau sweep {i + 1}/{len(taus)}] tau={tau:.4f}  (alpha fixed: {tag})")
        exp = _run_one_lvp(
            model_name,
            ModelClass,
            clients,
            profiles,
            tau,
            lvp_a,
            rounds,
            local_epochs,
            malicious_frac,
            seed,
            local_fit_maxiter,
            eval_fit_maxiter,
        )
        rows.append(
            _row_from_exp(
                exp,
                tau=tau,
                alpha_requested=lvp_a,
                alpha_tag=tag,
                fixed_tau=None,
                fixed_alpha=fixed_alpha,
                sweep="tau",
                model_name=model_name,
                malicious_frac=malicious_frac,
            )
        )

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ablation_ofat_tau_sweep.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df


def run_alpha_sweep(
    base_path: Path,
    model_name: str,
    n_clients: int,
    rounds: int,
    local_epochs: int,
    malicious_frac: float,
    seed: int,
    fixed_tau: float,
    alphas: List[Optional[float]],
    use_reuters: bool,
    out_dir: Path,
    enrich_topics: bool,
    topic_groups: int,
    local_fit_maxiter: Optional[int],
    eval_fit_maxiter: Optional[int],
) -> pd.DataFrame:
    print("Loading data (alpha sweep)...")
    clients, profiles = _load_clients_and_profiles(
        base_path, n_clients, use_reuters, enrich_topics, topic_groups
    )
    ModelClass = MODEL_REGISTRY[model_name]

    rows: List[Dict] = []
    for i, a in enumerate(alphas):
        lvp_a = _lvp_alpha_arg(a)
        tag = _alpha_tag(lvp_a)
        print(
            f"[alpha sweep {i + 1}/{len(alphas)}] alpha={tag}  (tau fixed: {fixed_tau:.4f})"
        )
        exp = _run_one_lvp(
            model_name,
            ModelClass,
            clients,
            profiles,
            fixed_tau,
            lvp_a,
            rounds,
            local_epochs,
            malicious_frac,
            seed,
            local_fit_maxiter,
            eval_fit_maxiter,
        )
        rows.append(
            _row_from_exp(
                exp,
                tau=fixed_tau,
                alpha_requested=lvp_a,
                alpha_tag=tag,
                fixed_tau=fixed_tau,
                fixed_alpha=None,
                sweep="alpha",
                model_name=model_name,
                malicious_frac=malicious_frac,
            )
        )

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "ablation_ofat_alpha_sweep.csv"
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df


def plot_tau_sweep(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    sub = df.sort_values("tau")
    fa_label = str(sub["alpha_config"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        sub["tau"],
        sub["network_mae_final"],
        "o-",
        color="C0",
        linewidth=2,
        markersize=7,
        label="Network MAE (final)",
    )
    ax.set_xlabel(r"Jaccard threshold $\tau$ (only varying parameter)")
    ax.set_ylabel("Network MAE (absolute, amt units)")
    ax.set_title(rf"OFAT: sweep $\tau$ — $\alpha$ fixed ($\alpha$ = {fa_label})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    p = out_dir / "ablation_ofat_tau.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def plot_alpha_sweep(df: pd.DataFrame, out_dir: Path) -> None:
    import matplotlib.pyplot as plt

    sub = df.copy()
    sub["_x"] = sub["lvp_alpha_effective"].astype(float)
    sub = sub.sort_values("_x")

    tau0 = float(sub["fixed_tau"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(
        sub["_x"],
        sub["network_mae_final"],
        "s-",
        color="C1",
        linewidth=2,
        markersize=7,
        label="Network MAE (final)",
    )
    ax.set_xlabel(r"LVP step $\alpha$ (effective, only varying parameter)")
    ax.set_ylabel("Network MAE (absolute, amt units)")
    ax.set_title(rf"OFAT: sweep $\alpha$ — $\tau$ fixed ($\tau$ = {tau0:.4g})")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    p = out_dir / "ablation_ofat_alpha.png"
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="LVP OFAT ablation: tau and alpha sweeps, network MAE only (absolute)."
    )
    p.add_argument("--base-path", type=str, default=str(_FL.parent))
    p.add_argument("--model", type=str, default="DynamicLinearModel")
    p.add_argument("--n-clients", type=int, default=8)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--malicious-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-reuters", action="store_true")
    p.add_argument(
        "--sweep",
        type=str,
        choices=("both", "tau", "alpha"),
        default="both",
        help="Run tau sweep, alpha sweep, or both (default: both).",
    )
    p.add_argument(
        "--fixed-alpha",
        type=str,
        default="auto",
        help="For tau sweep: fixed alpha (float) or 'auto' for heuristic alpha.",
    )
    p.add_argument(
        "--fixed-tau",
        type=float,
        default=0.35,
        help="For alpha sweep: fixed Jaccard threshold tau.",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Shorter tau grid and shorter alpha grid for smoke tests "
            "(full run: alpha = auto + 0.05..0.55 step 0.02, 27 points)."
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=str(_FL / "artifacts" / "ablation_lvp"),
    )
    p.add_argument(
        "--no-topic-overlay",
        action="store_true",
        help="Disable synthetic sync_topic_* tags.",
    )
    p.add_argument(
        "--topic-groups",
        type=int,
        default=4,
        help="Topic buckets for overlay.",
    )
    p.add_argument(
        "--local-fit-maxiter",
        type=int,
        default=10,
        help="Limited local optimization per round (FL variant 1).",
    )
    p.add_argument(
        "--eval-fit-maxiter",
        type=int,
        default=0,
        help="Refit budget for network_mae after sync.",
    )
    args = p.parse_args()
    base = Path(args.base_path).resolve()
    out_dir = Path(args.out_dir)
    fixed_alpha = _parse_fixed_alpha(args.fixed_alpha)

    if args.quick:
        taus = [
            0.0,
            0.2,
            0.28,
            0.32,
            0.36,
            0.4,
            0.44,
            0.48,
            0.52,
            0.56,
            0.6,
            0.65,
            0.72,
        ]
        alphas = _alpha_grid_quick()
    else:
        taus = [
            0.0,
            0.05,
            0.1,
            0.12,
            0.15,
            0.18,
            0.2,
            0.25,
            0.3,
            0.35,
            0.4,
            0.45,
            0.5,
            0.55,
            0.6,
            0.65,
            0.72,
        ]
        alphas = _alpha_grid_full()

    common = dict(
        base_path=base,
        model_name=args.model,
        n_clients=args.n_clients,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        malicious_frac=args.malicious_frac,
        seed=args.seed,
        use_reuters=not args.no_reuters,
        out_dir=out_dir,
        enrich_topics=not args.no_topic_overlay,
        topic_groups=args.topic_groups,
        local_fit_maxiter=args.local_fit_maxiter,
        eval_fit_maxiter=args.eval_fit_maxiter,
    )

    df_tau = None
    df_alpha = None

    if args.sweep in ("both", "tau"):
        df_tau = run_tau_sweep(taus=taus, fixed_alpha=fixed_alpha, **common)
        plot_tau_sweep(df_tau, out_dir)

    if args.sweep in ("both", "alpha"):
        df_alpha = run_alpha_sweep(
            fixed_tau=args.fixed_tau, alphas=alphas, **common
        )
        plot_alpha_sweep(df_alpha, out_dir)

    meta: Dict = {
        "sweep_mode": args.sweep,
        "metric": "network_mae_absolute_amt_units",
        "fixed_alpha_for_tau_sweep": args.fixed_alpha,
        "fixed_tau_for_alpha_sweep": args.fixed_tau,
        "local_fit_maxiter": args.local_fit_maxiter,
        "eval_fit_maxiter": args.eval_fit_maxiter,
        "fl_variant": "ofat_absolute_mae_only",
    }
    if df_tau is not None:
        meta["taus"] = taus
        meta["best_network_mae_tau_sweep"] = df_tau.loc[
            df_tau["network_mae_final"].idxmin()
        ].to_dict()
    if df_alpha is not None:
        meta["alphas"] = [("auto" if a is None else a) for a in alphas]
        meta["n_alpha_runs"] = len(alphas)
        meta["best_network_mae_alpha_sweep"] = df_alpha.loc[
            df_alpha["network_mae_final"].idxmin()
        ].to_dict()

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print("Done.")


if __name__ == "__main__":
    main()
