#!/usr/bin/env python3
"""Fair decentralized comparison pipeline with LVP-only smart topology.

Stage A (tuning):
- Tune only LVP on train scenario(s) using hybrid similarity and anti-spike score.
- Select tau from round-1 topology balancing (optional auto search).

Stage B (evaluation):
- Evaluate tuned LVP with hybrid topology.
- Evaluate decentralized analogs with static Jaccard topology only.
- Aggregate multi-seed metrics and produce plot/json/report.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent
sys.path.insert(0, str(_FL / "core"))
sys.path.insert(0, str(_FL / "data_loaders"))

from data_utils import (  # noqa: E402
    build_client_information_profiles,
    build_clients_from_mcc,
    load_mcc_series,
)
from run_real_experiments import (  # noqa: E402
    MODEL_REGISTRY,
    _build_exogenous,
    run_one_model,
    train_local_model,
)


BASELINES = ["decentralized_fedavg", "defta", "balance", "push_sum"]


@dataclass
class Scenario:
    partition: str
    attack_strategy: str
    attack_scale: float
    malicious_frac: float


def _parse_csv_floats(text: str) -> List[float]:
    vals = []
    for x in (text or "").split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(float(x))
    return vals


def _parse_csv_ints(text: str) -> List[int]:
    vals = []
    for x in (text or "").split(","):
        x = x.strip()
        if not x:
            continue
        vals.append(int(x))
    return vals


def _parse_scenarios(text: str) -> List[Scenario]:
    """Format: partition:attack:scale:mal;partition:attack:scale:mal"""
    out: List[Scenario] = []
    for chunk in (text or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        p, a, sc, m = [z.strip() for z in chunk.split(":")]
        out.append(Scenario(partition=p, attack_strategy=a, attack_scale=float(sc), malicious_frac=float(m)))
    return out


def _topic_profiles(clients: List, n_groups: int = 4) -> List[frozenset]:
    profiles = build_client_information_profiles(clients, "mcc")
    return [frozenset(p) | {f"sync_topic_{idx % n_groups}"} for idx, p in enumerate(profiles)]


def _history_series(exp: Dict[str, Any]) -> np.ndarray:
    vals = [float(r.get("network_mae", np.nan)) for r in (exp.get("history") or [])]
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return np.asarray([np.inf], dtype=float)
    return np.nan_to_num(arr, nan=np.inf, posinf=np.inf, neginf=np.inf)


def _anti_spike_score(exp: Dict[str, Any], beta: float, gamma: float) -> float:
    s = _history_series(exp)
    final = float(s[-1])
    if s.size < 2:
        return final
    jumps = np.abs(np.diff(s))
    return float(final + beta * float(np.max(jumps)) + gamma * float(np.std(s)))


def _flatten_params(params: Dict[str, np.ndarray]) -> np.ndarray:
    parts: List[np.ndarray] = []
    for key in sorted(params.keys()):
        try:
            arr = np.asarray(params[key], dtype=float).ravel()
        except (TypeError, ValueError):
            continue
        if arr.size:
            parts.append(arr)
    if not parts:
        return np.array([], dtype=float)
    return np.concatenate(parts)


def _cosine_matrix(params_list: Sequence[Dict[str, np.ndarray]]) -> np.ndarray:
    n = len(params_list)
    vecs = [_flatten_params(p) for p in params_list]
    cos = np.ones((n, n), dtype=float)
    np.fill_diagonal(cos, 0.0)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a = vecs[i]
            b = vecs[j]
            if a.size == 0 or b.size == 0:
                cos[i, j] = 1.0
                continue
            m = min(a.size, b.size)
            if m == 0:
                cos[i, j] = 1.0
                continue
            a = a[:m]
            b = b[:m]
            na = float(np.linalg.norm(a))
            nb = float(np.linalg.norm(b))
            if na <= 1e-12 or nb <= 1e-12:
                cos[i, j] = 1.0
            else:
                cos[i, j] = float(np.dot(a, b) / (na * nb))
    return np.clip(cos, -1.0, 1.0)


def _jaccard_matrix(profiles: List[frozenset]) -> np.ndarray:
    n = len(profiles)
    out = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            inter = len(profiles[i] & profiles[j])
            uni = len(profiles[i] | profiles[j])
            out[i, j] = float(inter) / float(uni) if uni else 0.0
    return out


def _neighbors_from_similarity(sim: np.ndarray, cos: np.ndarray, tau: float, tau_cos_min: float) -> List[List[int]]:
    n = sim.shape[0]
    neighbors: List[List[int]] = []
    for i in range(n):
        row: List[int] = []
        for j in range(n):
            if i == j:
                continue
            if sim[i, j] < tau:
                continue
            if cos[i, j] < tau_cos_min:
                continue
            row.append(j)
        neighbors.append(row)
    return neighbors


def _components_undirected(neighbors: List[List[int]]) -> List[List[int]]:
    n = len(neighbors)
    und = [set(row) for row in neighbors]
    for i in range(n):
        for j in neighbors[i]:
            und[j].add(i)

    seen = [False] * n
    comps: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        st = [i]
        seen[i] = True
        c: List[int] = []
        while st:
            v = st.pop()
            c.append(v)
            for u in und[v]:
                if not seen[u]:
                    seen[u] = True
                    st.append(u)
        comps.append(sorted(c))
    return sorted(comps, key=len, reverse=True)


def _topology_balance_score(components: List[List[int]], target_components: int) -> float:
    sizes = np.asarray([len(c) for c in components], dtype=float)
    if sizes.size == 0:
        return 1e9
    return float(np.std(sizes) + 3.0 * abs(len(components) - target_components) + 2.0 * np.sum(sizes == 1.0))


def _select_tau_round1(
    clients: List,
    profiles: List[frozenset],
    model_name: str,
    lambda_jaccard: float,
    tau_cos_min: float,
    tau_grid: np.ndarray,
    local_epochs: int,
    local_fit_maxiter: int,
    target_components: int,
) -> Tuple[float, Dict[str, Any]]:
    ModelClass = MODEL_REGISTRY[model_name]
    params_round1: List[Dict[str, np.ndarray]] = []
    for df in clients:
        p, _, _, _ = train_local_model(
            ModelClass,
            df,
            local_epochs=local_epochs,
            local_params={},
            local_fit_maxiter=local_fit_maxiter,
            strict_errors=True,
        )
        params_round1.append(p)

    jacc = _jaccard_matrix(profiles)
    cos = _cosine_matrix(params_round1)
    cos01 = 0.5 * (cos + 1.0)
    lam = float(np.clip(lambda_jaccard, 0.0, 1.0))
    sim = lam * jacc + (1.0 - lam) * cos01

    best: Optional[Tuple[float, float, List[List[int]], List[List[int]]]] = None
    for tau in tau_grid:
        neighbors = _neighbors_from_similarity(sim, cos, float(tau), float(tau_cos_min))
        comps = _components_undirected(neighbors)
        sc = _topology_balance_score(comps, target_components)
        if best is None or sc < best[0]:
            best = (sc, float(tau), neighbors, comps)

    assert best is not None
    _, tau_star, neighbors_star, comps_star = best
    topo_meta = {
        "tau_selected": tau_star,
        "n_components": len(comps_star),
        "component_sizes": [len(c) for c in comps_star],
        "components": comps_star,
        "degrees": [len(x) for x in neighbors_star],
    }
    return tau_star, topo_meta


def _aggregate_histories(exps: List[Dict[str, Any]]) -> Dict[str, Any]:
    min_len = min(len(e.get("history") or []) for e in exps)
    rounds = list(range(1, min_len + 1))
    matrix = np.asarray(
        [[float(e["history"][r - 1]["network_mae"]) for r in rounds] for e in exps],
        dtype=float,
    )
    return {
        "round": rounds,
        "mean_network_mae": np.mean(matrix, axis=0).tolist(),
        "std_network_mae": np.std(matrix, axis=0).tolist(),
    }


def _plot_eval(curves: Dict[str, Dict[str, Any]], out_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    order = ["lvp", "decentralized_fedavg", "defta", "balance", "push_sum"]
    labels = {
        "lvp": "LVP (hybrid topology)",
        "decentralized_fedavg": "Decentralized FedAvg",
        "defta": "DeFTA",
        "balance": "BALANCE",
        "push_sum": "Push-Sum",
    }

    fig, ax = plt.subplots(figsize=(10, 5.6))
    for agg in order:
        if agg not in curves:
            continue
        c = curves[agg]
        xs = np.asarray(c["round"], dtype=int)
        m = np.asarray(c["mean_network_mae"], dtype=float)
        s = np.asarray(c["std_network_mae"], dtype=float)
        lw = 2.8 if agg == "lvp" else 1.8
        ax.plot(xs, m, "o-", linewidth=lw, markersize=4.5, label=labels.get(agg, agg))
        ax.fill_between(xs, m - s, m + s, alpha=0.12)

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Network MAE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="LVP fair pipeline with LVP-only smart topology")
    p.add_argument("--base-path", type=str, default=str(_FL.parent))
    p.add_argument("--out-dir", type=str, default=str(_FL / "artifacts" / "lvp_fair_pipeline"))
    p.add_argument("--model", type=str, default="DynamicLinearModel")
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--seed-list", type=str, default="42,52")
    p.add_argument("--eval-seed-list", type=str, default="42,52,62")

    p.add_argument(
        "--stagea-scenarios",
        type=str,
        default="strided:noise_colluded:2.5:0.25;contiguous:noise_colluded:2.5:0.25",
        help="partition:attack:scale:mal;...",
    )
    p.add_argument("--stagea-rounds", type=int, default=12)
    p.add_argument("--stagea-local-epochs", type=int, default=2)
    p.add_argument("--stagea-local-fit-maxiter", type=int, default=10)
    p.add_argument("--stagea-beta", type=float, default=0.3)
    p.add_argument("--stagea-gamma", type=float, default=0.2)
    p.add_argument("--stagea-max-configs", type=int, default=12)

    p.add_argument("--grid-lambda-jaccard", type=str, default="0.55,0.65")
    p.add_argument("--grid-tau-cos-min", type=str, default="0.10,0.15")
    p.add_argument("--grid-lvp-alpha", type=str, default="0.20,0.25")
    p.add_argument("--grid-lvp-self-weight", type=str, default="0.25,0.35")
    p.add_argument("--tau-grid-min", type=float, default=0.55)
    p.add_argument("--tau-grid-max", type=float, default=0.90)
    p.add_argument("--tau-grid-steps", type=int, default=36)
    p.add_argument("--target-components", type=int, default=4)
    p.add_argument("--sync-topic-groups", type=int, default=4)

    p.add_argument("--eval-partition", type=str, default="strided", choices=["contiguous", "strided"])
    p.add_argument("--eval-attack-strategy", type=str, default="noise_colluded")
    p.add_argument("--eval-attack-scale", type=float, default=2.5)
    p.add_argument("--eval-malicious-frac", type=float, default=0.25)
    p.add_argument("--eval-rounds", type=int, default=30)
    p.add_argument("--eval-local-epochs", type=int, default=3)
    p.add_argument("--eval-local-fit-maxiter", type=int, default=10)
    p.add_argument("--eval-fit-maxiter", type=int, default=0)
    p.add_argument("--network-eval-mode", type=str, default="proxy", choices=["proxy", "refit"])
    p.add_argument("--baseline-tau", type=float, default=0.60)

    args = p.parse_args()

    if args.model not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {args.model}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base = Path(args.base_path).resolve()

    stagea_seeds = _parse_csv_ints(args.seed_list)
    eval_seeds = _parse_csv_ints(args.eval_seed_list)
    stagea_scenarios = _parse_scenarios(args.stagea_scenarios)

    lambdas = _parse_csv_floats(args.grid_lambda_jaccard)
    tau_cos_vals = _parse_csv_floats(args.grid_tau_cos_min)
    alphas = _parse_csv_floats(args.grid_lvp_alpha)
    self_weights = _parse_csv_floats(args.grid_lvp_self_weight)

    mcc_df = load_mcc_series(base)
    exog = _build_exogenous(base, mcc_df, use_reuters=True)

    print("=== Stage A: tune LVP only ===")
    candidate_configs = list(itertools.product(lambdas, tau_cos_vals, alphas, self_weights))
    if args.stagea_max_configs > 0:
        candidate_configs = candidate_configs[: args.stagea_max_configs]

    tau_grid = np.linspace(args.tau_grid_min, args.tau_grid_max, args.tau_grid_steps)
    trials: List[Dict[str, Any]] = []

    for cfg_idx, (lam, tau_cos, alpha, self_w) in enumerate(candidate_configs, start=1):
        print(f"[A {cfg_idx}/{len(candidate_configs)}] lambda={lam} tau_cos={tau_cos} alpha={alpha} self_w={self_w}")
        per_run_scores: List[float] = []
        per_run_final: List[float] = []
        tau_meta_collect: List[Dict[str, Any]] = []

        for sc in stagea_scenarios:
            clients = build_clients_from_mcc(
                mcc_df,
                exog,
                n_clients=args.n_clients,
                column_partition=sc.partition,
            )
            profiles = _topic_profiles(clients, n_groups=args.sync_topic_groups)
            tau_star, tau_meta = _select_tau_round1(
                clients=clients,
                profiles=profiles,
                model_name=args.model,
                lambda_jaccard=lam,
                tau_cos_min=tau_cos,
                tau_grid=tau_grid,
                local_epochs=1,
                local_fit_maxiter=args.stagea_local_fit_maxiter,
                target_components=args.target_components,
            )
            tau_meta_collect.append({"scenario": sc.__dict__, **tau_meta})

            for seed in stagea_seeds:
                exp = run_one_model(
                    args.model,
                    MODEL_REGISTRY[args.model],
                    clients,
                    profiles,
                    aggregator="lvp",
                    rounds=args.stagea_rounds,
                    local_epochs=args.stagea_local_epochs,
                    malicious_frac=sc.malicious_frac,
                    seed=seed,
                    attack_strategy=sc.attack_strategy,
                    attack_scale=sc.attack_scale,
                    similarity_tau=tau_star,
                    similarity_mode="jaccard_cosine_hybrid",
                    lambda_jaccard=lam,
                    tau_cos_min=tau_cos,
                    lvp_alpha=alpha,
                    lvp_self_weight=self_w,
                    krum_f=-1,
                    local_fit_maxiter=args.stagea_local_fit_maxiter,
                    eval_fit_maxiter=args.eval_fit_maxiter,
                    strict_errors=True,
                    network_eval_mode=args.network_eval_mode,
                )
                per_run_scores.append(_anti_spike_score(exp, beta=args.stagea_beta, gamma=args.stagea_gamma))
                per_run_final.append(float(_history_series(exp)[-1]))

        trials.append(
            {
                "config": {
                    "lambda_jaccard": lam,
                    "tau_cos_min": tau_cos,
                    "lvp_alpha": alpha,
                    "lvp_self_weight": self_w,
                },
                "score_mean": float(np.mean(per_run_scores)) if per_run_scores else float("inf"),
                "score_std": float(np.std(per_run_scores)) if per_run_scores else float("inf"),
                "final_mae_mean": float(np.mean(per_run_final)) if per_run_final else float("inf"),
                "tau_meta": tau_meta_collect,
            }
        )

    trials.sort(key=lambda x: x["score_mean"])
    best = trials[0]
    best_cfg = best["config"]
    print("Best LVP config:", best_cfg)

    # Use median of selected taus over Stage A scenarios to avoid scenario overfit.
    tau_candidates = [float(m["tau_selected"]) for m in best.get("tau_meta", []) if "tau_selected" in m]
    lvp_tau_eval = float(np.median(tau_candidates)) if tau_candidates else float(args.baseline_tau)

    print("=== Stage B: final decentralized comparison ===")
    eval_clients = build_clients_from_mcc(
        mcc_df,
        exog,
        n_clients=args.n_clients,
        column_partition=args.eval_partition,
    )
    eval_profiles = _topic_profiles(eval_clients, n_groups=args.sync_topic_groups)

    by_agg_exps: Dict[str, List[Dict[str, Any]]] = {"lvp": []}
    for b in BASELINES:
        by_agg_exps[b] = []

    for seed in eval_seeds:
        # LVP with smart hybrid topology
        lvp_exp = run_one_model(
            args.model,
            MODEL_REGISTRY[args.model],
            eval_clients,
            eval_profiles,
            aggregator="lvp",
            rounds=args.eval_rounds,
            local_epochs=args.eval_local_epochs,
            malicious_frac=args.eval_malicious_frac,
            seed=seed,
            attack_strategy=args.eval_attack_strategy,
            attack_scale=args.eval_attack_scale,
            similarity_tau=lvp_tau_eval,
            similarity_mode="jaccard_cosine_hybrid",
            lambda_jaccard=float(best_cfg["lambda_jaccard"]),
            tau_cos_min=float(best_cfg["tau_cos_min"]),
            lvp_alpha=float(best_cfg["lvp_alpha"]),
            lvp_self_weight=float(best_cfg["lvp_self_weight"]),
            krum_f=-1,
            local_fit_maxiter=args.eval_local_fit_maxiter,
            eval_fit_maxiter=args.eval_fit_maxiter,
            strict_errors=True,
            network_eval_mode=args.network_eval_mode,
        )
        by_agg_exps["lvp"].append(lvp_exp)

        # Baselines with static Jaccard topology only.
        for agg in BASELINES:
            exp = run_one_model(
                args.model,
                MODEL_REGISTRY[args.model],
                eval_clients,
                eval_profiles,
                aggregator=agg,
                rounds=args.eval_rounds,
                local_epochs=args.eval_local_epochs,
                malicious_frac=args.eval_malicious_frac,
                seed=seed,
                attack_strategy=args.eval_attack_strategy,
                attack_scale=args.eval_attack_scale,
                similarity_tau=float(args.baseline_tau),
                similarity_mode="jaccard",
                lambda_jaccard=0.5,
                tau_cos_min=-1.0,
                lvp_alpha=None,
                lvp_self_weight=0.0,
                krum_f=-1,
                local_fit_maxiter=args.eval_local_fit_maxiter,
                eval_fit_maxiter=args.eval_fit_maxiter,
                strict_errors=True,
                network_eval_mode=args.network_eval_mode,
            )
            by_agg_exps[agg].append(exp)

    curves: Dict[str, Dict[str, Any]] = {}
    final_metrics: Dict[str, Dict[str, float]] = {}
    for agg, exps in by_agg_exps.items():
        curves[agg] = _aggregate_histories(exps)
        finals = [float(_history_series(e)[-1]) for e in exps]
        jumps = [float(np.max(np.abs(np.diff(_history_series(e))))) if len(_history_series(e)) > 1 else 0.0 for e in exps]
        stds = [float(np.std(_history_series(e))) for e in exps]
        final_metrics[agg] = {
            "final_mae_mean": float(np.mean(finals)),
            "final_mae_std": float(np.std(finals)),
            "max_jump_mean": float(np.mean(jumps)),
            "round_std_mean": float(np.mean(stds)),
        }

    fig_path = out_dir / "fair_eval_network_mae.png"
    _plot_eval(
        curves,
        fig_path,
        title=(
            "Fair decentralized comparison | LVP uses hybrid topology only; "
            "baselines use static Jaccard"
        ),
    )

    ranking = sorted(final_metrics.items(), key=lambda kv: kv[1]["final_mae_mean"])
    payload = {
        "stage_a": {
            "beta": args.stagea_beta,
            "gamma": args.stagea_gamma,
            "trials": trials,
            "best_config": best_cfg,
            "selected_lvp_tau_eval": lvp_tau_eval,
        },
        "stage_b": {
            "eval_seeds": eval_seeds,
            "scenario": {
                "partition": args.eval_partition,
                "attack_strategy": args.eval_attack_strategy,
                "attack_scale": args.eval_attack_scale,
                "malicious_frac": args.eval_malicious_frac,
                "rounds": args.eval_rounds,
                "local_epochs": args.eval_local_epochs,
                "network_eval_mode": args.network_eval_mode,
            },
            "method_policy": {
                "lvp": "hybrid topology (dynamic)",
                "baselines": "static jaccard topology",
            },
            "final_metrics": final_metrics,
            "final_ranking": [{"aggregator": k, **v} for k, v in ranking],
            "curves": curves,
        },
        "artifacts": {
            "figure": str(fig_path),
        },
    }

    json_path = out_dir / "fair_pipeline_results.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    report_lines = [
        "# LVP Fair Pipeline",
        "",
        "## Stage A best LVP config",
        f"- lambda_jaccard: {best_cfg['lambda_jaccard']}",
        f"- tau_cos_min: {best_cfg['tau_cos_min']}",
        f"- lvp_alpha: {best_cfg['lvp_alpha']}",
        f"- lvp_self_weight: {best_cfg['lvp_self_weight']}",
        f"- selected lvp tau (eval): {lvp_tau_eval}",
        "",
        "## Stage B ranking (mean final MAE)",
    ]
    for row in payload["stage_b"]["final_ranking"]:
        report_lines.append(
            f"- {row['aggregator']}: {row['final_mae_mean']:.6f} +- {row['final_mae_std']:.6f}"
        )
    report_lines.extend(
        [
            "",
            "## Notes",
            "- LVP uses hybrid dynamic topology only.",
            "- Baselines use static Jaccard topology to keep policy separation explicit.",
        ]
    )
    report_path = out_dir / "fair_pipeline_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved: {fig_path}")
    print(f"Saved: {json_path}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
