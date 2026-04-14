#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


AGGS = ["lvp", "decentralized_fedavg", "defta", "balance", "push_sum"]


@dataclass(frozen=True)
class Candidate:
    tau: float
    lambda_jaccard: float
    tau_cos_min: float
    alpha: float


def _seed_to_reference_json(seed: int, artifacts_dir: Path) -> Path:
    if seed == 42:
        return artifacts_dir / "decentralized_comparison_article_selected_rounds30" / "fig3_decentralized_methods.json"
    return artifacts_dir / f"decentralized_comparison_article_selected_rounds30_seed{seed}" / "fig3_decentralized_methods.json"


def _extract_curves(payload: Dict) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for rec in payload.get("results", []):
        agg = str(rec.get("aggregator", ""))
        if agg not in AGGS:
            continue
        ys = [float(r.get("network_mae", np.nan)) for r in (rec.get("history") or [])]
        out[agg] = np.asarray(ys, dtype=float)
    return out


def _final_rank(curves: Dict[str, np.ndarray]) -> List[str]:
    pairs: List[Tuple[str, float]] = []
    for agg in AGGS:
        y = curves.get(agg)
        if y is None or y.size == 0:
            continue
        pairs.append((agg, float(y[-1])))
    pairs.sort(key=lambda x: x[1])
    return [x[0] for x in pairs]


def _rank_distance(a: List[str], b: List[str]) -> float:
    pa = {name: i for i, name in enumerate(a)}
    pb = {name: i for i, name in enumerate(b)}
    common = [x for x in a if x in pb]
    if not common:
        return 1.0
    dist = float(sum(abs(pa[x] - pb[x]) for x in common))
    max_dist = float(len(common) * max(len(common) - 1, 1))
    return dist / max_dist if max_dist > 0 else 0.0


def _score_vs_reference(curves: Dict[str, np.ndarray], ref_curves: Dict[str, np.ndarray]) -> Dict[str, float]:
    nrmse_vals: List[float] = []
    for agg in AGGS:
        y = curves.get(agg)
        r = ref_curves.get(agg)
        if y is None or r is None or y.size == 0 or r.size == 0:
            continue
        m = int(min(y.size, r.size))
        y = y[:m]
        r = r[:m]
        denom = float(np.mean(np.abs(r))) + 1e-12
        rmse = float(np.sqrt(np.mean((y - r) ** 2)))
        nrmse_vals.append(rmse / denom)

    mean_nrmse = float(np.mean(nrmse_vals)) if nrmse_vals else 1.0
    rank_ref = _final_rank(ref_curves)
    rank_cur = _final_rank(curves)
    rank_penalty = _rank_distance(rank_cur, rank_ref)

    # Weighted score: prioritize shape similarity, then ranking consistency.
    total = mean_nrmse + 0.35 * rank_penalty
    return {
        "mean_nrmse": mean_nrmse,
        "rank_penalty": rank_penalty,
        "total_score": total,
    }


def _load_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_candidate(
    candidate: Candidate,
    seed: int,
    out_dir: Path,
    exp_script: Path,
) -> Path:
    run_dir = out_dir / f"tau_{candidate.tau:.2f}_lam_{candidate.lambda_jaccard:.2f}_tcos_{candidate.tau_cos_min:.2f}_a_{candidate.alpha:.2f}" / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(exp_script),
        "--out-dir",
        str(run_dir),
        "--rounds",
        "30",
        "--local-epochs",
        "1",
        "--seed",
        str(seed),
        "--malicious-frac",
        "0.25",
        "--attack-strategy",
        "noise_colluded",
        "--attack-scale",
        "5.0",
        "--network-eval-mode",
        "proxy",
        "--column-partition",
        "contiguous",
        "--similarity-mode",
        "jaccard_cosine_hybrid",
        "--lambda-jaccard",
        str(candidate.lambda_jaccard),
        "--tau-cos-min",
        str(candidate.tau_cos_min),
        "--similarity-tau",
        str(candidate.tau),
        "--lvp-alpha",
        str(candidate.alpha),
    ]
    subprocess.run(cmd, check=True)
    return run_dir / "fig3_decentralized_methods.json"


def _default_candidates() -> List[Candidate]:
    return [
        Candidate(0.45, 0.25, 0.00, 0.53),
        Candidate(0.50, 0.25, 0.00, 0.53),
        Candidate(0.55, 0.25, 0.00, 0.53),
        Candidate(0.45, 0.35, 0.05, 0.53),
        Candidate(0.50, 0.35, 0.05, 0.53),
        Candidate(0.55, 0.35, 0.05, 0.53),
        Candidate(0.45, 0.20, 0.00, 0.45),
        Candidate(0.50, 0.20, 0.00, 0.45),
        Candidate(0.55, 0.20, 0.00, 0.45),
        Candidate(0.45, 0.25, 0.05, 0.60),
        Candidate(0.50, 0.25, 0.05, 0.60),
        Candidate(0.55, 0.25, 0.05, 0.60),
    ]


def main() -> None:
    p = argparse.ArgumentParser(description="Search hybrid topology params to match article reference dynamics")
    p.add_argument("--seeds", type=str, default="42,52,62")
    p.add_argument(
        "--out-dir",
        type=str,
        default="federated_learning/artifacts/topology_alignment_search",
    )
    p.add_argument(
        "--artifacts-dir",
        type=str,
        default="federated_learning/artifacts",
    )
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = Path(args.artifacts_dir)

    exp_script = Path(__file__).resolve().parent / "fig3_decentralized_methods.py"
    candidates = _default_candidates()

    ref_curves_by_seed: Dict[int, Dict[str, np.ndarray]] = {}
    for seed in seeds:
        ref_json = _seed_to_reference_json(seed, artifacts_dir)
        ref_payload = _load_json(ref_json)
        ref_curves_by_seed[seed] = _extract_curves(ref_payload)

    rows = []
    total_runs = len(candidates) * len(seeds)
    run_idx = 0

    for cand in candidates:
        per_seed_scores: List[float] = []
        per_seed_nrmse: List[float] = []
        per_seed_rank: List[float] = []

        for seed in seeds:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] candidate={cand} seed={seed}")
            out_json = _run_candidate(cand, seed, out_dir, exp_script)
            curves = _extract_curves(_load_json(out_json))
            score = _score_vs_reference(curves, ref_curves_by_seed[seed])
            per_seed_scores.append(float(score["total_score"]))
            per_seed_nrmse.append(float(score["mean_nrmse"]))
            per_seed_rank.append(float(score["rank_penalty"]))

        rows.append(
            {
                "candidate": asdict(cand),
                "score_mean": float(np.mean(per_seed_scores)),
                "score_std": float(np.std(per_seed_scores, ddof=0)),
                "nrmse_mean": float(np.mean(per_seed_nrmse)),
                "rank_penalty_mean": float(np.mean(per_seed_rank)),
            }
        )

    rows_sorted = sorted(rows, key=lambda r: r["score_mean"])
    top_k = max(1, int(args.top_k))
    top_rows = rows_sorted[:top_k]

    out_json = out_dir / "topology_alignment_search_summary.json"
    out_json.write_text(
        json.dumps(
            {
                "seeds": seeds,
                "n_candidates": len(candidates),
                "top_k": top_k,
                "results_sorted": rows_sorted,
                "best": top_rows[0] if top_rows else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    out_md = out_dir / "topology_alignment_search_summary.md"
    lines = [
        "# Topology Alignment Search Summary",
        "",
        f"Seeds: {', '.join(str(s) for s in seeds)}",
        f"Candidates evaluated: {len(candidates)}",
        "",
        "## Top candidates (lower is better)",
    ]
    for i, row in enumerate(top_rows, start=1):
        c = row["candidate"]
        lines.append(
            f"{i}. tau={c['tau']:.2f}, lambda={c['lambda_jaccard']:.2f}, tau_cos_min={c['tau_cos_min']:.2f}, alpha={c['alpha']:.2f} | "
            f"score={row['score_mean']:.6f} +- {row['score_std']:.6f}, nrmse={row['nrmse_mean']:.6f}, rank_penalty={row['rank_penalty_mean']:.6f}"
        )
    out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {out_json}")
    print(f"Saved: {out_md}")


if __name__ == "__main__":
    main()