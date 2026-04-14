#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

_EXP = Path(__file__).resolve().parent
_FL = _EXP.parent


def _run(cmd: List[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Build hybrid selected-tau experiment package")
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--seeds", type=str, default="42,52,62")
    p.add_argument("--out-dir", type=str, default=str(_FL / "artifacts" / "article_package_current_run_20260410" / "experiment_with_more_rounds_hybrid"))
    p.add_argument("--n-clients", type=int, default=20)
    p.add_argument("--column-partition", type=str, default="contiguous", choices=["contiguous", "strided"])
    p.add_argument("--malicious-frac", type=float, default=0.25)
    p.add_argument("--attack-strategy", type=str, default="noise_colluded")
    p.add_argument("--attack-scale", type=float, default=5.0)
    p.add_argument("--similarity-mode", type=str, default="jaccard_cosine_hybrid")
    p.add_argument("--lambda-jaccard", type=float, default=0.5)
    p.add_argument("--tau-cos-min", type=float, default=-1.0)
    p.add_argument("--lvp-alpha", type=float, default=0.6)
    p.add_argument("--network-eval-mode", type=str, default="proxy", choices=["proxy", "refit"])
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    raw_dirs = []
    for seed in seeds:
        seed_dir = out_dir / "raw" / "seeds" / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        raw_dirs.append(seed_dir)
        _run([
            sys.executable,
            str(_EXP / "fig3_decentralized_methods.py"),
            "--out-dir", str(seed_dir),
            "--rounds", str(args.rounds),
            "--seed", str(seed),
            "--n-clients", str(args.n_clients),
            "--column-partition", args.column_partition,
            "--malicious-frac", str(args.malicious_frac),
            "--attack-strategy", args.attack_strategy,
            "--attack-scale", str(args.attack_scale),
            "--similarity-tau", str(args.tau),
            "--similarity-mode", args.similarity_mode,
            "--lambda-jaccard", str(args.lambda_jaccard),
            "--tau-cos-min", str(args.tau_cos_min),
            "--lvp-alpha", str(args.lvp_alpha),
            "--network-eval-mode", args.network_eval_mode,
        ])

    _run([
        sys.executable,
        str(_EXP / "aggregate_fig3_multiseed.py"),
        "--input-dirs",
        *[str(d) for d in raw_dirs],
        "--out-dir", str(out_dir / "plots" / "multiseed"),
    ])
    _run([
        sys.executable,
        str(_EXP / "fig3_boxplot_with_without_fedavg_multiseed.py"),
        "--input-dirs",
        *[str(d) for d in raw_dirs],
        "--out-dir", str(out_dir / "plots" / "boxplots" / "all_methods"),
        "--single-panel",
    ])
    _run([
        sys.executable,
        str(_EXP / "fig3_boxplot_with_without_fedavg_multiseed.py"),
        "--input-dirs",
        *[str(d) for d in raw_dirs],
        "--out-dir", str(out_dir / "plots" / "boxplots" / "left2"),
        "--single-panel",
        "--exclude-aggregators", "defta,balance,push_sum",
    ])
    for seed_dir in raw_dirs:
        seed = int(seed_dir.name.replace("seed", ""))
        _run([
            sys.executable,
            str(_EXP / "plot_decentralized_coherence.py"),
            "--input-json", str(seed_dir / "fig3_decentralized_methods.json"),
            "--out-dir", str(out_dir / "plots" / "coherence" / f"seed{seed}"),
        ])
    _run([
        sys.executable,
        str(_EXP / "plot_decentralized_coherence_multiseed.py"),
        "--input-dirs",
        *[str(d) for d in raw_dirs],
        "--out-dir", str(out_dir / "plots" / "coherence" / "multiseed"),
    ])
    _run([
        sys.executable,
        str(_EXP / "hybrid_topology_round1.py"),
        "--out-dir", str(out_dir / "topology"),
        "--model", "DynamicLinearModel",
        "--column-partition", args.column_partition,
        "--similarity-tau", str(args.tau),
        "--lambda-jaccard", str(args.lambda_jaccard),
        "--tau-cos-min", str(args.tau_cos_min),
    ])

    print(f"Done. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
