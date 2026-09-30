#!/usr/bin/env bash
# Pipeline v2 (30.09.2026): corrected attacks, receiver-side hybrid cosine, profiles as in the
# manuscript, ablation on the same per-seed partitions as the scenarios, five seeds,
# plus reference runs (no exchange, no attack, no exogenous regressors).
#
# Usage (Git Bash):  bash run_pipeline_v2.sh
set -u
BASE="$(cd "$(dirname "$0")" && pwd)"          # experiments_datafusion2023/
ROOT="$(dirname "$BASE")"                      # repository root
ART="$BASE/artifacts_pipeline_v2"
PY="${PYTHON:-python}"                         # override: PYTHON=/path/to/python bash run_pipeline_v2.sh
EXP="federated_learning/experiments"
SEEDS="42,52,62,72,82"
cd "$ROOT/publication_release_20260414" || exit 1
mkdir -p "$ART/logs"

COMMON="--base-path $BASE --model DynamicLinearModel --n-clients 10 --rounds 10 --local-epochs 1 --local-fit-maxiter 5 --column-partition random --network-eval-mode refit --sync-topic-groups 0"

ablate() {  # name, similarity mode
  "$PY" "$EXP/ablate_tau_alpha_article_rounds10.py" $COMMON \
    --out-dir "$ART/ablation_$1" --seed-list "$SEEDS" \
    --malicious-frac 0.4 --attack-strategy label_flip --attack-scale 2.5 \
    --similarity-mode "$2" --lambda-jaccard 0.2 --tau-cos-min -1 \
    --tau-grid "0.20,0.30,0.40,0.50,0.60,0.70" --alpha-grid "0.20,0.30,0.40,0.53,0.60" \
    --fixed-alpha 0.4 --partition-per-seed --alpha-at-best-tau --grid-workers 2 \
    > "$ART/logs/ablation_$1.log" 2>&1
  echo "[done] ablation_$1 exit=$?"
}

echo "=== STAGE 1: ablations (Jaccard and hybrid) $(date +%H:%M)"
ablate jaccard jaccard &
ablate hybrid jaccard_cosine_hybrid &
wait

read -r TAU ALPHA < <("$PY" -c "
import json
d=json.load(open(r'$ART/ablation_hybrid/tau_alpha_ablation_rounds10_article_summary.json'))
print(d['best_tau']['tau'], d['best_alpha']['alpha'])")
echo "Hybrid best: kappa0=$TAU alpha=$ALPHA"
[ -n "${TAU:-}" ] || { echo "ablation failed"; exit 1; }

METHODS="lvp,decentralized_fedavg,defta,balance,push_sum,local"
SHARED="decentralized_fedavg:hybrid,defta:hybrid,balance:hybrid,push_sum:hybrid,local:hybrid"
NATIVE="local:ring"

scenario() {  # name, malicious frac, attack, scale, topology map, extra args...
  local name=$1 frac=$2 atk=$3 scale=$4 topo=$5; shift 5
  "$PY" "$EXP/compare_method_topologies_multiseed.py" $COMMON \
    --out-dir "$ART/$name" --methods "$METHODS" --seeds "$SEEDS" \
    --malicious-frac "$frac" --attack-strategy "$atk" --attack-scale "$scale" \
    --lvp-tau "$TAU" --lvp-alpha "$ALPHA" --lambda-jaccard 0.2 --tau-cos-min -1 \
    --topology-map "$topo" "$@" > "$ART/logs/$name.log" 2>&1
  echo "[done] $name exit=$?"
}

echo "=== STAGE 2a: scenarios A, B, C $(date +%H:%M)"
scenario scenario_A 0.4 noise_colluded 5.0 "$SHARED" &
scenario scenario_B 0.4 label_flip 2.5 "$SHARED" &
scenario scenario_C 0.4 label_flip 2.5 "$NATIVE" &
wait
echo "=== STAGE 2b: references (no attack; no exogenous regressors) $(date +%H:%M)"
scenario clean_shared 0.0 label_flip 2.5 "$SHARED" &
scenario clean_native 0.0 label_flip 2.5 "$NATIVE" &
scenario scenario_B_noexog 0.4 label_flip 2.5 "$SHARED" --no-exog &
wait
echo "=== ALL DONE $(date +%H:%M)"
