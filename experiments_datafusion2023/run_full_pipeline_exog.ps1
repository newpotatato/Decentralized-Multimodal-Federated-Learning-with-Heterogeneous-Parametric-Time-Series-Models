param([switch]$SkipAblation)  # reuse existing ablation summaries, rerun scenarios only
$ErrorActionPreference = "Stop"
$BASE       = $PSScriptRoot                       # experiments_datafusion2023
$ROOT       = Split-Path $BASE -Parent
Start-Transcript -Path "$BASE\pipeline_exog_transcript.log" -Force
Set-Location "$ROOT\publication_release_20260414"

$ARTIFACTS  = "$BASE\artifacts_pipeline_exog"
$SCRIPTS    = "federated_learning\experiments"
$PYTHON     = if ($env:PYTHON) { $env:PYTHON } else { "python" }

$MAL_FRAC  = 0.4
$ROUNDS    = 10
$N_CLIENTS = 10
$LAMBDA    = 0.2

# Topology override: all comparison methods use hybrid (for Scenarios A and B)
$HYBRID_ALL = "decentralized_fedavg:hybrid,defta:hybrid,balance:hybrid,push_sum:hybrid"

# Exogenous news (Fontanka + Reuters headline sentiment, lagged 10 days) enter the DLM.
# Stage 0 (model selection figure) is not used in the article and is skipped here.

# For scenario comparison we need a model that shows differentiation between aggregators.
# DynamicLinearModel is sensitive to parameter aggregation under Byzantine attack;
# the auto-selected best model is used only for the selection figure above.
$SCENARIO_MODEL = "DynamicLinearModel"
Write-Output "Scenario/ablation model: $SCENARIO_MODEL"

# ============================================================
Write-Output ""
Write-Output "=== STAGE 1a: Ablation - pure Jaccard (model=$SCENARIO_MODEL) ==="
# ============================================================
$ABL_J_DIR = "$ARTIFACTS\ablation_jaccard"
New-Item -ItemType Directory -Force $ABL_J_DIR | Out-Null

if (-not $SkipAblation) {
& $PYTHON "$SCRIPTS\ablate_tau_alpha_article_rounds10.py" `
    --base-path         $BASE `
    --out-dir           $ABL_J_DIR `
    --model             $SCENARIO_MODEL `
    --n-clients         $N_CLIENTS `
    --rounds            $ROUNDS `
    --local-epochs      1 `
    --local-fit-maxiter 5 `
    --seed-list         "42,52,62" `
    --malicious-frac    $MAL_FRAC `
    --attack-strategy   label_flip `
    --attack-scale      2.5 `
    --column-partition  random `
    --similarity-mode   jaccard `
    --tau-grid          "0.20,0.30,0.40,0.50,0.60,0.70" `
    --alpha-grid        "0.20,0.30,0.40,0.53,0.60" `
    --fixed-alpha       0.4 `
    --fixed-tau         0.35 `
    --network-eval-mode refit

if ($LASTEXITCODE -ne 0) { throw "Jaccard ablation failed (exit $LASTEXITCODE)" }
}

$ablJ         = Get-Content "$ABL_J_DIR\tau_alpha_ablation_rounds10_article_summary.json" | ConvertFrom-Json
$BEST_TAU_J   = [double]$ablJ.best_tau.tau
$BEST_ALPHA_J = [double]$ablJ.best_alpha.alpha
Write-Output "Jaccard  -> best tau=$BEST_TAU_J  best alpha=$BEST_ALPHA_J"

# ============================================================
Write-Output ""
Write-Output "=== STAGE 1b: Ablation - Hybrid Jaccard+Cosine (model=$SCENARIO_MODEL) ==="
# ============================================================
$ABL_H_DIR = "$ARTIFACTS\ablation_hybrid"
New-Item -ItemType Directory -Force $ABL_H_DIR | Out-Null

if (-not $SkipAblation) {
& $PYTHON "$SCRIPTS\ablate_tau_alpha_article_rounds10.py" `
    --base-path         $BASE `
    --out-dir           $ABL_H_DIR `
    --model             $SCENARIO_MODEL `
    --n-clients         $N_CLIENTS `
    --rounds            $ROUNDS `
    --local-epochs      1 `
    --local-fit-maxiter 5 `
    --seed-list         "42,52,62" `
    --malicious-frac    $MAL_FRAC `
    --attack-strategy   label_flip `
    --attack-scale      2.5 `
    --column-partition  random `
    --similarity-mode   jaccard_cosine_hybrid `
    --lambda-jaccard    $LAMBDA `
    --tau-grid          "0.20,0.30,0.40,0.50,0.60,0.70" `
    --alpha-grid        "0.20,0.30,0.40,0.53,0.60" `
    --fixed-alpha       0.4 `
    --fixed-tau         0.35 `
    --network-eval-mode refit

if ($LASTEXITCODE -ne 0) { throw "Hybrid ablation failed (exit $LASTEXITCODE)" }
}

$ablH         = Get-Content "$ABL_H_DIR\tau_alpha_ablation_rounds10_article_summary.json" | ConvertFrom-Json
$BEST_TAU_H   = [double]$ablH.best_tau.tau
$BEST_ALPHA_H = [double]$ablH.best_alpha.alpha
Write-Output "Hybrid   -> best tau=$BEST_TAU_H  best alpha=$BEST_ALPHA_H"

# ============================================================
Write-Output ""
Write-Output "=== STAGE 2: Scenario A - noise_colluded x5, 40%, shared hybrid topology ==="
# ============================================================
$SC_DIR = "$ARTIFACTS\scenario_A"
foreach ($d in @(
    "$SC_DIR\raw\seeds\seed42", "$SC_DIR\raw\seeds\seed52", "$SC_DIR\raw\seeds\seed62",
    "$SC_DIR\plots\analyze",
    "$SC_DIR\plots\boxplots\all_methods",
    "$SC_DIR\plots\boxplots\general_methods",
    "$SC_DIR\plots\boxplots\lvp_only"
)) { New-Item -ItemType Directory -Force $d | Out-Null }

& $PYTHON "$SCRIPTS\compare_method_topologies_multiseed.py" `
    --base-path         $BASE `
    --out-dir           $SC_DIR `
    --model             $SCENARIO_MODEL `
    --methods           lvp,decentralized_fedavg,defta,balance,push_sum `
    --seeds             42,52,62 `
    --rounds            $ROUNDS `
    --local-epochs      1 `
    --local-fit-maxiter 5 `
    --n-clients         $N_CLIENTS `
    --column-partition  random `
    --malicious-frac    $MAL_FRAC `
    --attack-strategy   noise_colluded `
    --attack-scale      5.0 `
    --network-eval-mode refit `
    --lvp-tau           $BEST_TAU_H `
    --lambda-jaccard    $LAMBDA `
    --lvp-alpha         $BEST_ALPHA_H `
    --topology-map      $HYBRID_ALL

if ($LASTEXITCODE -ne 0) { throw "Scenario A experiments failed" }

& $PYTHON "$BASE\aggregate_seeds.py" "$SC_DIR\raw\seeds" "$SC_DIR\plots\analyze\aggregated_3seeds.json"
if ($LASTEXITCODE -ne 0) { throw "Scenario A aggregation failed" }

& $PYTHON "$SCRIPTS\analyze_and_plot.py" `
    "$SC_DIR\plots\analyze\aggregated_3seeds.json" "$SC_DIR\plots\analyze"
if ($LASTEXITCODE -ne 0) { Write-Warning "Scenario A analyze_and_plot failed (non-fatal)" }

$d42 = "$SC_DIR\raw\seeds\seed42"
$d52 = "$SC_DIR\raw\seeds\seed52"
$d62 = "$SC_DIR\raw\seeds\seed62"
$TITLE = "Scenario A: noise_colluded x5, mal=40%, shared hybrid topology"

& $PYTHON "$SCRIPTS\fig3_boxplot_with_without_fedavg_multiseed.py" `
    --input-dirs $d42 $d52 $d62 --out-dir "$SC_DIR\plots\boxplots\all_methods" `
    --single-panel --method-labels "lvp=LVP-FL" --panel-title $TITLE
if ($LASTEXITCODE -ne 0) { Write-Warning "Scenario A boxplot all_methods failed" }

& $PYTHON "$SCRIPTS\fig3_boxplot_with_without_fedavg_multiseed.py" `
    --input-dirs $d42 $d52 $d62 --out-dir "$SC_DIR\plots\boxplots\general_methods" `
    --single-panel --exclude-aggregators lvp --panel-title "$TITLE (comparison methods)"
if ($LASTEXITCODE -ne 0) { Write-Warning "Scenario A boxplot general_methods failed" }

& $PYTHON "$SCRIPTS\fig3_boxplot_with_without_fedavg_multiseed.py" `
    --input-dirs $d42 $d52 $d62 --out-dir "$SC_DIR\plots\boxplots\lvp_only" `
    --single-panel --exclude-aggregators decentralized_fedavg,defta,balance,push_sum `
    --method-labels "lvp=LVP-FL" --panel-title "$TITLE (LVP-FL only)"
if ($LASTEXITCODE -ne 0) { Write-Warning "Scenario A boxplot lvp_only failed" }

# ============================================================
Write-Output ""
Write-Output "=== STAGE 3: Scenario B - label_flip x2.5, 40%, shared hybrid topology ==="
# ============================================================
$SC_DIR = "$ARTIFACTS\scenario_B"
foreach ($d in @(
    "$SC_DIR\raw\seeds\seed42", "$SC_DIR\raw\seeds\seed52", "$SC_DIR\raw\seeds\seed62",
    "$SC_DIR\plots\analyze",
    "$SC_DIR\plots\boxplots\all_methods",
    "$SC_DIR\plots\boxplots\general_methods",
    "$SC_DIR\plots\boxplots\lvp_only"
)) { New-Item -ItemType Directory -Force $d | Out-Null }

& $PYTHON "$SCRIPTS\compare_method_topologies_multiseed.py" `
    --base-path         $BASE `
    --out-dir           $SC_DIR `
    --model             $SCENARIO_MODEL `
    --methods           lvp,decentralized_fedavg,defta,balance,push_sum `
    --seeds             42,52,62 `
    --rounds            $ROUNDS `
    --local-epochs      1 `
    --local-fit-maxiter 5 `
    --n-clients         $N_CLIENTS `
    --column-partition  random `
    --malicious-frac    $MAL_FRAC `
    --attack-strategy   label_flip `
    --attack-scale      2.5 `
    --network-eval-mode refit `
    --lvp-tau           $BEST_TAU_H `
    --lambda-jaccard    $LAMBDA `
    --lvp-alpha         $BEST_ALPHA_H `
    --topology-map      $HYBRID_ALL

if ($LASTEXITCODE -ne 0) { throw "Scenario B experiments failed" }

& $PYTHON "$BASE\aggregate_seeds.py" "$SC_DIR\raw\seeds" "$SC_DIR\plots\analyze\aggregated_3seeds.json"
if ($LASTEXITCODE -ne 0) { throw "Scenario B aggregation failed" }

& $PYTHON "$SCRIPTS\analyze_and_plot.py" `
    "$SC_DIR\plots\analyze\aggregated_3seeds.json" "$SC_DIR\plots\analyze"
if ($LASTEXITCODE -ne 0) { Write-Warning "Scenario B analyze_and_plot failed (non-fatal)" }

$d42 = "$SC_DIR\raw\seeds\seed42"
$d52 = "$SC_DIR\raw\seeds\seed52"
$d62 = "$SC_DIR\raw\seeds\seed62"
$TITLE = "Scenario B: label_flip x2.5, mal=40%, shared hybrid topology"

& $PYTHON "$SCRIPTS\fig3_boxplot_with_without_fedavg_multiseed.py" `
    --input-dirs $d42 $d52 $d62 --out-dir "$SC_DIR\plots\boxplots\all_methods" `
    --single-panel --method-labels "lvp=LVP-FL" --panel-title $TITLE
if ($LASTEXITCODE -ne 0) { Write-Warning "Scenario B boxplot all_methods failed" }

& $PYTHON "$SCRIPTS\fig3_boxplot_with_without_fedavg_multiseed.py" `
    --input-dirs $d42 $d52 $d62 --out-dir "$SC_DIR\plots\boxplots\general_methods" `
    --single-panel --exclude-aggregators lvp --panel-title "$TITLE (comparison methods)"
if ($LASTEXITCODE -ne 0) { Write-Warning "Scenario B boxplot general_methods failed" }

& $PYTHON "$SCRIPTS\fig3_boxplot_with_without_fedavg_multiseed.py" `
    --input-dirs $d42 $d52 $d62 --out-dir "$SC_DIR\plots\boxplots\lvp_only" `
    --single-panel --exclude-aggregators decentralized_fedavg,defta,balance,push_sum `
    --method-labels "lvp=LVP-FL" --panel-title "$TITLE (LVP-FL only)"
if ($LASTEXITCODE -ne 0) { Write-Warning "Scenario B boxplot lvp_only failed" }

# ============================================================
Write-Output ""
Write-Output "=== STAGE 4: Scenario C - label_flip x2.5, 40%, canonical topologies ==="
# ============================================================
$SC_DIR = "$ARTIFACTS\scenario_C"
foreach ($d in @(
    "$SC_DIR\raw\seeds\seed42", "$SC_DIR\raw\seeds\seed52", "$SC_DIR\raw\seeds\seed62",
    "$SC_DIR\plots\analyze",
    "$SC_DIR\plots\boxplots\all_methods",
    "$SC_DIR\plots\boxplots\general_methods",
    "$SC_DIR\plots\boxplots\lvp_only"
)) { New-Item -ItemType Directory -Force $d | Out-Null }

& $PYTHON "$SCRIPTS\compare_method_topologies_multiseed.py" `
    --base-path         $BASE `
    --out-dir           $SC_DIR `
    --model             $SCENARIO_MODEL `
    --methods           lvp,decentralized_fedavg,defta,balance,push_sum `
    --seeds             42,52,62 `
    --rounds            $ROUNDS `
    --local-epochs      1 `
    --local-fit-maxiter 5 `
    --n-clients         $N_CLIENTS `
    --column-partition  random `
    --malicious-frac    $MAL_FRAC `
    --attack-strategy   label_flip `
    --attack-scale      2.5 `
    --network-eval-mode refit `
    --lvp-tau           $BEST_TAU_H `
    --lambda-jaccard    $LAMBDA `
    --lvp-alpha         $BEST_ALPHA_H

if ($LASTEXITCODE -ne 0) { throw "Scenario C experiments failed" }

& $PYTHON "$BASE\aggregate_seeds.py" "$SC_DIR\raw\seeds" "$SC_DIR\plots\analyze\aggregated_3seeds.json"
if ($LASTEXITCODE -ne 0) { throw "Scenario C aggregation failed" }

& $PYTHON "$SCRIPTS\analyze_and_plot.py" `
    "$SC_DIR\plots\analyze\aggregated_3seeds.json" "$SC_DIR\plots\analyze"
if ($LASTEXITCODE -ne 0) { Write-Warning "Scenario C analyze_and_plot failed (non-fatal)" }

$d42 = "$SC_DIR\raw\seeds\seed42"
$d52 = "$SC_DIR\raw\seeds\seed52"
$d62 = "$SC_DIR\raw\seeds\seed62"
$TITLE = "Scenario C: label_flip x2.5, mal=40%, canonical topologies"

& $PYTHON "$SCRIPTS\fig3_boxplot_with_without_fedavg_multiseed.py" `
    --input-dirs $d42 $d52 $d62 --out-dir "$SC_DIR\plots\boxplots\all_methods" `
    --single-panel --method-labels "lvp=LVP-FL" --panel-title $TITLE
if ($LASTEXITCODE -ne 0) { Write-Warning "Scenario C boxplot all_methods failed" }

& $PYTHON "$SCRIPTS\fig3_boxplot_with_without_fedavg_multiseed.py" `
    --input-dirs $d42 $d52 $d62 --out-dir "$SC_DIR\plots\boxplots\general_methods" `
    --single-panel --exclude-aggregators lvp --panel-title "$TITLE (comparison methods)"
if ($LASTEXITCODE -ne 0) { Write-Warning "Scenario C boxplot general_methods failed" }

& $PYTHON "$SCRIPTS\fig3_boxplot_with_without_fedavg_multiseed.py" `
    --input-dirs $d42 $d52 $d62 --out-dir "$SC_DIR\plots\boxplots\lvp_only" `
    --single-panel --exclude-aggregators decentralized_fedavg,defta,balance,push_sum `
    --method-labels "lvp=LVP-FL" --panel-title "$TITLE (LVP-FL only)"
if ($LASTEXITCODE -ne 0) { Write-Warning "Scenario C boxplot lvp_only failed" }

# ============================================================
Write-Output ""
Write-Output "=== ALL DONE ==="
Write-Output "Artifacts: $ARTIFACTS"
Write-Output ""
Write-Output "Generated plots:"
Get-ChildItem "$ARTIFACTS" -Recurse -Filter "*.png" | ForEach-Object { Write-Output "  $($_.FullName)" }
Write-Output ""
Write-Output "Summary:"
Write-Output "  Jaccard    : tau=$BEST_TAU_J  alpha=$BEST_ALPHA_J"
Write-Output "  Hybrid     : tau=$BEST_TAU_H  alpha=$BEST_ALPHA_H"
