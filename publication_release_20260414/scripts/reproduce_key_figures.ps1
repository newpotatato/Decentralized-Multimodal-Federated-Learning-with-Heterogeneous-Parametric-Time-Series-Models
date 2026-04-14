$ErrorActionPreference = "Stop"

# Rebuild selected boxplots from prepared seeds
python federated_learning/experiments/fig3_boxplot_with_without_fedavg_multiseed.py `
  --input-dirs `
  federated_learning/artifacts/article_package_current_run_20260410/raw/seeds/seed42 `
  federated_learning/artifacts/article_package_current_run_20260410/raw/seeds/seed52 `
  federated_learning/artifacts/article_package_current_run_20260410/raw/seeds/seed62 `
  --out-dir federated_learning/artifacts/article_package_current_run_20260410/plots/boxplots/lvpfl_fedavg_only `
  --single-panel `
  --exclude-aggregators defta,balance,push_sum `
  --method-labels "lvp=LVP-FL" `
  --panel-title "10 rounds, 10 agents, fixed static topology, kappa = Jaccard coeff"

python federated_learning/experiments/fig3_boxplot_with_without_fedavg_multiseed.py `
  --input-dirs `
  federated_learning/artifacts/article_package_current_run_20260410/raw/seeds/seed42 `
  federated_learning/artifacts/article_package_current_run_20260410/raw/seeds/seed52 `
  federated_learning/artifacts/article_package_current_run_20260410/raw/seeds/seed62 `
  --out-dir federated_learning/artifacts/article_package_current_run_20260410/plots/boxplots/all_methods_no_pushsum `
  --single-panel `
  --exclude-aggregators push_sum `
  --method-labels "lvp=LVP-FL" `
  --panel-title "10 rounds, 10 agents, fixed static topology, kappa = Jaccard coeff"

# Rebuild tau/kappa_0 ablation figures
python federated_learning/experiments/ablate_tau_alpha_article_rounds10.py

# Ensure publication package uses the rebuilt ablation figure under article_package path
New-Item -ItemType Directory -Force federated_learning/artifacts/article_package_current_run_20260410/plots/ablation | Out-Null
Copy-Item `
  federated_learning/artifacts/ablation_tau_alpha_article_rounds10/tau_ablation_rounds10_article.png `
  federated_learning/artifacts/article_package_current_run_20260410/plots/ablation/tau_ablation_rounds10_article.png `
  -Force

Write-Output "Key figures rebuilt."
Write-Output "Ablation figure synced to: federated_learning/artifacts/article_package_current_run_20260410/plots/ablation/tau_ablation_rounds10_article.png"
