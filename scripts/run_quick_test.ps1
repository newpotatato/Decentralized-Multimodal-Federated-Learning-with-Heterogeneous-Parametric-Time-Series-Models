# Quick Test Script for Byzantine Resilience Experiments
# Run this to verify your setup before full experiments
#
# Usage: .\run_quick_test.ps1

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Byzantine Resilience - Quick Test" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check Python
Write-Host "[1/5] Checking Python installation..." -ForegroundColor Yellow
$pythonVersion = python --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Python not found. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}
Write-Host "  Found: $pythonVersion" -ForegroundColor Green

# Check dependencies
Write-Host ""
Write-Host "[2/5] Checking dependencies..." -ForegroundColor Yellow
$missingPackages = @()

$requiredPackages = @("numpy", "pandas", "matplotlib", "sklearn", "scipy", "statsmodels")
foreach ($package in $requiredPackages) {
    python -c "import $package" 2>$null
    if ($LASTEXITCODE -ne 0) {
        $missingPackages += $package
    }
}

if ($missingPackages.Count -gt 0) {
    Write-Host "  Missing packages: $($missingPackages -join ', ')" -ForegroundColor Red
    Write-Host "  Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements_publication.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "  All dependencies installed" -ForegroundColor Green
}

# Check data files
Write-Host ""
Write-Host "[3/5] Checking data files..." -ForegroundColor Yellow
$dataPath = "01_data_transactions\dat_mcc.csv"
if (Test-Path $dataPath) {
    Write-Host "  Found: $dataPath" -ForegroundColor Green
} else {
    Write-Host "  Warning: $dataPath not found (will use synthetic data)" -ForegroundColor DarkYellow
}

# Run quick test
Write-Host ""
Write-Host "[4/5] Running quick test (5-10 minutes)..." -ForegroundColor Yellow
Write-Host "  This tests 2 models with minimal configuration" -ForegroundColor Gray
Write-Host ""

$startTime = Get-Date
python reproduce_experiments.py --mode quick --output-dir "quick_test_results"
$exitCode = $LASTEXITCODE
$elapsed = (Get-Date) - $startTime

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "[5/5] SUCCESS! Test completed in $($elapsed.ToString('mm\:ss'))" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Review results in: quick_test_results/" -ForegroundColor White
    Write-Host "  2. Run full experiments: python reproduce_experiments.py --mode all" -ForegroundColor White
    Write-Host "  3. See README_REPRODUCTION.md for details" -ForegroundColor White
} else {
    Write-Host "[5/5] FAILED! Test failed with exit code $exitCode" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Check error messages above" -ForegroundColor White
    Write-Host "  2. Verify Python 3.8+ is installed" -ForegroundColor White
    Write-Host "  3. Try: pip install -r requirements_publication.txt" -ForegroundColor White
    Write-Host "  4. See README_REPRODUCTION.md for help" -ForegroundColor White
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
