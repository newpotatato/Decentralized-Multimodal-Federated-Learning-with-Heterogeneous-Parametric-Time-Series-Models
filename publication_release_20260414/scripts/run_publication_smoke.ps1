$ErrorActionPreference = "Stop"

Write-Output "[1/3] Install deps if needed: pip install -r requirements.txt"
Write-Output "[2/3] Verify required publication files"
python scripts/verify_publication_package.py

Write-Output "[3/3] Run publication smoke tests"
python -m pytest tests/test_publication_artifacts.py tests/test_param_vector.py tests/test_mcc_loader.py -q

Write-Output "Publication smoke run completed."
