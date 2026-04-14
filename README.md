# Decentralized FL Repository (Curated Layout)

This repository is organized for publication-first usage.

## Main Entry Point

- `publication_release_20260414/` - curated publication package with reproducible scripts, tests, and prepared article artifacts.

## Archive

- `archive/` - previous project implementation and legacy materials kept for historical traceability.

## Quick Start

```powershell
cd publication_release_20260414
pip install -r requirements-lock.txt
python scripts/verify_publication_package.py
python -m pytest tests/test_publication_artifacts.py tests/test_param_vector.py tests/test_mcc_loader.py -q
```

## Rebuild Key Publication Figures

```powershell
cd publication_release_20260414
./scripts/reproduce_key_figures.ps1
```

## Notes

- Active publication workflow should use only `publication_release_20260414/`.
- Legacy files are preserved under `archive/` and are not part of the current publication pipeline.
