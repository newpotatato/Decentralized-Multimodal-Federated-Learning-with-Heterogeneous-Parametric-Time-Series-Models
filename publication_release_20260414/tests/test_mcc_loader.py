"""MCC loader must prefer real datasets (see data_utils.resolve_mcc_csv_path)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
_DL = REPO / "federated_learning" / "data_loaders"
sys.path.insert(0, str(_DL))

from data_utils import load_mcc_series, mcc_csv_candidate_paths, resolve_mcc_csv_path  # noqa: E402


def test_at_least_one_real_candidate_exists_or_synthetic_allowed():
    allow = os.environ.get("FL_ALLOW_SYNTHETIC_MCC", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    any_real = any(p.is_file() for p in mcc_csv_candidate_paths(REPO))
    synth = (_DL / "data" / "dat_mcc_synthetic_smoke.csv").is_file()
    assert any_real or (allow and synth), (
        "Place real MCC under 01_data_transactions/ or data_LVP/01_data_transactions/, "
        "or set FL_ALLOW_SYNTHETIC_MCC=1 for bundled smoke CSV."
    )


def test_resolve_mcc_not_bundled_smoke_when_real_present():
    allow = os.environ.get("FL_ALLOW_SYNTHETIC_MCC", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    p = resolve_mcc_csv_path(REPO)
    assert p.is_file()
    if not allow:
        assert "synthetic_smoke" not in str(p).replace("\\", "/").lower()


def test_load_mcc_series_meets_client_builder_minimum():
    df = load_mcc_series(REPO)
    assert len(df) >= 150
    assert "date" in df.columns
    num_cols = [c for c in df.columns if c != "date"]
    assert len(num_cols) >= 1


def test_build_clients_strided_partition():
    from data_utils import build_clients_from_mcc

    df = load_mcc_series(REPO)
    clients_c = build_clients_from_mcc(df, None, n_clients=8, column_partition="contiguous")
    clients_s = build_clients_from_mcc(df, None, n_clients=8, column_partition="strided")
    assert len(clients_c) == len(clients_s) >= 2
    # Strided mix: first client's amt should differ from contiguous (same seed data)
    assert not np.allclose(
        clients_c[0]["amt"].values[:50],
        clients_s[0]["amt"].values[:50],
        rtol=1e-6,
        atol=1e-3,
    )
