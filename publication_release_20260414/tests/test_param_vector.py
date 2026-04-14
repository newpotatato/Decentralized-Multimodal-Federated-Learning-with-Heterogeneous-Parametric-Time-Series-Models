"""Federated param dict uses ndarray scalars correctly in statsmodels start vector."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_CORE = Path(__file__).resolve().parents[1] / "federated_learning" / "core"
sys.path.insert(0, str(_CORE))

from base_model import vector_from_param_names  # noqa: E402


def test_vector_from_param_names_numpy_arrays():
    names = ["ar.L1", "sigma2"]
    d = {"ar.L1": np.array([0.61]), "sigma2": np.array([19.0])}
    v = vector_from_param_names(names, d)
    assert v is not None
    np.testing.assert_allclose(v, [0.61, 19.0], rtol=0, atol=1e-9)
