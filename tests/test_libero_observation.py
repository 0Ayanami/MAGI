"""Regression coverage for actual signed Panda finger observations."""

import numpy as np
import pytest

from vlm_orchestrator.aux_benchmarks.libero_observation import normalized_panda_gripper


@pytest.mark.parametrize(
    ("positions", "expected"),
    [([0.04, -0.04], 0.0), ([0.0387, -0.0387], 0.0325),
     ([0.0, 0.0], 1.0), ([0.02, -0.02], 0.5),
     ([0.05, -0.05], 0.0), ([0.04, 0.04], 0.0)],
)
def test_signed_finger_positions(positions, expected):
    result = normalized_panda_gripper(positions)
    assert result.shape == (1,)
    assert result.dtype == np.float64
    assert result[0] == pytest.approx(expected)
