"""Sensor conversions for the LIBERO Panda observation stream."""

import numpy as np


def normalized_panda_gripper(finger_positions) -> np.ndarray:
    """Return 0 for open and 1 for closed, preserving signed finger symmetry."""
    width = float(np.mean(np.abs(np.asarray(finger_positions, dtype=float))))
    return np.array([1.0 - np.clip(width / 0.04, 0.0, 1.0)], dtype=np.float64)
