from __future__ import annotations

import numpy as np


def compute_slope_aspect(dem: np.ndarray, spacing: float, eps: float) -> tuple[np.ndarray, np.ndarray]:
    grad_y, grad_x = np.gradient(dem, spacing, spacing)
    slope = np.degrees(np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2)))
    aspect = (np.degrees(np.arctan2(-grad_x, grad_y + eps)) + 360) % 360
    return slope, aspect


def compute_curvature(dem: np.ndarray, spacing: float) -> np.ndarray:
    grad_y, grad_x = np.gradient(dem, spacing, spacing)
    grad_yy, _ = np.gradient(grad_y, spacing, spacing)
    _, grad_xx = np.gradient(grad_x, spacing, spacing)
    return grad_xx + grad_yy
