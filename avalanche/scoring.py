from __future__ import annotations

import numpy as np


def _normalize_range(values: np.ndarray, min_val: float, max_val: float, eps: float) -> np.ndarray:
    return np.clip((values - min_val) / (max_val - min_val + eps), 0.0, 1.0)


def _aspect_score(aspect: np.ndarray, ranges: list[list[float]]) -> np.ndarray:
    mask = np.zeros_like(aspect, dtype=bool)
    for start, end in ranges:
        if start <= end:
            mask |= (aspect >= start) & (aspect <= end)
        else:
            mask |= (aspect >= start) | (aspect <= end)
    return mask.astype(np.float32)


def compute_avalanche_score(features: dict[str, np.ndarray], config: dict) -> np.ndarray:
    cfg = config["scoring"]["avalanche"]
    eps = config["preprocessing"]["normalization"]["epsilon"]

    slope_cfg = cfg["slope"]
    aspect_cfg = cfg["aspect"]
    curvature_cfg = cfg["curvature"]
    elevation_cfg = cfg["elevation"]
    weights = cfg["weights"]

    slope_score = _normalize_range(features["slope"], slope_cfg["min_deg"], slope_cfg["max_deg"], eps)
    aspect_score = _aspect_score(features["aspect"], aspect_cfg["favorable_ranges"])
    curvature_score = _normalize_range(features["curvature"], curvature_cfg["min"], curvature_cfg["max"], eps)
    elevation_score = _normalize_range(features["elevation"], elevation_cfg["min"], elevation_cfg["max"], eps)

    weighted = (
        slope_score * weights["slope"]
        + aspect_score * weights["aspect"]
        + curvature_score * weights["curvature"]
        + elevation_score * weights["elevation"]
    )
    weight_sum = weights["slope"] + weights["aspect"] + weights["curvature"] + weights["elevation"]
    score = weighted / (weight_sum + eps)

    if cfg.get("normalize", False):
        score = (score - score.min()) / (score.max() - score.min() + eps)

    return score
