from __future__ import annotations

import numpy as np


def compute_vegetation_score(probabilities: np.ndarray, dem: np.ndarray, config: dict) -> np.ndarray:
    score_cfg = config["scoring"]["vegetation"]
    eps = config["preprocessing"]["normalization"]["epsilon"]
    class_indices = score_cfg["class_indices"]

    score = probabilities[class_indices].mean(axis=0)

    elevation_limit = score_cfg["elevation_limit"]
    if elevation_limit is not None:
        if dem.ndim == 3:
            dem_slice = dem[0]
        else:
            dem_slice = dem
        score = np.where(dem_slice <= elevation_limit, score, 0.0)

    if score_cfg.get("normalize", False):
        score = (score - score.min()) / (score.max() - score.min() + eps)

    return score
