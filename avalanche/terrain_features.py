from __future__ import annotations

import numpy as np

from utils.geo_utils import compute_slope_aspect, compute_curvature


def compute_terrain_features(dem: np.ndarray, config: dict) -> dict[str, np.ndarray]:
    if dem.ndim == 3:
        dem_slice = dem[0]
    else:
        dem_slice = dem
    spacing = config["scoring"]["avalanche"]["dem_spacing"]
    eps = config["preprocessing"]["normalization"]["epsilon"]
    slope, aspect = compute_slope_aspect(dem_slice, spacing, eps)
    curvature = compute_curvature(dem_slice, spacing)
    return {"slope": slope, "aspect": aspect, "curvature": curvature, "elevation": dem_slice}
