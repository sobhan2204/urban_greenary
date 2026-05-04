from __future__ import annotations

import numpy as np


def generate_pseudo_labels(ndvi: np.ndarray, ndsi: np.ndarray, config: dict) -> np.ndarray:
    class_map = config["labeling"]["class_map"]
    ndvi_cfg = config["labeling"]["ndvi_thresholds"]
    ndsi_cfg = config["labeling"]["ndsi_thresholds"]
    barren_cfg = config["labeling"]["barren_thresholds"]

    labels = np.full(ndvi.shape, class_map["background"], dtype=np.uint8)

    snow_mask = ndsi >= ndsi_cfg["snow_min"]
    labels[snow_mask] = class_map["snow"]

    water_mask = ndvi <= ndvi_cfg["water_max"]
    labels[water_mask] = class_map["water"]

    veg_mask = (ndvi >= ndvi_cfg["vegetation_min"]) & (ndvi <= ndvi_cfg["vegetation_max"])
    labels[veg_mask] = class_map["vegetation"]

    barren_mask = (ndvi <= barren_cfg["ndvi_max"]) & (ndsi <= barren_cfg["ndsi_max"])
    labels[barren_mask] = class_map["barren"]

    return labels
