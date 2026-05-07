from __future__ import annotations

from pathlib import Path
from typing import Any
import logging

from utils.config_loader import resolve_path
from avalanche.terrain_features import compute_terrain_features
from avalanche.scoring import compute_avalanche_score
from avalanche.visualization import save_overlay_map
from utils.visualization_utils import build_rgb_composite


def run_avalanche_pipeline(config: dict, tiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output_dir = resolve_path(config, "paths", "data", "output_final")
    naming = config["visualization"]["output_naming"]
    dpi = config["visualization"]["dpi"]
    fmt = config["visualization"]["format"]
    alpha = config["visualization"]["overlay_alpha"]
    overlay_cfg = config["visualization"]["overlay_thresholds"]
    color_cfg = config["visualization"]["overlay_colors"]
    rgb_cfg = config["visualization"]["base_rgb"]
    eps = config["preprocessing"]["normalization"]["epsilon"]

    results = []
    for tile in tiles:
        if not tile.get("meta", {}).get("valid", True):
            logging.warning("Skipping avalanche inference for invalid tile %s", tile.get("tile_id"))
            continue
        features = compute_terrain_features(tile["dem"], config)
        score = compute_avalanche_score(features, config)
        base_rgb = build_rgb_composite(tile["sentinel"], rgb_cfg["rgb_band_indices"], eps)
        output_name = naming["avalanche"].format(tile_id=tile["tile_id"]) + f".{fmt}"
        output_path = Path(output_dir) / output_name
        mask = score <= overlay_cfg["avalanche_safe_max"]
        save_overlay_map(base_rgb, mask, output_path, tuple(color_cfg["avalanche_safe"]), alpha, dpi)

        results.append({
            "tile_id": tile["tile_id"],
            "score": score,
            "meta": tile.get("meta", {}),
            "features": features,
        })

    return results
