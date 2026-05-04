from __future__ import annotations

from pathlib import Path
from typing import Any
import logging
import numpy as np
import torch

from utils.config_loader import resolve_path
from vegetation.inference import load_trained_model, predict_probabilities
from vegetation.data_loader import prepare_tile
from vegetation.scoring import compute_vegetation_score
from vegetation.visualization import save_overlay_map
from utils.visualization_utils import build_rgb_composite


def run_vegetation_pipeline(config: dict, tiles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    model = load_trained_model(config)
    device = torch.device(config["inference"]["device"])
    model.to(device)

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
            logging.warning("Skipping vegetation inference for invalid tile %s", tile.get("tile_id"))
            continue
        sample = prepare_tile(tile, config)
        inputs = torch.from_numpy(sample["input"]).unsqueeze(0).float().to(device)
        probs = predict_probabilities(model, inputs)[0]
        score = compute_vegetation_score(probs, sample["dem"], config)
        base_rgb = build_rgb_composite(tile["sentinel"], rgb_cfg["rgb_band_indices"], eps)
        mask = score >= overlay_cfg["vegetation_min"]

        output_name = naming["vegetation"].format(tile_id=tile["tile_id"]) + f".{fmt}"
        output_path = Path(output_dir) / output_name
        save_overlay_map(base_rgb, mask, output_path, tuple(color_cfg["vegetation"]), alpha, dpi)

        results.append({"tile_id": tile["tile_id"], "score": score, "meta": tile.get("meta", {})})

    return results
