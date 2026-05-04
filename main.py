from __future__ import annotations

import argparse
import logging
from pathlib import Path

from utils.config_loader import resolve_config_path, load_config, resolve_path
from utils.io_utils import setup_logging
from utils.data_collection import collect_gangotri_tiles
from vegetation.pipeline import run_vegetation_pipeline
from avalanche.pipeline import run_avalanche_pipeline
from utils.visualization_utils import build_rgb_composite, overlay_mask, save_image


def combine_scores(
    vegetation_results: list[dict],
    avalanche_results: list[dict],
    tiles: list[dict],
    config: dict,
) -> list[dict[str, float]]:
    vegetation_by_tile = {item["tile_id"]: item for item in vegetation_results}
    avalanche_by_tile = {item["tile_id"]: item for item in avalanche_results}
    naming = config["visualization"]["output_naming"]
    dpi = config["visualization"]["dpi"]
    fmt = config["visualization"]["format"]
    output_dir = resolve_path(config, "paths", "data", "output_final")
    alpha = config["visualization"]["overlay_alpha"]
    overlay_cfg = config["visualization"]["overlay_thresholds"]
    color_cfg = config["visualization"]["overlay_colors"]
    rgb_cfg = config["visualization"]["base_rgb"]
    eps = config["preprocessing"]["normalization"]["epsilon"]

    combined = []
    for tile in tiles:
        tile_id = tile["tile_id"]
        veg = vegetation_by_tile.get(tile_id)
        avalanche = avalanche_by_tile.get(tile_id)
        if veg is None or avalanche is None:
            logging.warning("Missing scores for tile %s", tile_id)
            continue
        combined_score = veg["score"] * (1.0 - avalanche["score"])
        mask = combined_score >= overlay_cfg["combined_min"]
        base_rgb = build_rgb_composite(tile["sentinel"], rgb_cfg["rgb_band_indices"], eps)
        overlay = overlay_mask(base_rgb, mask, tuple(color_cfg["combined"]), alpha)

        output_name = naming["combined"].format(tile_id=tile_id) + f".{fmt}"
        output_path = Path(output_dir) / output_name
        save_image(overlay, output_path, dpi)
        combined.append({"tile_id": tile_id})
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    setup_logging("INFO")

    tiles = collect_gangotri_tiles(config)
    vegetation_results = run_vegetation_pipeline(config, tiles)
    avalanche_results = run_avalanche_pipeline(config, tiles)
    combine_scores(vegetation_results, avalanche_results, tiles, config)


if __name__ == "__main__":
    main()
