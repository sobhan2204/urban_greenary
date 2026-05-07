from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from utils.config_loader import resolve_config_path, load_config, resolve_path
from utils.io_utils import setup_logging
from utils.data_collection import collect_gangotri_tiles
from vegetation.pipeline import run_vegetation_pipeline
from avalanche.pipeline import run_avalanche_pipeline
from utils.visualization_utils import build_rgb_composite, overlay_mask, save_image
from utils.report_generator import generate_report


def _sample_highlight_coords(
    score_map: np.ndarray,
    mask: np.ndarray,
    center_lat: float,
    center_lng: float,
    deg_per_px: float,
    size: int,
    max_points: int = 30,
) -> list[list[float]]:
    """Return lat/lng points for highlighted pixels, evenly spaced."""
    rows, cols = np.where(mask)
    if not rows.size:
        return []
    # Score and shuffle, keep top-scoring
    scores = score_map[rows, cols]
    order = np.argsort(-scores)
    # Take up to max_points, evenly spaced from top results
    indices = order[:max_points]
    result = []
    for i in indices:
        r, c = int(rows[i]), int(cols[i])
        dlat = (size // 2 - r) * deg_per_px
        dlng = (c - size // 2) * deg_per_px
        result.append([round(center_lat + dlat, 6), round(center_lng + dlng, 6)])
    return result


def combine_scores(
    vegetation_results: list[dict],
    avalanche_results: list[dict],
    tiles: list[dict],
    config: dict,
) -> list[dict[str, object]]:
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

    report_dir = resolve_path(config, "paths", "data", "output") / "reports"
    vegetation_threshold = overlay_cfg["vegetation_min"]

    combined = []
    for tile in tiles:
        tile_id = tile["tile_id"]
        veg = vegetation_by_tile.get(tile_id)
        avalanche = avalanche_by_tile.get(tile_id)
        if veg is None or avalanche is None:
            logging.warning("Missing scores for tile %s", tile_id)
            continue
        combined_score = veg["score"] * (1.0 - avalanche["score"])
        features = avalanche.get("features", {})
        mask = combined_score >= overlay_cfg["combined_min"]
        base_rgb = build_rgb_composite(tile["sentinel"], rgb_cfg["rgb_band_indices"], eps)
        overlay = overlay_mask(base_rgb, mask, tuple(color_cfg["combined"]), alpha)

        output_name = naming["combined"].format(tile_id=tile_id) + f".{fmt}"
        output_path = Path(output_dir) / output_name
        save_image(overlay, output_path, dpi)

        report_result = generate_report(
            vegetation_score=veg["score"],
            avalanche_score=avalanche["score"],
            combined_score=combined_score,
            slope=features.get("slope"),
            aspect=features.get("aspect"),
            elevation=features.get("elevation"),
            tile_id=tile_id,
            output_dir=report_dir,
            vegetation_threshold=vegetation_threshold,
        )

        combined.append({
            "tile_id": tile_id,
            "score": combined_score,
            "report": report_result["report"],
            "report_files": report_result["files"],
        })
    return combined


def run_pipeline_for_point(
    lat: float,
    lng: float,
    config: dict | None = None,
    config_path: str | None = None,
    target_date: str | None = None,
) -> dict[str, str]:
    """Run the full pipeline for a single clicked coordinate.

    Fetches a satellite tile at (lat, lng), runs vegetation and avalanche
    scoring, combines the results, and saves three map images.

    Returns
    -------
    dict with keys vegetation_path, avalanche_path, combined_path,
    plus analysis/report fields for UI rendering.
    """
    if config is None:
        cp = config_path if config_path else None
        config = load_config(resolve_config_path(cp))

    # Build a tile at the clicked location via GEE
    from utils.interactive_collection import fetch_satellite_image, build_tile

    if target_date is None:
        from datetime import date
        target_date = date.today().isoformat()

    gee_result = fetch_satellite_image(lat, lng, target_date)
    tile = build_tile(gee_result, config)
    tiles = [tile]
    source = gee_result.get("source", "Unknown")

    # Run the three pipeline stages
    vegetation_results = run_vegetation_pipeline(config, tiles)
    avalanche_results = run_avalanche_pipeline(config, tiles)
    combined_results = combine_scores(vegetation_results, avalanche_results, tiles, config)

    # Extract highlighted coordinates for display on the map
    veg_cfg = config["visualization"]["overlay_thresholds"]
    aval_cfg = config["visualization"]["overlay_thresholds"]
    comb_cfg = config["visualization"]["overlay_thresholds"]
    target_size = config["data_collection"]["tile_size"]["height"]
    scale = 10 if source == "Sentinel-2" else 30
    deg_per_px = scale / 111320.0

    veg_scores = vegetation_results[0]["score"] if vegetation_results else np.zeros((target_size, target_size))
    aval_scores = avalanche_results[0]["score"] if avalanche_results else np.zeros((target_size, target_size))
    veg_mask = veg_scores >= veg_cfg["vegetation_min"]
    aval_mask = aval_scores <= aval_cfg["avalanche_safe_max"]
    combined_score = veg_scores * (1.0 - aval_scores)
    comb_mask = combined_score >= comb_cfg["combined_min"]

    veg_coords = _sample_highlight_coords(veg_scores, veg_mask, lat, lng, deg_per_px, target_size)
    aval_coords = _sample_highlight_coords(aval_scores, aval_mask, lat, lng, deg_per_px, target_size)
    comb_coords = _sample_highlight_coords(combined_score, comb_mask, lat, lng, deg_per_px, target_size)

    # Resolve the output filenames
    tile_id = tile["tile_id"]
    naming = config["visualization"]["output_naming"]
    fmt = config["visualization"]["format"]

    veg_name = naming["vegetation"].format(tile_id=tile_id) + f".{fmt}"
    aval_name = naming["avalanche"].format(tile_id=tile_id) + f".{fmt}"
    comb_name = naming["combined"].format(tile_id=tile_id) + f".{fmt}"

    report = combined_results[0].get("report") if combined_results else None
    report_files = combined_results[0].get("report_files") if combined_results else None
    report_urls = None
    if report_files:
        report_urls = {
            "json": f"/data/output/reports/{report_files['json']}",
            "txt": f"/data/output/reports/{report_files['txt']}",
        }

    return {
        "vegetation_path": f"/data/output/final/{veg_name}",
        "avalanche_path": f"/data/output/final/{aval_name}",
        "combined_path": f"/data/output/final/{comb_name}",
        "tile_id": tile_id,
        "image_date": gee_result.get("image_date", "Unknown"),
        "source": gee_result.get("source", "Unknown"),
        "veg_coords": veg_coords,
        "aval_coords": aval_coords,
        "comb_coords": comb_coords,
        "analysis": report,
        "report_files": report_files,
        "report_urls": report_urls,
    }


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
