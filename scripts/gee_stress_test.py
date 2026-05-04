from __future__ import annotations

import argparse
import logging

from utils.config_loader import resolve_config_path, load_config
from utils.data_collection import (
    _sample_gee_image,
    _normalize_gee_band_name,
    _build_valid_mask,
    _compute_mask_ratio,
    _find_patch_region,
)


def _init_gee(project_id: str):
    import ee

    ee.Initialize(project=project_id)
    return ee


def _get_region(config: dict, ee):
    bbox = config["data_collection"]["bounding_boxes"][0]
    region = ee.Geometry.Rectangle([bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]])
    return region, bbox


def _make_constant_image(ee, bands: list[str], value: float = 1.0):
    return ee.Image.constant([value] * len(bands)).rename(bands)


def _run_fully_masked(ee, bands: list[str], region, scale: float) -> None:
    image = _make_constant_image(ee, bands).updateMask(ee.Image(0))
    try:
        _sample_gee_image(image, bands, region, ee, scale, 0.0, 30, 4, 100000000)
        logging.error("Fully masked patch: UNEXPECTED success")
    except RuntimeError as exc:
        logging.info("Fully masked patch: expected failure (%s)", exc)


def _run_outside_footprint(ee, bands: list[str], region, scale: float) -> None:
    bbox = region.bounds().getInfo()["coordinates"][0]
    min_lon, min_lat = bbox[0]
    max_lon, max_lat = bbox[2]
    small = ee.Geometry.Rectangle([min_lon, min_lat, min_lon + 0.01, min_lat + 0.01])
    outside = ee.Geometry.Rectangle([max_lon + 0.5, max_lat + 0.5, max_lon + 0.6, max_lat + 0.6])
    image = _make_constant_image(ee, bands).clip(small)
    try:
        _sample_gee_image(image, bands, outside, ee, scale, 0.0, 30, 4, 100000000)
        logging.error("Outside footprint: UNEXPECTED success")
    except RuntimeError as exc:
        logging.info("Outside footprint: expected failure (%s)", exc)


def _run_cloud_scene(config: dict, ee, bands: list[str], region, scale: float) -> None:
    gee_cfg = config["data_collection"]["gee"]
    window = gee_cfg["date_windows"][0]
    cloud_max = gee_cfg["cloud_filters"][0]
    collection = ee.ImageCollection(gee_cfg["sentinel_collection"])
    collection = collection.filterDate(window["start_date"], window["end_date"])
    collection = collection.filterBounds(region)
    collection = collection.filter(ee.Filter.lte(gee_cfg["cloud_property"], cloud_max))

    size = collection.size().getInfo()
    logging.info("Cloud scene collection size=%s", size)
    if size == 0:
        logging.warning("Cloud scene test skipped: no images found")
        return

    image = collection.median()
    mask = _build_valid_mask(image, bands, ee)
    ratio = _compute_mask_ratio(mask, region, 30, 4, 100000000, ee)
    logging.info("Cloud scene valid ratio=%.3f", ratio)

    patch_region = _find_patch_region(mask, region, config["data_collection"]["tile_size"]["height"], scale, ee)
    if patch_region is None:
        logging.warning("Cloud scene test skipped: no valid patch")
        return

    try:
        _sample_gee_image(image, bands, patch_region, ee, scale, 0.0, 30, 4, 100000000)
        logging.info("Cloud scene sampling: success")
    except RuntimeError as exc:
        logging.warning("Cloud scene sampling: failed (%s)", exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    config_path = resolve_config_path(args.config)
    config = load_config(config_path)

    gee_cfg = config["data_collection"]["gee"]
    ee = _init_gee(gee_cfg["project_id"])

    bands = config["data_collection"]["sentinel"]["bands"]
    gee_bands = [_normalize_gee_band_name(band) for band in bands]
    region, _ = _get_region(config, ee)
    scale = gee_cfg.get("sample_scale", 10)

    _run_fully_masked(ee, gee_bands, region, scale)
    _run_outside_footprint(ee, gee_bands, region, scale)
    _run_cloud_scene(config, ee, gee_bands, region, scale)


if __name__ == "__main__":
    main()
