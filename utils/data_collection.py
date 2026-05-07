from __future__ import annotations

from pathlib import Path
import calendar
import json
import logging
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F

from utils.config_loader import resolve_path
from utils.geo_utils import compute_slope_aspect
from vegetation.preprocessing import compute_ndvi, compute_ndsi, compute_ndwi


def _bbox_intersects(a: dict, b: dict) -> bool:
    return not (
        a["max_lat"] < b["min_lat"]
        or a["min_lat"] > b["max_lat"]
        or a["max_lon"] < b["min_lon"]
        or a["min_lon"] > b["max_lon"]
    )


def _load_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        meta = {
            "crs": src.crs.to_string() if src.crs else None,
            "transform": src.transform,
            "height": src.height,
            "width": src.width,
        }
    return data, meta


def _resize_stack(stack: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    if stack.ndim == 2:
        stack = stack[None, :, :]
    if stack.shape[1] == target_height and stack.shape[2] == target_width:
        return stack
    tensor = torch.from_numpy(stack).unsqueeze(0)
    resized = F.interpolate(tensor, size=(target_height, target_width), mode="bilinear", align_corners=False)
    return resized.squeeze(0).numpy()


def _init_gee(project_id: str, timeout_ms: int | None = None):
    try:
        import ee
    except ImportError as exc:
        raise ImportError("earthengine-api is required for provider=gee") from exc

    try:
        ee.Initialize(project=project_id)
        if timeout_ms:
            ee.data.setDeadline(timeout_ms)
    except Exception as exc:
        raise RuntimeError(
            "Earth Engine initialization failed. Run `earthengine authenticate` and verify the project ID."
        ) from exc
    return ee


def _apply_scl_mask(image, scl_band: str, keep_classes: list[int]):
    scl = image.select(scl_band)
    mask = None
    for cls in keep_classes:
        cls_mask = scl.eq(cls)
        mask = cls_mask if mask is None else mask.Or(cls_mask)
    return image.updateMask(mask) if mask is not None else image


def _normalize_gee_band_name(band: str) -> str:
    if band.startswith("B0") and band[2:].isdigit():
        return "B" + band[2:].lstrip("0")
    return band


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _shift_months(date_value: datetime, months: int) -> datetime:
    year_offset, month_index = divmod(date_value.month - 1 + months, 12)
    year = date_value.year + year_offset
    month = month_index + 1
    day = min(date_value.day, calendar.monthrange(year, month)[1])
    return datetime(year, month, day)


def _expand_window(window: dict, months: int) -> dict:
    return {
        "name": f"{window['name']}_expanded",
        "start_date": _shift_months(_parse_date(window["start_date"]), -months).strftime("%Y-%m-%d"),
        "end_date": _shift_months(_parse_date(window["end_date"]), months).strftime("%Y-%m-%d"),
    }


def _iter_window_variants(window: dict, expand_months: int) -> list[dict]:
    variants = [window]
    if expand_months and expand_months > 0:
        variants.append(_expand_window(window, expand_months))
    return variants


def _to_float_array(band_data) -> np.ndarray:
    arr = np.array(band_data)
    if arr.size == 0:
        return arr.astype(np.float32)
    if arr.dtype == object:
        arr = np.vectorize(lambda x: np.nan if x is None else float(x))(arr).astype(np.float32)
    else:
        arr = arr.astype(np.float32)
    return arr


def _compute_valid_fraction(stack: np.ndarray) -> float:
    if stack.size == 0:
        return 0.0
    valid = np.isfinite(stack)
    return float(valid.mean())


def _compute_band_stats(stack: np.ndarray) -> list[tuple[float, float]]:
    stats: list[tuple[float, float]] = []
    for band in stack:
        if np.isfinite(band).any():
            stats.append((float(np.nanmin(band)), float(np.nanmax(band))))
        else:
            stats.append((float("nan"), float("nan")))
    return stats


def load_dem_source(ee, source: str, band: str):
    dem_img = ee.Image(source)
    try:
        # GEE won't tell us if it's an Image or Collection until we ask for info
        bands = dem_img.bandNames().getInfo()
    except ee.ee_exception.EEException as e:
        # If it complains that it's an ImageCollection, mosaic it together!
        if 'not an Image' in str(e) or 'not found' in str(e):
            dem_img = ee.ImageCollection(source).mosaic()
            bands = dem_img.bandNames().getInfo()
        else:
            raise

    if band not in bands:
        raise RuntimeError(f"DEM band '{band}' not found in source '{source}'")
    return dem_img


def _sample_gee_image(
    image,
    bands: list[str],
    region,
    ee,
    scale: float,
    default_value: float,
    reduce_scale: float,
    reduce_tile_scale: int,
    max_pixels: int,
) -> np.ndarray:
    try:
        base = image.select(bands)
        selected = base.unmask(default_value).reproject("EPSG:4326", None, scale)
        footprint = selected.geometry()
        intersects = footprint.intersects(region, ee.ErrorMargin(1)).getInfo()
        logging.info("Patch footprint intersects=%s", intersects)
        if not intersects:
            raise RuntimeError("Region outside image footprint")

        stats = base.reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=region,
            scale=reduce_scale,
            maxPixels=max_pixels,
            tileScale=reduce_tile_scale,
            bestEffort=True,
        )
        count = stats.get(bands[0])
        count_val = 0 if count is None else int(count.getInfo())
        logging.info("Patch pixel count=%s", count_val)
        if count_val == 0:
            raise RuntimeError("Fully masked patch")

        valid_mask = base.mask().reduce(ee.Reducer.min()).reproject("EPSG:4326", None, scale).rename("valid")
        sample = selected.addBands(valid_mask).sampleRectangle(region=region, defaultValue=default_value)
        info = sample.getInfo()
        
        # --- Extract the nested 'properties' dictionary --
        properties = info.get("properties", {}) 

        arrays = []
        for band in bands:
            band_data = properties.get(band) # Pull from properties, not info
            if band_data is None:
                raise RuntimeError(f"Band '{band}' missing from sampleRectangle output")
            arrays.append(_to_float_array(band_data))

        valid_data = properties.get("valid") 
        if valid_data is None:
            raise RuntimeError("Valid mask missing from sampleRectangle output")
        
        valid_mask_arr = _to_float_array(valid_data)
        valid_mask_arr = valid_mask_arr > 0.5
        
        # --- THE FIX: Force compressed GEE arrays back to full 2D grids ---
        spatial_shape = arrays[0].shape
        if valid_mask_arr.shape != spatial_shape:
            valid_mask_arr = np.broadcast_to(valid_mask_arr, spatial_shape)

        stack = np.stack(arrays, axis=0)
        stack[:, ~valid_mask_arr] = np.nan

        if stack.size == 0:
            raise RuntimeError("Empty arrays returned from sampleRectangle")
        return stack
    except Exception as exc:
        raise RuntimeError(f"Sampling failed: {exc}") from exc


def _build_valid_mask(image, bands: list[str], ee):
    return image.select(bands).mask().reduce(ee.Reducer.min()).rename("valid")


def _compute_mask_ratio(mask, region, scale: float, tile_scale: int, max_pixels: int, ee) -> float:
    try:
        stats = mask.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=region,
            scale=scale,
            maxPixels=max_pixels,
            tileScale=tile_scale,
            bestEffort=True,
        )
        value = stats.get("valid")
        if value is None:
            return 0.0
        return float(value.getInfo())
    except Exception:
        return 0.0


def _find_patch_region(valid_mask, region, patch_size: int, scale: float, ee):
    """
    Finds a patch region by scanning for dense blocks of valid pixels 
    using a continuous moving average, rather than strict erosion.
    """
    # 1. Define the radius of our search window
    radius = max(int(patch_size // 2), 1)

    # 2. Create a square kernel representing our desired patch footprint
    kernel = ee.Kernel.square(radius=radius, units='pixels')

    # 3. Calculate the moving average (percentage of valid pixels in the kernel)
    # valid_mask is 1 (good) and 0 (bad). focal_mean gives a score from 0.0 to 1.0.
    patch_quality = valid_mask.focal_mean(kernel=kernel)

    # 4. Find center points where the surrounding patch is at least 95% valid
    # selfMask() drops any pixels that don't meet this standard
    high_quality_centers = patch_quality.gte(0.95).selfMask()

    # 5. Sample one of these excellent center points
    samples = high_quality_centers.sample(
        region=region, 
        numPixels=1, 
        scale=scale, 
        geometries=True
    )

    if samples.size().getInfo() == 0:
        return None

    # 6. Build the bounding box around our chosen center point
    point = ee.Feature(samples.first()).geometry()
    half_width_meters = float(patch_size) * 0.5 * float(scale)
    patch_bounds = point.buffer(half_width_meters).bounds()

    resolved = patch_bounds.getInfo()
    return ee.Geometry(resolved)


def _validate_range(values: np.ndarray, min_val: float, max_val: float) -> bool:
    if not np.isfinite(values).any():
        return False
    return float(np.nanmin(values)) >= min_val and float(np.nanmax(values)) <= max_val


def _validate_tile_data(
    sentinel: np.ndarray,
    dem: np.ndarray,
    ndvi: np.ndarray,
    ndwi: np.ndarray,
    ndsi: np.ndarray,
    slope: np.ndarray,
    config: dict,
    band_names: list[str],
) -> bool:
    val_cfg = config["data_validation"]
    sentinel_valid = _compute_valid_fraction(sentinel)
    if sentinel_valid < val_cfg["sentinel_valid_fraction_min"]:
        logging.warning("Sentinel valid fraction %.3f below threshold", sentinel_valid)
        return False

    dem_valid = _compute_valid_fraction(dem)
    if dem_valid < val_cfg["dem_valid_fraction_min"]:
        logging.warning("DEM valid fraction %.3f below threshold", dem_valid)
        return False

    band_stds = [float(np.nanstd(band)) for band in sentinel]
    if float(np.nanmax(band_stds)) < val_cfg["min_band_std"]:
        logging.warning("Sentinel bands appear constant (max std %.6f)", float(np.nanmax(band_stds)))
        return False

    band_flat = np.array([float(np.nanmax(band) - np.nanmin(band)) for band in sentinel])
    if float(np.nanmax(band_flat)) <= 0:
        logging.warning("Sentinel bands have zero dynamic range")
        return False

    band_ranges = val_cfg.get("band_value_ranges", {})
    for idx, name in enumerate(band_names):
        if name in band_ranges:
            min_val, max_val = band_ranges[name]
            if not _validate_range(sentinel[idx], min_val, max_val):
                logging.warning("Band %s out of expected range", name)
                return False

    if not _validate_range(ndvi, *val_cfg["ndvi_range"]):
        logging.warning("NDVI out of expected range")
        return False
    if not _validate_range(ndwi, *val_cfg["ndwi_range"]):
        logging.warning("NDWI out of expected range")
        return False
    if not _validate_range(ndsi, *val_cfg["ndsi_range"]):
        logging.warning("NDSI out of expected range")
        return False
    if not _validate_range(slope, *val_cfg["slope_range"]):
        logging.warning("Slope out of expected range")
        return False
    if not _validate_range(dem[0], *val_cfg["elevation_range"]):
        logging.warning("Elevation out of expected range")
        return False

    return True


def _collect_local(config: dict) -> list[dict[str, Any]]:
    manifest_path = resolve_path(config, "data_collection", "local", "manifest_path")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Tile manifest not found: {manifest_path}")

    if manifest_path.suffix.lower() == ".json":
        with manifest_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)
        df = pd.DataFrame(records)
    else:
        df = pd.read_csv(manifest_path)

    required = {"tile_id", "sentinel_path", "dem_path", "min_lat", "min_lon", "max_lat", "max_lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")

    tiles: list[dict[str, Any]] = []
    bboxes = config["data_collection"]["bounding_boxes"]

    for _, row in df.iterrows():
        tile_bbox = {
            "min_lat": float(row["min_lat"]),
            "min_lon": float(row["min_lon"]),
            "max_lat": float(row["max_lat"]),
            "max_lon": float(row["max_lon"]),
        }
        if not any(_bbox_intersects(tile_bbox, bbox) for bbox in bboxes):
            continue

        sentinel_path = Path(row["sentinel_path"]).expanduser()
        if not sentinel_path.is_absolute():
            sentinel_path = Path(config["runtime"]["root_dir"]) / sentinel_path
        dem_path = Path(row["dem_path"]).expanduser()
        if not dem_path.is_absolute():
            dem_path = Path(config["runtime"]["root_dir"]) / dem_path

        sentinel, s_meta = _load_raster(sentinel_path)
        dem, d_meta = _load_raster(dem_path)
        if dem.ndim == 2:
            dem = dem[None, :, :]

        tiles.append(
            {
                "tile_id": str(row["tile_id"]),
                "sentinel": sentinel,
                "dem": dem,
                "meta": {"bbox": tile_bbox, "sentinel": s_meta, "dem": d_meta},
            }
        )

    logging.info("Loaded %d tiles from local manifest", len(tiles))
    return tiles


def _collect_synthetic(config: dict) -> list[dict[str, Any]]:
    collection_cfg = config["data_collection"]
    rng = np.random.default_rng(collection_cfg["synthetic"]["seed"])
    num_tiles = collection_cfg["synthetic"]["num_tiles_per_bbox"]
    height = collection_cfg["tile_size"]["height"]
    width = collection_cfg["tile_size"]["width"]
    sentinel_channels = collection_cfg["sentinel"]["num_channels"]
    dem_channels = collection_cfg["dem"]["num_channels"]
    sentinel_min, sentinel_max = collection_cfg["synthetic"]["sentinel_value_range"]
    dem_min, dem_max = collection_cfg["synthetic"]["dem_value_range"]

    tiles: list[dict[str, Any]] = []
    tile_index = 0
    for bbox in collection_cfg["bounding_boxes"]:
        for _ in range(num_tiles):
            sentinel = rng.uniform(sentinel_min, sentinel_max, size=(sentinel_channels, height, width)).astype(
                np.float32
            )
            dem = rng.uniform(dem_min, dem_max, size=(dem_channels, height, width)).astype(np.float32)
            tiles.append(
                {
                    "tile_id": f"synthetic_{tile_index}",
                    "sentinel": sentinel,
                    "dem": dem,
                    "meta": {"bbox": bbox},
                }
            )
            tile_index += 1

    logging.info("Generated %d synthetic tiles", len(tiles))
    return tiles


def _collect_gee(config: dict) -> list[dict[str, Any]]:
    gee_cfg = config["data_collection"]["gee"]
    ee = _init_gee(gee_cfg["project_id"], gee_cfg.get("request_timeout_ms"))

    bands = config["data_collection"]["sentinel"]["bands"]
    gee_bands = [_normalize_gee_band_name(band) for band in bands]
    tile_size = config["data_collection"]["tile_size"]
    target_height = tile_size["height"]
    target_width = tile_size["width"]

    date_windows = gee_cfg.get("date_windows")
    if not date_windows:
        raise ValueError("data_collection.gee.date_windows is required")
    expand_months = gee_cfg.get("expand_months", 0)

    
    cloud_filters = gee_cfg.get("cloud_filters") or [10, 20, 30, 50]
    cloud_property = gee_cfg["cloud_property"]
    sample_scale = gee_cfg.get("sample_scale", 10)
    patch_sizes = gee_cfg.get("sample_patch_sizes") or [target_height]
    reduce_scale = gee_cfg.get("reduce_scale", 30)
    reduce_tile_scale = gee_cfg.get("reduce_tile_scale", 4)
    max_pixels = gee_cfg.get("max_pixels", 100000000)
    default_value = 0.0

    valid_fraction_min = config["data_validation"]["sentinel_valid_fraction_min"]
    dem_valid_min = config["data_validation"]["dem_valid_fraction_min"]

    scl_cfg = gee_cfg.get("scl_mask", {})
    mask_enabled_global = scl_cfg.get("enabled", False)
    keep_classes = scl_cfg.get("keep_classes")
    if keep_classes is None:
        keep_classes = [4, 5, 6, 7]
        if scl_cfg.get("allow_snow", True):
            keep_classes.append(11)
    min_survival_ratio = scl_cfg.get("min_survival_ratio", 0.05)

    s2_primary = ee.ImageCollection(gee_cfg["sentinel_collection"])
    collections = [("primary", s2_primary)]
    sentinel_fallback_cfg = gee_cfg.get("sentinel_fallback", {})
    if sentinel_fallback_cfg.get("enabled", False):
        collections.append(("fallback", ee.ImageCollection(sentinel_fallback_cfg["collection"])))

    dem_sources = gee_cfg.get("dem_sources")
    if not dem_sources:
        raise ValueError("data_collection.gee.dem_sources is required")

    tiles: list[dict[str, Any]] = []
    for idx, bbox in enumerate(config["data_collection"]["bounding_boxes"]):
        region = ee.Geometry.Rectangle([bbox["min_lon"], bbox["min_lat"], bbox["max_lon"], bbox["max_lat"]])

        sentinel = None
        sentinel_meta: dict[str, Any] = {}
        for window in date_windows:
            for variant in _iter_window_variants(window, expand_months):
                for cloud_max in cloud_filters:
                    for collection_name, base_collection in collections:
                        collection = base_collection.filterDate(variant["start_date"], variant["end_date"])
                        collection = collection.filter(ee.Filter.lte(cloud_property, cloud_max))
                        collection = collection.filterBounds(region)
                        size = collection.size().getInfo()
                        logging.info(
                            "Collection size %s for bbox %s window %s cloud<=%s (%s)",
                            size,
                            bbox,
                            variant["name"],
                            cloud_max,
                            collection_name,
                        )
                        if size == 0:
                            continue

                        composite_raw = collection.median()
                        raw_mask = _build_valid_mask(composite_raw, gee_bands, ee)
                        raw_ratio = _compute_mask_ratio(raw_mask, region, reduce_scale, reduce_tile_scale, max_pixels, ee)
                        logging.info(
                            "Raw valid ratio %.3f for bbox %s window %s cloud<=%s",
                            raw_ratio,
                            bbox,
                            variant["name"],
                            cloud_max,
                        )
                        if raw_ratio <= 0:
                            logging.warning("Raw valid ratio is zero for bbox %s", bbox)
                            continue

                        chosen_image = composite_raw
                        chosen_mask = raw_mask
                        mask_used = False
                        survival_ratio = None
                        if mask_enabled_global and collection_name == "primary":
                            masked_collection = collection.map(
                                lambda img: _apply_scl_mask(
                                    img, scl_cfg["scl_band"], keep_classes
                                )
                            )
                            if masked_collection.size().getInfo() > 0:
                                composite_masked = masked_collection.median()
                                masked_mask = _build_valid_mask(composite_masked, gee_bands, ee)
                                masked_ratio = _compute_mask_ratio(masked_mask, region, reduce_scale, reduce_tile_scale, max_pixels, ee)
                                survival_ratio = masked_ratio / max(raw_ratio, 1.0e-6)
                                logging.info(
                                    "Mask survival %.3f for bbox %s window %s cloud<=%s",
                                    survival_ratio,
                                    bbox,
                                    variant["name"],
                                    cloud_max,
                                )
                                if masked_ratio > 0 and survival_ratio >= min_survival_ratio:
                                    chosen_image = composite_masked
                                    chosen_mask = masked_mask
                                    mask_used = True

                        sentinel_candidate = None
                        patch_used = None
                        patch_region_used = None
                        for patch_size in patch_sizes:
                            for attempt in range(3):
                                patch_region = _find_patch_region(
                                    chosen_mask, region, patch_size, sample_scale, ee
                                )
                                if patch_region is None:
                                    logging.warning(
                                        "No valid patch for bbox %s patch=%s attempt=%s",
                                        bbox,
                                        patch_size,
                                        attempt + 1,
                                    )
                                    continue
                                patch_valid_ratio = _compute_mask_ratio(
                                    chosen_mask, patch_region, sample_scale, reduce_tile_scale, max_pixels, ee
                                )
                                if patch_valid_ratio < valid_fraction_min:
                                    logging.warning(
                                        "Patch is mostly clouds/masked (%.2f valid). Skipping doomed download.", 
                                        patch_valid_ratio
                                    )
                                    continue
                                try:
                                    sample = _sample_gee_image(
                                        chosen_image,
                                        gee_bands,
                                        patch_region,
                                        ee,
                                        sample_scale,
                                        default_value,
                                        reduce_scale,
                                        reduce_tile_scale,
                                        max_pixels,
                                    )
                                except RuntimeError as exc:
                                    logging.warning(
                                        "Sample failed for bbox %s patch=%s attempt=%s: %s",
                                        bbox,
                                        patch_size,
                                        attempt + 1,
                                        exc,
                                    )
                                    continue
                                sentinel_candidate = _resize_stack(sample, target_height, target_width)
                                patch_used = patch_size
                                patch_region_used = patch_region
                                break
                            if sentinel_candidate is not None:
                                break

                        if sentinel_candidate is None:
                            logging.warning("No valid samples for bbox %s after patch retries", bbox)
                            continue

                        valid_fraction = _compute_valid_fraction(sentinel_candidate)
                        valid_bands = sum(np.isfinite(band).any() for band in sentinel_candidate)
                        logging.info(
                            "Valid fraction %.3f (bands=%d) for bbox %s window %s cloud<=%s masked=%s patch=%s",
                            valid_fraction,
                            valid_bands,
                            bbox,
                            variant["name"],
                            cloud_max,
                            mask_used,
                            patch_used,
                        )
                        if valid_bands == 0 or valid_fraction < valid_fraction_min:
                            logging.warning("Valid fraction below threshold for bbox %s", bbox)
                            continue

                        sentinel = sentinel_candidate
                        sentinel_meta = {
                            "sentinel_valid_fraction": valid_fraction,
                            "sentinel_valid_bands": valid_bands,
                            "sentinel_band_stats": _compute_band_stats(sentinel_candidate),
                            "sentinel_window": variant["name"],
                            "sentinel_cloud_max": cloud_max,
                            "sentinel_masked": mask_used,
                            "sentinel_mask_survival": survival_ratio,
                            "sentinel_collection": collection_name,
                            "sentinel_raw_valid_ratio": raw_ratio,
                            "sentinel_patch_size": patch_used,
                        }
                        break
                    if sentinel is not None:
                        break
                if sentinel is not None:
                    break
            if sentinel is not None:
                break

        if sentinel is None:
            logging.error("Skipping bbox %s due to missing Sentinel-2 data after retries", bbox)
            continue

        dem = None
        dem_meta: dict[str, Any] = {}
        dem_region = patch_region_used if patch_region_used is not None else region
        for source in dem_sources:
            dem_img = load_dem_source(ee, source["source"], source["band"]).select(source["band"])
            dem_candidate = None
            for attempt in range(2):
                try:
                    dem_sample = _sample_gee_image(
                        dem_img,
                        [source["band"]],
                        dem_region,
                        ee,
                        sample_scale,
                        default_value,
                        reduce_scale,
                        reduce_tile_scale,
                        max_pixels,
                    )
                    dem_candidate = _resize_stack(dem_sample, target_height, target_width)
                    break
                except RuntimeError as exc:
                    logging.warning(
                        "DEM sample failed for bbox %s (%s) attempt %d: %s",
                        bbox,
                        source["source"],
                        attempt + 1,
                        exc,
                    )
            if dem_candidate is None:
                continue
            dem_valid = _compute_valid_fraction(dem_candidate)
            logging.info("DEM valid fraction %.3f for bbox %s source %s", dem_valid, bbox, source["source"])
            if dem_valid >= dem_valid_min:
                dem = dem_candidate
                dem_meta = {
                    "dem_source": source["source"],
                    "dem_band": source["band"],
                    "dem_valid_fraction": dem_valid,
                }
                break

        if dem is None:
            logging.error("Skipping bbox %s due to missing DEM", bbox)
            continue

        eps = config["preprocessing"]["normalization"]["epsilon"]
        ndvi_cfg = config["preprocessing"]["ndvi"]
        ndwi_cfg = config["preprocessing"]["ndwi"]
        ndsi_cfg = config["preprocessing"]["ndsi"]

        ndvi = compute_ndvi(sentinel, ndvi_cfg["nir_band_index"], ndvi_cfg["red_band_index"], eps)
        ndwi = compute_ndwi(sentinel, ndwi_cfg["green_band_index"], ndwi_cfg["nir_band_index"], eps)
        ndsi = compute_ndsi(sentinel, ndsi_cfg["green_band_index"], ndsi_cfg["swir_band_index"], eps)
        slope, _ = compute_slope_aspect(dem[0], config["scoring"]["avalanche"]["dem_spacing"], eps)

        if not _validate_tile_data(sentinel, dem, ndvi, ndwi, ndsi, slope, config, bands):
            logging.warning("Skipping bbox %s due to validation failure", bbox)
            continue

        meta = {"bbox": bbox, "valid": True, **sentinel_meta, **dem_meta}
        if gee_cfg.get("ndsi", {}).get("enabled", False):
            snow_fraction = float((ndsi >= gee_cfg["ndsi"]["snow_threshold"]).mean())
            meta["snow_fraction"] = snow_fraction
            meta["snow_present"] = snow_fraction >= gee_cfg["ndsi"].get("snow_fraction_min", 0.0)

        tile_id = bbox.get("id", f"gee_{idx}")
        tiles.append({"tile_id": tile_id, "sentinel": sentinel, "dem": dem, "meta": meta})

    if not tiles:
        logging.error("No valid tiles collected from Google Earth Engine")
        return []

    logging.info("Loaded %d tiles from Google Earth Engine", len(tiles))
    return tiles


def collect_gangotri_tiles(config: dict) -> list[dict[str, Any]]:
    provider = config["data_collection"]["provider"].lower()
    if provider == "local":
        return _collect_local(config)
    if provider == "synthetic":
        return _collect_synthetic(config)
    if provider == "gee":
        return _collect_gee(config)
    raise ValueError(f"Unsupported provider: {provider}")