from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import ee
import numpy as np

from utils.config_loader import load_config, resolve_config_path


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stage 2: Reverse geocoding
# ---------------------------------------------------------------------------

def reverse_geocode(lat: float, lng: float) -> dict[str, str]:
    """Reverse-geocode coordinates via Nominatim.

    Returns a dict with keys: location_name, city, country, full_address.
    Each falls back to "Unknown" when the API provides no value.
    """
    try:
        from geopy.geocoders import Nominatim
    except ImportError as exc:
        raise ImportError("geopy is required for reverse geocoding") from exc

    geolocator = Nominatim(user_agent="himalayan-veg-restoration")
    try:
        location = geolocator.reverse((lat, lng))
    except Exception as exc:
        logger.warning("Reverse geocoding failed: %s", exc)
        return {
            "location_name": "Unknown",
            "city": "Unknown",
            "country": "Unknown",
            "full_address": "Unknown",
        }

    addr = location.raw.get("address", {}) if location else {}
    return {
        "location_name": location.address.split(",")[0] if location.address else "Unknown",
        "city": addr.get("city", addr.get("town", addr.get("village", "Unknown"))),
        "country": addr.get("country", "Unknown"),
        "full_address": location.address if location.address else "Unknown",
    }


# ---------------------------------------------------------------------------
# Stage 3: GEE image retrieval
# ---------------------------------------------------------------------------

def _normalize_band(band: str) -> str:
    """Normalize band name: B02 -> B2, etc."""
    if band.startswith("B0") and band[2:].isdigit():
        return "B" + band[2:].lstrip("0")
    return band


_SENTINEL2_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12"]
_SENTINEL2_RGB   = ["B4", "B3", "B2"]
_SENTINEL2_VIZ   = {"min": 0, "max": 3000}

_LANDSAT8_BANDS  = ["B2", "B3", "B4", "B5", "B6", "B7"]
_LANDSAT8_RGB    = ["B4", "B3", "B2"]
_LANDSAT8_VIZ    = {"min": 0, "max": 15000}


def _date_range(target: str) -> tuple[str, str]:
    """Return (start, end) ISO dates for a +/-15 day window."""
    center = datetime.strptime(target, "%Y-%m-%d")
    start = center - timedelta(days=15)
    end   = center + timedelta(days=15)
    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


def _mask_s2_clouds(image: ee.Image) -> ee.Image:
    """Apply SCL cloud-free mask to a Sentinel-2 image."""
    scl = image.select("SCL")
    # Keep: 4=vegetation, 5=trees, 6=bare soil, 7=built-up, 11=snow
    mask = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7)).Or(scl.eq(11))
    return image.updateMask(mask)


def _find_sentinel2(
    aoi: ee.Geometry,
    point: ee.Geometry,
    start_date: str,
    end_date: str,
    max_cloud: float = 20.0,
) -> ee.Image | None:
    """Query Sentinel-2 SR, return the least cloudy RAW image (no SCL mask).

    Tries multiple images until one with actual data coverage at the AOI
    centre is found — tiles near scene boundaries can have zero-coverage
    in parts of the sampling rectangle even though they overlap the AOI.

    The caller should apply _mask_s2_clouds for display separately.
    """
    col = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUDY_PIXEL_PERCENTAGE", max_cloud))
    )

    count = col.size().getInfo()
    if count == 0:
        return None

    # Sort by cloud percentage and try each image until we find one
    # with non-zero reflectance at the centre of the AOI.
    sorted_col = col.sort("CLOUDY_PIXEL_PERCENTAGE")
    ids = sorted_col.aggregate_array("system:id").getInfo()

    for img_id in ids:
        img = ee.Image(img_id)
        # Quick coverage check: mean B8 at point should be > 0
        mean_val = img.select("B8").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=10,
            maxPixels=100,
        ).get("B8").getInfo()
        if mean_val is not None and mean_val > 0:
            return img

    # Fallback: return the first image anyway
    return sorted_col.first()


def _find_landsat8(
    aoi: ee.Geometry,
    point: ee.Geometry,
    start_date: str,
    end_date: str,
    max_cloud: float = 20.0,
) -> ee.Image | None:
    """Query Landsat 8 TOA, return the least cloudy RAW image (no QA mask).

    Tries multiple images until one with actual data coverage at the AOI
    centre is found.
    """
    col = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .filter(ee.Filter.lte("CLOUD_COVER", max_cloud))
    )

    count = col.size().getInfo()
    if count == 0:
        return None

    sorted_col = col.sort("CLOUD_COVER")
    ids = sorted_col.aggregate_array("system:id").getInfo()

    for img_id in ids:
        img = ee.Image(img_id)
        mean_val = img.select("B5").reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=30,
            maxPixels=100,
        ).get("B5").getInfo()
        if mean_val is not None and mean_val > 0:
            return img

    return sorted_col.first()


def fetch_satellite_image(
    lat: float,
    lng: float,
    target_date: str,
    cloud_cover: float = 20.0,
    buffer_m: float = 5000.0,
) -> dict[str, Any]:
    """Retrieve the best satellite image for the given location and date.

    Returns
    -------
    dict with keys:
        image_raw   : ee.Image — raw unmasked, unclipped (for sampling)
        image_display: ee.Image — cloud-masked + clipped (for map layer)
        source      : "Sentinel-2" or "Landsat-8"
        bands       : list of band name strings
        rgb_bands   : list of 3 RGB band names
        viz_params  : {"min": ..., "max": ...}
        image_date  : ISO date string of the selected image
        aoi         : ee.Geometry (buffered point bounds)
        point       : ee.Geometry.Point
    """
    point = ee.Geometry.Point([lng, lat])
    buffered = point.buffer(buffer_m)
    aoi = buffered.bounds()
    start_date, end_date = _date_range(target_date)

    # Try Sentinel-2 first (raw, unmasked)
    s2_raw = _find_sentinel2(aoi, point, start_date, end_date, cloud_cover)
    if s2_raw is not None:
        img_date = s2_raw.get("system:time_start")
        img_date_str = ee.Date(img_date).format("yyyy-MM-dd").getInfo()
        s2_display = _mask_s2_clouds(s2_raw).clip(aoi)
        return {
            "image_raw": s2_raw,
            "image_display": s2_display,
            "source": "Sentinel-2",
            "bands": _SENTINEL2_BANDS,
            "rgb_bands": _SENTINEL2_RGB,
            "viz_params": _SENTINEL2_VIZ,
            "image_date": img_date_str,
            "aoi": aoi,
            "point": point,
        }

    # Fallback to Landsat-8 (raw, unmasked)
    l8_raw = _find_landsat8(aoi, point, start_date, end_date, cloud_cover)
    if l8_raw is not None:
        img_date = l8_raw.get("SYSTEM:TIME_START")
        img_date_str = ee.Date(img_date).format("yyyy-MM-dd").getInfo()
        l8_display = l8_raw.clip(aoi)
        return {
            "image_raw": l8_raw,
            "image_display": l8_display,
            "source": "Landsat-8",
            "bands": _LANDSAT8_BANDS,
            "rgb_bands": _LANDSAT8_RGB,
            "viz_params": _LANDSAT8_VIZ,
            "image_date": img_date_str,
            "aoi": aoi,
            "point": point,
        }

    raise RuntimeError(
        f"No satellite imagery found near ({lat}, {lng}) "
        f"for date range {start_date} to {end_date} with cloud cover < {cloud_cover}%"
    )


# ---------------------------------------------------------------------------
# Stage 4: Band extraction and ML pipeline interface
# ---------------------------------------------------------------------------

def extract_band_values(result: dict[str, Any]) -> dict[str, float]:
    """Extract per-band reflectance values at the clicked point.

    Parameters
    ----------
    result : dict returned by fetch_satellite_image()

    Returns
    -------
    dict mapping band name -> float reflectance value at the point.
    """
    image: ee.Image = result["image_raw"]
    point: ee.Geometry = result["point"]
    bands: list[str] = result["bands"]
    source: str = result["source"]

    scale = 10 if source == "Sentinel-2" else 30

    sample = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=scale,
        bestEffort=True,
    )

    values: dict[str, float] = {}
    for b in bands:
        val = sample.get(b)
        if val is not None:
            values[b] = float(val.getInfo())
        else:
            values[b] = 0.0

    return values


def _sample_image_rect(image: ee.Image, region: ee.Geometry, target_h: int, target_w: int) -> np.ndarray:
    """Sample an image over a rectangular region and resize to target shape.

    GEE sampleRectangle returns arrays whose dimensions correspond to
    (pixels-along-x, pixels-along-y). For north-aligned images this is
    effectively (width, height).  When the region aligns with the image
    grid the shape is approximately (target_w, target_h).

    This function pads or crops each dimension independently so the result
    is always exactly (target_h, target_w), avoiding the old square-crop
    bug that turned half the image black when dimensions didn't match.
    """
    bands = image.bandNames().getInfo()
    if not bands:
        return np.zeros((0, target_h, target_w), dtype=np.float32)

    sample_info = image.sampleRectangle(region=region, defaultValue=0).getInfo()
    props = sample_info.get("properties", {})
    stack: list[np.ndarray] = []
    for b in bands:
        raw = np.array(props.get(b, []), dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        d0, d1 = raw.shape
        # GEE returns (width, height).  If the two dimensions differ and
        # d0 is closer to target_w we need a transpose; otherwise no-op.
        if d0 != d1:
            dist_no  = abs(d0 - target_h) + abs(d1 - target_w)
            dist_swap = abs(d1 - target_h) + abs(d0 - target_w)
            if dist_swap < dist_no:
                raw = raw.T
        h, w = raw.shape

        # Pad to exact target size if smaller; crop if larger
        patch = np.zeros((target_h, target_w), dtype=np.float32)
        copy_h, copy_w = min(h, target_h), min(w, target_w)
        patch[:copy_h, :copy_w] = raw[:copy_h, :copy_w]
        stack.append(patch)
    return np.stack(stack, axis=0).astype(np.float32)


def _get_dem_at_point(point: ee.Geometry, rect: ee.Geometry, region_size: int = 256) -> np.ndarray:
    """Sample DEM over a square region centred on *point*."""
    sources = [
        ("USGS/SRTMGL1_003", "elevation"),
        ("COPERNICUS/DEM/GLO30", "DEM"),
    ]
    for src_id, band_name in sources:
        try:
            dem = ee.Image(src_id).select([band_name])
            arr = _sample_image_rect(dem, rect, region_size, region_size)
            if arr.size > 0 and np.isfinite(arr).any():
                return arr[0].reshape(1, region_size, region_size)
        except Exception as exc:
            logger.warning("DEM source %s failed: %s", src_id, exc)
            continue
    raise RuntimeError("Could not sample DEM from any source")


def build_tile(
    result: dict[str, Any],
    config: dict | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Build a tile dict compatible with the ML pipeline from GEE results.

    Fetches a 256x256 patch of Sentinel bands + DEM centred on the clicked point.

    Parameters
    ----------
    result : dict from fetch_satellite_image()
    config : loaded config dict (optional, auto-loaded if None)
    config_path : override path to config.yaml

    Returns
    -------
    dict with keys: tile_id, sentinel (C, H, W), dem (1, H, W), meta
    """
    if config is None:
        config = load_config(resolve_config_path(config_path))

    image: ee.Image = result["image_raw"]
    point: ee.Geometry = result["point"]
    bands: list[str] = result["bands"]
    source: str = result["source"]
    target_size = config["data_collection"]["tile_size"]["height"]

    scale = 10 if source == "Sentinel-2" else 30
    deg_per_px = scale / 111320.0
    half_deg = (target_size // 2) * deg_per_px

    coords = point.coordinates().getInfo()
    lng, lat = coords[0], coords[1]
    rect = ee.Geometry.Rectangle(
        [lng - half_deg, lat - half_deg, lng + half_deg, lat + half_deg]
    )

    # Use the bands defined in config for the ML pipeline tile
    # (config has 5 bands; extract display has 6 including B12)
    pipeline_bands = [_normalize_band(b) for b in config["data_collection"]["sentinel"]["bands"]]
    pipeline_channels = config["data_collection"]["sentinel"]["num_channels"]

    # Sample satellite bands — no reproject(), sample at native resolution
    # and resize to target. This matches the proven pattern from data_collection.py.
    sat_stack = _sample_image_rect(image.select(bands), rect, target_size, target_size)

    sentinel = sat_stack[:pipeline_channels].astype(np.float32)

    # Sample DEM
    try:
        dem = _get_dem_at_point(point, rect, target_size)
    except RuntimeError:
        dem = np.zeros((1, target_size, target_size), dtype=np.float32)

    return {
        "tile_id": f"interactive_{lat:.4f}_{lng:.4f}",
        "sentinel": sentinel,
        "dem": dem,
        "meta": {
            "source": source,
            "bands": bands,
            "lat": lat,
            "lng": lng,
            "image_date": result.get("image_date", "Unknown"),
        },
    }


def run_ml_pipeline(
    tile: dict[str, Any],
    config: dict | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    """Run the full ML inference pipeline on a single tile.

    Loads the trained vegetation model, predicts class probabilities,
    computes vegetation and avalanche scores, and returns results.

    Parameters
    ----------
    tile : dict from build_tile()
    config : loaded config dict (optional)
    config_path : override path to config.yaml

    Returns
    -------
    dict with keys: probabilities, vegetation_score, avalanche_score,
                    combined_score, tile_id
    """
    if config is None:
        config = load_config(resolve_config_path(config_path))

    import torch
    from vegetation.data_loader import prepare_tile
    from vegetation.inference import predict_probabilities
    from vegetation.model import build_segmenter
    from utils.io_utils import load_model
    from utils.config_loader import resolve_path
    from utils.geo_utils import compute_slope_aspect, compute_curvature
    from avalanche.terrain_features import compute_terrain_features
    from avalanche.scoring import compute_avalanche_score

    tile_id = tile["tile_id"]

    # Prepare input tensor
    sample = prepare_tile(tile, config)
    input_tensor = sample["input"]  # (C, H, W)

    # Load model
    tile_size = config["data_collection"]["tile_size"]
    in_channels = (
        config["data_collection"]["sentinel"]["num_channels"]
        + config["data_collection"]["dem"]["num_channels"]
    )
    model = build_segmenter(config, (tile_size["height"], tile_size["width"]), in_channels)
    model_path = resolve_path(config, "paths", "models", "vegetation_model")
    state = load_model(model_path, map_location=config["inference"]["device"])
    model.encoder.load_state_dict(state["encoder_state"])
    model.decoder.load_state_dict(state["decoder_state"])
    model.eval()

    # Predict
    inp = torch.from_numpy(input_tensor).unsqueeze(0).float()
    probs = predict_probabilities(model, inp)  # (1, num_classes, H, W)
    probs_mean = probs[0].mean(axis=(1, 2))  # per-class mean probability

    # Vegetation score (class index 2)
    veg_class_idx = config["scoring"]["vegetation"]["class_indices"][0]
    veg_score = float(probs_mean[veg_class_idx])

    # Avalanche score from terrain features
    terrain = compute_terrain_features(tile["dem"], config)
    aval_map = compute_avalanche_score(terrain, config)
    aval_score = float(np.nanmean(aval_map)) if aval_map.size > 0 else 0.0

    # Combined: veg * (1 - avalanche)
    combined = veg_score * (1.0 - aval_score)

    return {
        "tile_id": tile_id,
        "probabilities": probs_mean,
        "vegetation_score": veg_score,
        "avalanche_score": aval_score,
        "combined_score": combined,
    }
