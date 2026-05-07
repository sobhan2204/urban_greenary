from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
from typing import Any

import numpy as np

from utils.io_utils import ensure_dir


HIGH_SUITABILITY_THRESHOLD = 0.7
MEDIUM_SUITABILITY_THRESHOLD = 0.4
DEFAULT_VEGETATION_THRESHOLD = 0.6
DEFAULT_AVALANCHE_HIGH_THRESHOLD = 0.7


def _prepare_array(values: np.ndarray | None) -> np.ndarray | None:
    if values is None:
        return None
    return np.asarray(values, dtype=np.float32)


def _round(value: float | None, digits: int) -> float | None:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def _mean(values: np.ndarray | None, digits: int = 4) -> float | None:
    if values is None or values.size == 0:
        return None
    if not np.isfinite(values).any():
        return None
    return _round(float(np.nanmean(values)), digits)


def _max(values: np.ndarray | None, digits: int = 2) -> float | None:
    if values is None or values.size == 0:
        return None
    if not np.isfinite(values).any():
        return None
    return _round(float(np.nanmax(values)), digits)


def _percent(values: np.ndarray | None, mask: np.ndarray) -> float | None:
    if values is None or values.size == 0:
        return None
    valid = np.isfinite(values)
    if not valid.any():
        return None
    hits = mask & valid
    return _round(100.0 * float(hits.sum()) / float(valid.sum()), 2)


def _aspect_distribution(aspect: np.ndarray | None) -> dict[str, float] | None:
    if aspect is None or aspect.size == 0:
        return None
    valid = np.isfinite(aspect)
    if not valid.any():
        return None
    asp = aspect[valid]
    total = float(asp.size)
    north = ((asp >= 315) | (asp < 45)).sum() / total * 100.0
    east = ((asp >= 45) & (asp < 135)).sum() / total * 100.0
    south = ((asp >= 135) & (asp < 225)).sum() / total * 100.0
    west = ((asp >= 225) & (asp < 315)).sum() / total * 100.0
    return {
        "north": _round(north, 2) or 0.0,
        "east": _round(east, 2) or 0.0,
        "south": _round(south, 2) or 0.0,
        "west": _round(west, 2) or 0.0,
    }


def _dominant_aspect(distribution: dict[str, float] | None) -> str | None:
    if not distribution:
        return None
    return max(distribution.items(), key=lambda item: item[1])[0].capitalize()


def _build_summary(metrics: dict[str, float | None], terrain: dict[str, Any]) -> str:
    sentences: list[str] = []

    high_pct = metrics.get("high_suitability_pct")
    med_pct = metrics.get("moderate_suitability_pct")
    low_pct = metrics.get("low_suitability_pct")
    if high_pct is not None and med_pct is not None and low_pct is not None:
        sentences.append(
            f"{high_pct:.1f}% of the analyzed terrain is highly suitable for restoration, "
            f"with {med_pct:.1f}% moderately suitable and {low_pct:.1f}% unsuitable."
        )
    elif high_pct is not None:
        sentences.append(f"{high_pct:.1f}% of the analyzed terrain is highly suitable for restoration.")

    high_aval_pct = metrics.get("high_avalanche_risk_pct")
    mean_slope = metrics.get("mean_slope_deg")
    dominant_aspect = terrain.get("dominant_aspect")
    if high_aval_pct is not None:
        if high_aval_pct >= 25:
            risk_phrase = "High avalanche susceptibility is widespread"
        elif high_aval_pct >= 10:
            risk_phrase = "High avalanche susceptibility is localized"
        else:
            risk_phrase = "High avalanche susceptibility appears limited"
        if dominant_aspect:
            risk_phrase += f" on {dominant_aspect.lower()}-facing slopes"
        if mean_slope is not None:
            risk_phrase += f" with a mean slope of {mean_slope:.1f} deg"
        sentences.append(risk_phrase + ".")

    veg_pct = metrics.get("vegetation_coverage_pct")
    max_elev = metrics.get("max_elevation_m")
    if veg_pct is not None:
        if veg_pct >= 60:
            veg_desc = "high"
        elif veg_pct >= 35:
            veg_desc = "moderate"
        else:
            veg_desc = "low"
        elev_phrase = ""
        if max_elev is not None:
            if max_elev >= 4500:
                elev_phrase = " in high-elevation zones"
            elif max_elev >= 3000:
                elev_phrase = " across mid-elevation zones"
        sentences.append(f"Vegetation coverage is {veg_desc} ({veg_pct:.1f}%)" + elev_phrase + ".")

    if not sentences:
        return "Terrain summary not available for the selected tile."
    return " ".join(sentences)


def generate_report(
    vegetation_score: np.ndarray,
    avalanche_score: np.ndarray,
    combined_score: np.ndarray,
    tile_id: str,
    output_dir: Path | str | None = None,
    slope: np.ndarray | None = None,
    aspect: np.ndarray | None = None,
    elevation: np.ndarray | None = None,
    vegetation_threshold: float = DEFAULT_VEGETATION_THRESHOLD,
    avalanche_high_threshold: float = DEFAULT_AVALANCHE_HIGH_THRESHOLD,
    high_suitability_threshold: float = HIGH_SUITABILITY_THRESHOLD,
    medium_suitability_threshold: float = MEDIUM_SUITABILITY_THRESHOLD,
) -> dict[str, Any]:
    vegetation_score = _prepare_array(vegetation_score)
    avalanche_score = _prepare_array(avalanche_score)
    combined_score = _prepare_array(combined_score)
    slope = _prepare_array(slope)
    aspect = _prepare_array(aspect)
    elevation = _prepare_array(elevation)

    high_mask = combined_score > high_suitability_threshold
    med_mask = (combined_score >= medium_suitability_threshold) & (combined_score <= high_suitability_threshold)
    low_mask = combined_score < medium_suitability_threshold

    metrics = {
        "high_suitability_pct": _percent(combined_score, high_mask),
        "moderate_suitability_pct": _percent(combined_score, med_mask),
        "low_suitability_pct": _percent(combined_score, low_mask),
        "high_avalanche_risk_pct": _percent(avalanche_score, avalanche_score >= avalanche_high_threshold),
        "vegetation_coverage_pct": _percent(vegetation_score, vegetation_score >= vegetation_threshold),
        "average_suitability": _mean(combined_score, 4),
        "average_avalanche_risk": _mean(avalanche_score, 4),
        "mean_slope_deg": _mean(slope, 2),
        "max_elevation_m": _max(elevation, 1),
    }

    aspect_dist = _aspect_distribution(aspect)
    terrain_stats = {
        "mean_slope_deg": metrics["mean_slope_deg"],
        "max_slope_deg": _max(slope, 2),
        "slope_p90_deg": _round(float(np.nanpercentile(slope, 90)), 2) if slope is not None and np.isfinite(slope).any() else None,
        "mean_elevation_m": _mean(elevation, 1),
        "max_elevation_m": metrics["max_elevation_m"],
        "aspect_distribution": aspect_dist,
        "dominant_aspect": _dominant_aspect(aspect_dist),
    }

    summary = _build_summary(metrics, terrain_stats)

    report = {
        "tile_id": tile_id,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "thresholds": {
            "high_suitability": high_suitability_threshold,
            "medium_suitability": medium_suitability_threshold,
            "vegetation_coverage": vegetation_threshold,
            "high_avalanche_risk": avalanche_high_threshold,
        },
        "metrics": metrics,
        "terrain_statistics": terrain_stats,
        "summary": summary,
    }

    output_path = Path(output_dir) if output_dir else Path(__file__).resolve().parents[1] / "data" / "output" / "reports"
    ensure_dir(output_path)

    json_path = output_path / f"summary_{tile_id}.json"
    txt_path = output_path / f"summary_{tile_id}.txt"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    with txt_path.open("w", encoding="utf-8") as handle:
        handle.write(summary)

    return {
        "report": report,
        "files": {
            "json": json_path.name,
            "txt": txt_path.name,
        },
    }
