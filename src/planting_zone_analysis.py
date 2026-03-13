import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


CLASS_INFO = {
    0: "built_up",
    1: "tree",
    2: "grassland",
    3: "water",
    4: "wasteland",
}


def area_stats(mask: np.ndarray, pixel_area_m2: float) -> Dict[str, Dict[str, float]]:
    total_pixels = mask.size
    stats = {}
    for class_id, class_name in CLASS_INFO.items():
        count = int(np.sum(mask == class_id))
        stats[class_name] = {
            "pixels": count,
            "percent": (count / total_pixels) * 100.0,
            "area_m2": count * pixel_area_m2,
            "area_km2": (count * pixel_area_m2) / 1_000_000,
        }
    return stats


def coordinate_from_pixel(reference_raster: str, row: int, col: int) -> Optional[Tuple[float, float]]:
    if rasterio is None:
        return None
    with rasterio.open(reference_raster) as src:
        x, y = src.transform * (col + 0.5, row + 0.5)
    return x, y


def planting_score(tile_counts: Dict[int, int], tile_total: int) -> Tuple[float, Dict[str, float]]:
    proportions = {
        "built_up": tile_counts.get(0, 0) / tile_total,
        "tree": tile_counts.get(1, 0) / tile_total,
        "grassland": tile_counts.get(2, 0) / tile_total,
        "water": tile_counts.get(3, 0) / tile_total,
        "wasteland": tile_counts.get(4, 0) / tile_total,
    }

    score = (
        0.50 * proportions["grassland"]
        + 0.40 * proportions["wasteland"]
        - 0.30 * proportions["tree"]
        - 0.20 * proportions["water"]
        - 0.10 * proportions["built_up"]
    )
    return score, proportions


def priority_label(score: float) -> str:
    if score >= 0.25:
        return "high"
    if score >= 0.10:
        return "medium"
    return "low"


def find_priority_tiles(mask: np.ndarray, tile_size: int) -> Tuple[List[dict], np.ndarray]:
    h, w = mask.shape
    heatmap = np.full((h, w), np.nan, dtype=np.float32)
    candidates = []

    for top in range(0, h - tile_size + 1, tile_size):
        for left in range(0, w - tile_size + 1, tile_size):
            tile = mask[top : top + tile_size, left : left + tile_size]
            values, counts = np.unique(tile, return_counts=True)
            tile_counts = {int(v): int(c) for v, c in zip(values, counts)}
            tile_total = tile.size

            score, p = planting_score(tile_counts, tile_total)
            heatmap[top : top + tile_size, left : left + tile_size] = score

            # Relaxed filters: any tile with some open land and positive score
            is_candidate = (
                (p["grassland"] + p["wasteland"] >= 0.15)
                and (p["tree"] <= 0.50)
                and (p["water"] <= 0.40)
                and (score >= 0.05)
            )

            if is_candidate:
                candidates.append(
                    {
                        "row": top,
                        "col": left,
                        "score": round(score, 4),
                        "priority": priority_label(score),
                        "tree_pct": round(p["tree"] * 100.0, 2),
                        "grassland_pct": round(p["grassland"] * 100.0, 2),
                        "wasteland_pct": round(p["wasteland"] * 100.0, 2),
                        "built_up_pct": round(p["built_up"] * 100.0, 2),
                        "water_pct": round(p["water"] * 100.0, 2),
                    }
                )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates, heatmap


def save_csv(path: str, rows: List[dict]):
    if not rows:
        print("WARNING: No planting zone candidates found — writing header-only CSV.")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["row", "col", "score", "priority", "tree_pct",
                 "grassland_pct", "wasteland_pct", "built_up_pct", "water_pct", "x", "y"]
            )
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Convert multiclass masks into actionable planting-zone recommendations")
    parser.add_argument("--mask", default="maps/predictions_multiclass/img_0_mask.png", help="Predicted multiclass mask image")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--pixel-area-m2", type=float, default=100.0)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--reference-raster", default="", help="Optional georeferenced raster")
    parser.add_argument("--csv-out", default="data/planting_priority_zones.csv")
    parser.add_argument("--heatmap-out", default="img/planting_priority_heatmap.png")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.csv_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.heatmap_out), exist_ok=True)

    mask = cv2.imread(args.mask, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found: {args.mask}")

    print(f"Mask shape: {mask.shape}, unique values: {np.unique(mask).tolist()}")

    stats = area_stats(mask, args.pixel_area_m2)
    print("\nArea statistics from predicted mask:")
    for cls_name, cls_stats in stats.items():
        print(f"  {cls_name}: {cls_stats['percent']:.2f}% | {cls_stats['area_km2']:.4f} km²")

    candidates, heatmap = find_priority_tiles(mask, args.tile_size)
    print(f"\nTotal candidate tiles found: {len(candidates)}")
    top_rows = candidates[: args.top_k]

    if args.reference_raster and rasterio is not None:
        for row in top_rows:
            centroid_row = int(row["row"] + args.tile_size // 2)
            centroid_col = int(row["col"] + args.tile_size // 2)
            coord = coordinate_from_pixel(args.reference_raster, centroid_row, centroid_col)
            if coord is not None:
                row["x"] = coord[0]
                row["y"] = coord[1]
            else:
                row["x"] = ""
                row["y"] = ""
    else:
        for row in top_rows:
            row["x"] = ""
            row["y"] = ""

    save_csv(args.csv_out, top_rows)
    print(f"Saved top {len(top_rows)} planting zones to: {args.csv_out}")

    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        valid = heatmap[~np.isnan(heatmap)]
        if valid.size > 0 and valid.max() > valid.min():
            im = ax.imshow(heatmap, cmap="RdYlGn", vmin=valid.min(), vmax=valid.max())
        else:
            im = ax.imshow(heatmap, cmap="RdYlGn")
        plt.colorbar(im, label="Planting suitability score")
        ax.set_title("Tree Planting Priority Heatmap")
        plt.tight_layout()
        plt.savefig(args.heatmap_out, dpi=150)
        print(f"Saved planting heatmap to: {args.heatmap_out}")
    else:
        print("matplotlib not installed; skipped heatmap image.")


if __name__ == "__main__":
    main()