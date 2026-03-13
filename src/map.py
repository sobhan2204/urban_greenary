"""
Master pipeline – run this single file to execute every stage
and generate all maps, CSVs and visualisations.

Usage:
    python src/map.py
"""

import os
import subprocess
import sys

import cv2
import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import folium
except ImportError:
    folium = None


# ── Paths ────────────────────────────────────────────────────────────
IMAGE_PATH      = "data/delhi_satellite.tif"
DATASET_IMAGES  = "dataset/images"
DATASET_MASKS   = "dataset/masks"
MODEL_BINARY    = "models/unet_vegetation.pth"
MODEL_MULTI     = "models/unet_landcover_multiclass.pth"
PRED_BINARY     = "maps/predictions"
PRED_MULTI      = "maps/predictions_multiclass"
CSV_OUT         = "data/planting_priority_zones.csv"
IMG_DIR         = "img"
DATA_DIR        = "data"


def run(cmd: str):
    print(f"\n{'='*60}")
    print(f"▶  {cmd}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"⚠  Command exited with code {result.returncode}")
    return result.returncode


def ensure_dirs():
    for d in [DATASET_IMAGES, DATASET_MASKS, PRED_BINARY, PRED_MULTI, IMG_DIR, DATA_DIR, "models"]:
        os.makedirs(d, exist_ok=True)


# ── Stage 0 : helpers ────────────────────────────────────────────────
def show_satellite_info():
    if rasterio is None:
        print("rasterio not available — skipping satellite info")
        return
    with rasterio.open(IMAGE_PATH) as src:
        print(f"Satellite image: {IMAGE_PATH}")
        print(f"  Bands: {src.count}, Size: {src.width}×{src.height}")
        print(f"  CRS: {src.crs}")
        print(f"  Bounds: {src.bounds}")


# ── Stage 1 : NDVI visualisation ────────────────────────────────────
def generate_ndvi_visuals():
    if rasterio is None or plt is None:
        print("Skipping NDVI visuals (missing rasterio/matplotlib)")
        return

    with rasterio.open(IMAGE_PATH) as src:
        red = src.read(1).astype(np.float32)
        nir = src.read(4).astype(np.float32)

    if red.max() > 100:
        red, nir = red / 10000.0, nir / 10000.0

    ndvi = (nir - red) / (nir + red + 1e-8)
    veg_mask = ndvi > 0.3

    green_pct = np.sum(veg_mask) / veg_mask.size * 100
    print(f"\nNDVI range: {ndvi.min():.3f} .. {ndvi.max():.3f}")
    print(f"Green coverage (NDVI>0.3): {green_pct:.2f}%")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    im0 = axes[0].imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
    axes[0].set_title("NDVI Heatmap – Delhi")
    plt.colorbar(im0, ax=axes[0], label="NDVI")

    axes[1].imshow(veg_mask, cmap="Greens")
    axes[1].set_title(f"Vegetation Mask (NDVI>0.3) — {green_pct:.1f}%")

    plt.tight_layout()
    path = os.path.join(IMG_DIR, "delhi_ndvi_combined.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Stage 2 : prepare dataset (multiclass) ──────────────────────────
def prepare_dataset():
    run(
        f"{sys.executable} src/prepare_dataset.py "
        f"--mode multiclass --image {IMAGE_PATH} "
        f"--out-images {DATASET_IMAGES} --out-masks {DATASET_MASKS} "
        f"--patch-size 256 --step 256"
    )


# ── Stage 3 : train U-Net (multiclass) ──────────────────────────────
def train_model():
    run(
        f"{sys.executable} src/train_unet.py "
        f"--mode multiclass --img-dir {DATASET_IMAGES} --mask-dir {DATASET_MASKS} "
        f"--epochs 30 --batch-size 4 --num-classes 5 --lr 1e-3"
    )


# ── Stage 4 : predict ───────────────────────────────────────────────
def predict():
    run(
        f"{sys.executable} src/predict_unet.py "
        f"--input {DATASET_IMAGES} --checkpoint {MODEL_MULTI} "
        f"--output {PRED_MULTI}"
    )


# ── Stage 5 : planting zone analysis ────────────────────────────────
def planting_analysis():
    # Find the first mask in predictions
    masks = sorted([
        f for f in os.listdir(PRED_MULTI)
        if f.endswith("_mask.png")
    ])
    if not masks:
        print("No prediction masks found — skipping planting analysis")
        return

    mask_path = os.path.join(PRED_MULTI, masks[0])
    run(
        f"{sys.executable} src/planting_zone_analysis.py "
        f"--mask {mask_path} --tile-size 64 --top-k 30 "
        f"--reference-raster {IMAGE_PATH} "
        f"--csv-out {CSV_OUT} "
        f"--heatmap-out {os.path.join(IMG_DIR, 'planting_priority_heatmap.png')}"
    )


# ── Stage 6 : prediction montage ────────────────────────────────────
def show_prediction_montage():
    if plt is None:
        return

    overlays = sorted([
        f for f in os.listdir(PRED_MULTI)
        if f.endswith("_overlay.png")
    ])[:9]

    if not overlays:
        print("No overlay images found")
        return

    n = len(overlays)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
    if rows * cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, fname in enumerate(overlays):
        img = cv2.imread(os.path.join(PRED_MULTI, fname))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i].imshow(img)
        axes[i].set_title(fname.replace("_overlay.png", ""), fontsize=8)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Multiclass Predictions (overlay)", fontsize=14)
    plt.tight_layout()
    path = os.path.join(IMG_DIR, "prediction_montage.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")


# ── Stage 7 : Folium interactive map ────────────────────────────────
def generate_folium_map():
    if folium is None:
        print("folium not installed — skipping interactive map")
        return

    import csv as csv_mod

    m = folium.Map(location=[28.6, 77.15], zoom_start=12, tiles="OpenStreetMap")

    # Add planting zones from CSV
    csv_path = CSV_OUT
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                try:
                    x = float(row.get("x", ""))
                    y = float(row.get("y", ""))
                except (ValueError, TypeError):
                    continue
                score = row.get("score", "?")
                priority = row.get("priority", "?")
                color = {"high": "red", "medium": "orange", "low": "green"}.get(priority, "blue")
                folium.CircleMarker(
                    location=[y, x],
                    radius=8,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"Score: {score}<br>Priority: {priority}<br>"
                           f"Grass: {row.get('grassland_pct','')}%<br>"
                           f"Waste: {row.get('wasteland_pct','')}%",
                ).add_to(m)

    html_path = os.path.join(DATA_DIR, "delhi_map.html")
    m.save(html_path)
    print(f"Saved interactive map: {html_path}")


# ── Stage 8 : summary ───────────────────────────────────────────────
def print_summary():
    print("\n" + "=" * 60)
    print(" PIPELINE COMPLETE — Generated outputs")
    print("=" * 60)
    outputs = [
        ("NDVI + Vegetation visuals", f"{IMG_DIR}/delhi_ndvi_combined.png"),
        ("Prediction montage",        f"{IMG_DIR}/prediction_montage.png"),
        ("Planting heatmap",           f"{IMG_DIR}/planting_priority_heatmap.png"),
        ("Planting zones CSV",         CSV_OUT),
        ("Interactive map",            f"{DATA_DIR}/delhi_map.html"),
        ("U-Net model (multiclass)",   MODEL_MULTI),
    ]
    for label, path in outputs:
        status = "✓" if os.path.exists(path) else "✗"
        print(f"  {status}  {label:30s}  →  {path}")
    print()


# ── Main ─────────────────────────────────────────────────────────────
def main():
    ensure_dirs()
    show_satellite_info()

    # 1. NDVI
    generate_ndvi_visuals()

    # 2. Prepare dataset
    prepare_dataset()

    # 3. Train CNN
    train_model()

    # 4. Predict
    predict()

    # 5. Planting analysis
    planting_analysis()

    # 6. Montage
    show_prediction_montage()

    # 7. Map
    generate_folium_map()

    # 8. Summary
    print_summary()


if __name__ == "__main__":
    main()