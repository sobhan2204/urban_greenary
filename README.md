# Himalayan Snow & Avalanche Susceptibility Mapping for Vegetation Restoration

Maps snow cover, avalanche susceptibility, and vegetation restoration suitability for the **Gangotri** region using **Sentinel-2** multispectral imagery and **DEM** terrain data. The core ML pipeline uses a **MultiSSL** transformer encoder pretraining phase followed by a **U-KAN** decoder for pixel-wise segmentation. A separate rule-based avalanche module scores terrain risk. Both scores combine into a final restoration suitability map.

## Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              START (main.py)                                │
│                        Load config / config/config.yaml                     │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     Data Collection (data_collection)                        │
│                                                                             │
│   ┌──────────────────┐   ┌──────────────────┐   ┌───────────────────────┐  │
│   │  Google Earth Eng │   │   Local Rasters  │   │   Synthetic Data      │  │
│   │  (Sentinel-2 +   │   │   (manifest CSV/  │   │   (random tiles for   │  │
│   │   SRTM/DEM)       │   │    JSON + TIFF)  │   │    debugging)         │  │
│   └──────────────────┘   └──────────────────┘   └───────────────────────┘  │
│                              │                                               │
│                              ▼                                               │
│                 Tiles: {sentinel, dem, bbox, meta}                           │
└─────────────────────────────┬───────────────────────────────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
┌───────────────────────────┐   ┌─────────────────────────────────────────────┐
│   VEGETATION PIPELINE     │   │          AVALANCHE PIPELINE                  │
│   (vegetation/pipeline)   │   │          (avalanche/pipeline)                │
│                           │   │                                             │
│  ┌─────────────────────┐  │   │  ┌────────────────────────────────────────┐ │
│  │ Load trained model  │  │   │  │ Compute Terrain Features from DEM     │ │
│  │ (encoder + decoder) │  │   │  │  - Slope (slope/aspect from DEM)     │ │
│  └──────────┬──────────┘  │   │  │  - Curvature (2nd derivative)        │ │
│             │              │   │  │  - Elevation                          │ │
│             ▼              │   │  └─────────────────┬────────────────────┘ │
│  ┌─────────────────────┐  │   │                    │                       │
│  │ Predict pixel-class │  │   │                    ▼                       │
│  │ probabilities       │  │   │  ┌────────────────────────────────────────┐ │
│  └──────────┬──────────┘  │   │  │ Avalanche Score (weighted sum)        │ │
│             │              │   │  │  0.4×slope + 0.3×aspect               │ │
│             ▼              │   │  │  +0.2×curvature +0.1×elevation        │ │
│  ┌─────────────────────┐  │   │  └─────────────────┬────────────────────┘ │
│  │ Vegetation Score    │  │   │                    │                       │
│  │ (mean of vegetation │  │   │                    ▼                       │
│  │  class probs,       │  │   │  ┌────────────────────────────────────────┐ │
│  │  masked by elevation│  │   │  │ Save Avalanche Overlay Map            │ │
│  └──────────┬──────────┘  │   │  │ (data/output/final/avalanche_map.png) │ │
│             │              │   │  └────────────────────────────────────────┘ │
│             ▼              │   │                                             │
│  ┌─────────────────────┐  │   │                                             │
│  │ Save Vegetation     │  │   │                                             │
│  │ Overlay Map          │  │   │                                             │
│  │ (data/output/final/  │  │   │                                             │
│  │  vegetation_map.png) │  │   │                                             │
│  └─────────────────────┘  │   │                                             │
└─────────────┬─────────────┘   └───────────────┬─────────────────────────────┘
              │                                  │
              ▼                                  ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Combine Scores (main.py)                              │
│                                                                             │
│              Combined = Vegetation_Score × (1 − Avalanche_Score)            │
│                                                                             │
│              Per-tile: overlay combined mask on RGB composite                │
│              Save → data/output/final/combined_map_<tile_id>.png            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Training flow (separate from inference)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Training (vegetation/train.py)                        │
│                                                                             │
│  1. Collect tiles  ──────────────────────────────────────────────────────►  │
│                                                                             │
│  2. MultiSSL Pretraining                                                    │
│     ┌──────────────────────────────────────────────────────┐                │
│     │ Masked Autoencoder on raw tiles                      │                │
│     │ Randomly mask 35% of patches → transformer predicts  │                │
│     │ the original pixel values (MSE loss)                 │                │
│     └──────────────────────┬───────────────────────────────┘                │
│                            │                                                │
│                            ▼                                                │
│                  Frozen encoder weights                                     │
│                                                                             │
│  3. U-KAN Finetuning                                                       │
│     ┌──────────────────────────────────────────────────────┐                │
│     │ Frozen MultiSSL encoder + trainable U-KAN decoder   │                │
│     │ Pixel-wise segmentation (5 classes)                  │                │
│     │ Cross-entropy loss on pseudo-labels (NDVI/NDSI/NDWI) │                │
│     └──────────────────────┬───────────────────────────────┘                │
│                            │                                                │
│                            ▼                                                │
│                  Save model → models/saved_models/vegetation_model.pkl      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Google Earth Engine account (for `gee` data provider)

### Setup

    pip install -r requirements.txt

### Configure

Edit `config/config.yaml` to set:

- **Bounding boxes** — target areas under `data_collection.bounding_boxes`
- **Date windows** — Sentinel-2 acquisition periods under `data_collection.gee.date_windows`
- **GEE project** — your project ID under `data_collection.gee.project_id`
- **Device** — `cpu` or `cuda` under `model.training.device`
- **Thresholds** — scoring/overlay cutoffs under `scoring` and `visualization`

### Train

    python vegetation/train.py

Two phases run sequentially:
1. **MultiSSL pretraining** — masked patch reconstruction on raw tiles (MSE loss)
2. **U-KAN finetuning** — segmentation with frozen encoder (cross-entropy loss)

The trained model saves to `models/saved_models/vegetation_model.pkl`.

### Run inference

    python main.py

This runs the full inference pipeline — data collection → vegetation scoring → avalanche scoring → combined map generation.

## Project Structure

```
├── config/
│   └── config.yaml              # All paths, hyperparameters, thresholds
├── data/
│   ├── raw/                     # Raw downloaded tiles
│   ├── processed/               # Preprocessed arrays
│   └── output/
│       └── final/               # Final overlay maps
├── models/
│   └── saved_models/            # Trained model checkpoints
├── vegetation/
│   ├── data_loader.py           # Dataset classes (PretrainDataset, SegmentationDataset)
│   ├── labeling.py              # Pseudo-label generation from NDVI/NDSI/NDWI indices
│   ├── model.py                 # MultiSSL encoder, U-KAN decoder, pretrainer/segmenter
│   ├── preprocessing.py         # Normalization, NDVI/NDSI/NDWI computation, stacking
│   ├── train.py                 # Training entry point (pretrain + finetune)
│   ├── inference.py             # Model loading, prediction
│   ├── scoring.py               # Vegetation suitability scoring
│   ├── visualization.py         # Vegetation overlay map saving
│   └── pipeline.py              # Orchestrates vegetation inference
├── avalanche/
│   ├── data_loader.py           # Tile loading for avalanche module
│   ├── terrain_features.py      # Slope, aspect, curvature extraction from DEM
│   ├── scoring.py               # Weighted avalanche risk scoring
│   ├── visualization.py         # Avalanche overlay map saving
│   └── pipeline.py              # Orchestrates avalanche inference
├── utils/
│   ├── config_loader.py         # YAML loading, path resolution
│   ├── data_collection.py       # GEE / local / synthetic tile collection
│   ├── geo_utils.py             # Slope, aspect, curvature computation
│   ├── io_utils.py              # Logging setup, model save/load helpers
│   └── visualization_utils.py   # RGB composites, mask overlay, image saving
├── scripts/
│   └── gee_stress_test.py       # GEE connection/data retrieval stress test
├── google_earth.py              # GEE project authentication check
├── main.py                      # Main inference entry point
├── requirements.txt
└── README.md
```

## Segmentation Classes

| Class      | Index | Labeling heuristic                        |
|------------|-------|-------------------------------------------|
| Background | 0     | Default fallback                          |
| Snow       | 1     | NDSI ≥ 0.4                                |
| Vegetation | 2     | NDVI ≥ 0.2                                |
| Water      | 3     | NDVI ≤ 0.0                                |
| Barren     | 4     | NDVI ≤ 0.1 AND NDSI ≤ 0.2                 |

## Scoring

### Vegetation score

Mean probability of the vegetation class across the tile, zeroed out above the elevation limit (default 3500 m). Normalized to `[0, 1]`.

### Avalanche score

Weighted combination of normalized terrain features:

| Feature    | Weight | Range              |
|------------|--------|--------------------|
| Slope      | 0.4    | 25°–45°           |
| Aspect     | 0.3    | NW-facing (315–45°) |
| Curvature  | 0.2    | -0.2 to 0.2       |
| Elevation  | 0.1    | 2000–5500 m       |

### Combined score

```
Combined = Vegetation_Score × (1 − Avalanche_Score)
```

A higher combined score means the area has good vegetation potential and low avalanche risk — ideal for restoration.

## Outputs

All maps are saved to `data/output/final/`:

| File                              | Format | Description                        |
|-----------------------------------|--------|------------------------------------|
| `vegetation_map_<tile_id>.png`    | PNG    | Vegetation suitability overlay     |
| `avalanche_map_<tile_id>.png`     | PNG    | Avalanche-safe areas overlay       |
| `combined_map_<tile_id>.png`      | PNG    | Combined restoration suitability   |

## Dependencies

| Package          | Purpose                              |
|------------------|--------------------------------------|
| torch / torchvision | Deep learning framework           |
| numpy / pandas    | Numerical computation, data handling |
| rasterio          | GeoTIFF I/O                          |
| geopandas         | Spatial data handling                |
| earthengine-api   | Google Earth Engine integration      |
| scikit-learn      | Preprocessing utilities              |
| matplotlib        | Map visualization                    |
| folium            | Interactive map (optional)           |
| pyyaml            | Config parsing                       |

## Notes

- **Fully config-driven** — no hardcoded paths, thresholds, or hyperparameters in code. Everything flows through `config/config.yaml`.
- **Training and inference are separate** — train once with `vegetation/train.py`, then run inference any number of times with `main.py`.
- **Data providers** — switch between `gee`, `local`, and `synthetic` by changing `data_collection.provider` in config.
