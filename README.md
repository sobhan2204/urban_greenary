# Himalayan Snow & Avalanche Susceptibility Mapping for Vegetation Restoration

This project builds map-based suitability outputs for mountain restoration planning. It combines Sentinel-2 multispectral imagery with DEM terrain data to estimate where vegetation can thrive and where avalanche risk is low enough to support restoration work.

The repository has two major analysis branches:

1. A vegetation pipeline that loads a trained deep learning model, predicts pixel-wise land-cover probabilities, and converts them into a vegetation suitability score.
2. An avalanche pipeline that derives terrain features from elevation data and computes a rule-based avalanche risk score.

The final output is a combined restoration map created from both signals.

## What This Project Does

The project is designed for the Himalayan region and focuses on:

- collecting geospatial tiles from Google Earth Engine or local data sources
- training a vegetation segmentation model with a MultiSSL pretraining stage and a U-KAN decoder
- computing avalanche susceptibility from slope, aspect, curvature, and elevation
- generating visualization overlays for vegetation, avalanche safety, and combined restoration suitability
- serving the workflow through a Flask app for interactive coordinate-based runs

## How It Works

### Training

The training entry point is `vegetation/train.py`.

It runs in two phases:

1. MultiSSL pretraining learns image representations with masked patch reconstruction.
2. U-KAN finetuning uses the pretrained encoder and trains a segmentation decoder with pseudo-labels derived from vegetation indices such as NDVI, NDSI, and NDWI.

The trained model is saved to `models/saved_models/vegetation_model.pkl`.

### Inference

The main batch inference entry point is `main.py`.

At runtime it:

1. loads configuration from `config/config.yaml`
2. collects tiles for the configured bounding boxes
3. runs the vegetation pipeline
4. runs the avalanche pipeline
5. combines the scores into a final suitability map

For interactive use, `app.py` exposes a Flask app with API routes that can trigger the pipeline for a clicked coordinate.

### Scoring Logic

- Vegetation score: derived from the predicted probability of the vegetation class, with elevation-based masking.
- Avalanche score: derived from a weighted terrain model using slope, aspect, curvature, and elevation.
- Combined score: calculated as vegetation suitability multiplied by $(1 - \text{avalanche risk})$.

Higher combined values indicate areas that are better candidates for restoration.

## Technical Details

### Model Overview (MultiSSL + U-KAN)

The vegetation pipeline uses a two-stage model:

1. MultiSSL pretraining to learn general Earth-image features from Sentinel-2 tiles.
2. U-KAN segmentation finetuning to produce pixel-wise terrain probabilities.

End-to-end flow:

```text
Sentinel-2 tile (256 x 256 x 5)
	|
	v
MultiSSL encoder learns features (masked patch reconstruction)
	|
Pretrained encoder weights
	|
	v
U-KAN segmentation network
	|
Pixel-wise terrain understanding
	|
Vegetation probability map
	|
Vegetation suitability score
```

### MultiSSL Pretraining (Masked Patch Reconstruction)

MultiSSL hides random patches in the input tile and trains the encoder to reconstruct them. This forces the encoder to learn patterns rather than memorize pixels. During reconstruction, the encoder learns features that are useful for terrain understanding:

| Learned Feature     | Why It Matters          |
| ------------------- | ----------------------- |
| Snow texture        | Avalanche relevance     |
| Vegetation patterns | Restoration suitability |
| Glacier edges       | Terrain classification  |
| Spectral signatures | Surface type separation |
| Terrain boundaries  | Segmentation accuracy   |

After pretraining, the encoder weights are saved and reused in the segmentation stage.

### U-KAN Segmentation (U-Net + KAN)

Goal: predict a terrain class for every pixel.

- Input: 256 x 256 x 5 tile
- Output: 256 x 256 segmentation mask with per-pixel probabilities
- Each pixel receives probabilities for vegetation, snow, barren land, and other classes.

Why KAN:

- Terrain relationships are non-linear across spectral bands and textures.
- KAN models non-linear interactions better than linear feature assumptions in classic CNNs.
- U-Net provides spatial understanding, while KAN captures complex spectral and environmental dependencies.

### Training Loop and Pseudo-Labels

1. Input tile enters the network.
2. Prediction is generated.
3. Prediction is compared with pseudo-labels.
4. Loss computes the error.
5. Backpropagation updates weights over many epochs.

Pseudo-labels are derived from vegetation indices such as NDVI, NDSI, and NDWI.

Over time, the model learns terrain semantics, vegetation structures, snow boundaries, and spectral relationships.

### Pixel-wise Interpretation

Example semantic mappings learned by the model:

- High NIR + texture pattern -> vegetation (alpine vegetation signatures)
- High visible reflectance + cold terrain texture -> snow or glacier edges

Typical spectral behaviors:

- Vegetation: high NIR reflectance, moderate green, specific texture
- Snow: bright visible bands, unique SWIR behavior
- Water: absorbs NIR, dark spectral response
- Rocky terrain: irregular textures, mixed reflectance

The model does not explicitly know if a location is a mountain, plain, or water body. It learns mathematical patterns associated with terrain types.

Example vegetation probability output:

| Pixel | Vegetation Probability |
| ----- | ---------------------- |
| A     | 0.95                   |
| B     | 0.10                   |
| C     | 0.70                   |

### Avalanche and Aspect Interpretation

- North (315-360, 0-45): colder, snow-accumulating slopes
- East (45-135): morning sun, moderate conditions
- South (135-225): warmer, sun-exposed slopes
- West (225-315): afternoon sun, warm and sometimes dry

Implications:

- Avalanche risk: north-facing slopes retain snow longer and have higher avalanche susceptibility.
- Vegetation growth: south-facing slopes receive more sun and dry faster; north-facing slopes are cooler and wetter.
- Restoration planning: aspect guides which slopes are more suitable for planting strategies.

## Tech Stack

### Core Language and Runtime

- Python 3.10+

### Machine Learning

- PyTorch
- Torchvision
- scikit-learn

### Geospatial and Remote Sensing

- Google Earth Engine API
- geemap
- rasterio
- geopandas
- geopy
- ipyleaflet

### Data and Scientific Computing

- numpy
- pandas

### Visualization and Web App

- matplotlib
- folium
- Flask

### Configuration

- PyYAML

## Repository Structure

```text
├── app.py                  # Flask app for interactive runs and API routes
├── main.py                 # Batch inference entry point
├── vegetation/             # Vegetation training, inference, scoring, and visualization
├── avalanche/              # Terrain-based avalanche scoring pipeline
├── utils/                  # Config loading, I/O, geospatial helpers, and data collection
├── config/config.yaml      # Project paths, model settings, scoring thresholds
├── data/                   # Raw, processed, and generated outputs
├── models/saved_models/    # Saved model artifacts
├── scripts/                # Utility scripts such as GEE stress tests
├── templates/              # Flask HTML templates
└── requirements.txt        # Python dependencies
```

## Key Modules

- `vegetation/pipeline.py` loads the trained model, predicts probabilities, computes vegetation scores, and saves vegetation overlays.
- `avalanche/pipeline.py` extracts terrain features from DEM data, computes avalanche scores, and saves avalanche overlays.
- `utils/data_collection.py` handles tile collection from Google Earth Engine, local manifests, or synthetic data.
- `utils/visualization_utils.py` builds RGB composites and overlays score masks on imagery.
- `utils/config_loader.py` resolves and loads the YAML configuration.

## Configuration

Most behavior is controlled from `config/config.yaml`.

Common settings you may want to change:

- `data_collection.bounding_boxes` for target regions
- `data_collection.provider` to switch between `gee`, `local`, and `synthetic`
- `data_collection.gee.project_id` for your Earth Engine project
- `model.training.device` for `cpu` or `cuda`
- `scoring` and `visualization` thresholds for map generation

## Installation

```bash
pip install -r requirements.txt
```

If you are using Google Earth Engine, make sure your account and project are authenticated before running the pipeline.

## Usage

### Train the vegetation model

```bash
python vegetation/train.py
```

### Run batch inference

```bash
python main.py
```

### Run the Flask app

```bash
python app.py
```

The Flask app serves the interface in `templates/index.html` and exposes API routes for running the pipeline and reverse geocoding coordinates.

## Outputs

Generated maps are written to `data/output/final/`.

Typical files include:

- `combined_map_<tile_id>.png`
- `vegetation_map_<tile_id>.png`
- `avalanche_map_<tile_id>.png`


Sample maps:

![Final output map](data/output/final/vegetation_map_solang_valley.png)

![Vegetation output map](data/output/final/vegetation_map_interactive_32.1287_76.2474.png)

![Avalanche map](data/output/final/combined_map_interactive_32.0910_76.5255.png)

## Input Data

The project works with Sentinel-2 multispectral bands and DEM elevation data. The default configuration includes:

- 5 Sentinel-2 channels: `B02`, `B03`, `B04`, `B08`, `B11`
- 1 DEM channel for terrain analysis
- 256 x 256 tiles

Datasets used:

- USGS/SRTMGL1_003 for vegetation terrain context
- COPERNICUS/DEM/GLO30 for height

The vegetation model operates on reflectance values from these Sentinel-2 bands:

| Band | Meaning       |
| ---- | ------------- |
| B02  | Blue          |
| B03  | Green         |
| B04  | Red           |
| B08  | Near Infrared |
| B11  | SWIR          |

## Why This Project Exists

The goal is to support restoration planning in steep Himalayan terrain by combining two different perspectives:

- vegetation potential, which tells you where plant growth is likely to succeed
- avalanche safety, which tells you where terrain instability is lower

Using both together produces a more practical restoration suitability map than using either signal alone.

## Notes

- The project is configuration-driven rather than hardcoded.
- Training and inference are intentionally separated.
- GEE, local, and synthetic data providers are all supported through configuration.
