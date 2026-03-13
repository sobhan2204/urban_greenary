pip install -r req.txt

# === FULL PIPELINE (single command) ===
python src/map.py

# === OR run stages individually ===
# python src/prepare_dataset.py --mode multiclass --image data/delhi_satellite.tif --out-images dataset/images --out-masks dataset/masks --patch-size 256 --step 256
# python src/train_unet.py --mode multiclass --img-dir dataset/images --mask-dir dataset/masks --epochs 30 --batch-size 4 --num-classes 5
# python src/predict_unet.py --input dataset/images --checkpoint models/unet_landcover_multiclass.pth --output maps/predictions_multiclass
# python src/planting_zone_analysis.py --mask maps/predictions_multiclass/img_0_mask.png --tile-size 64 --top-k 30 --reference-raster data/delhi_satellite.tif --csv-out data/planting_priority_zones.csv --heatmap-out img/planting_priority_heatmap.png