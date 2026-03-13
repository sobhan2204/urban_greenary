import ee
import geemap

ee.Initialize(project="urban-green-mapping")

# Define region (Delhi approx coordinates)
delhi = ee.Geometry.Rectangle([77.1, 28.55, 77.2, 28.65])

# Sentinel-2 dataset
dataset = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterDate("2023-01-01", "2023-04-10")
    .filterBounds(delhi)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 10))
)

# Median composite
image = dataset.median().clip(delhi).select(["B4","B3","B2","B8"])

# Export image
geemap.ee_export_image(
    image,
    filename="data/delhi_satellite.tif",
    scale=40,
    region=delhi
)

print("Satellite image downloaded!")