image_path  = "data/delhi_satellite.tif"

try:
    from src.vegetation_detection import green_coverage_percentage
except ModuleNotFoundError:
    from vegetation_detection import green_coverage_percentage

green_percent = green_coverage_percentage(image_path)

import rasterio
import numpy as np

CITY_NAME = "Delhi"
CITY_TOTAL_AREA = 1484  
with rasterio.open("data/delhi_satellite.tif") as src:

    red = src.read(1).astype(float)
    nir = src.read(4).astype(float)

    ndvi = (nir - red) / (nir + red + 1e-5)

# vegetation threshold
vegetation_mask = ndvi > 0.4

green_pixels = np.sum(vegetation_mask)
total_pixels = vegetation_mask.size

green_percentage = (green_pixels / total_pixels) * 100

# pixel area (Sentinel-2 resolution)
pixel_area_km2 = 0.0001

vegetation_area = green_pixels * pixel_area_km2

# classification
def classify_city(percent):

    if percent > 40:
        return "Highly Green City"

    elif percent > 20:
        return "Moderate Green City"

    else:
        return "Low Green City"

classification = classify_city(green_percentage)

print(f"City: {CITY_NAME}")
print(f"Total Area: {CITY_TOTAL_AREA} km²")
print(f"Vegetation Area: {vegetation_area:.2f} km²")
print(f"Green Coverage: {green_percentage:.2f}%")
print()
print("Classification:")
print(classification)
    
import folium

m = folium.Map(location=[28.6, 77.2], zoom_start=10)

m.save("data/delhi_map.html")