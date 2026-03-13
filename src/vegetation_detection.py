import rasterio
import numpy as np

def green_coverage_percentage(image_path):
 with rasterio.open(image_path) as src:

    red = src.read(1).astype(float)
    nir = src.read(4).astype(float)

    ndvi = (nir - red) / (nir + red + 1e-5)

 vegetation_mask = ndvi > 0.4

 green_pixels = np.sum(vegetation_mask)
 total_pixels = vegetation_mask.size

 green_percentage = (green_pixels / total_pixels) * 100

 print("Green Coverage:", round(green_percentage,2), "%")
 return green_percentage

if __name__ == "__main__":
    image_path  = "data/delhi_satellite.tif"
    green_coverage_percentage(image_path)