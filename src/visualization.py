import rasterio
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open("data/delhi_satellite.tif") as src:

    red = src.read(1).astype(float)
    nir = src.read(4).astype(float)

    ndvi = (nir - red) / (nir + red + 1e-5)

# vegetation threshold
vegetation_mask = ndvi > 0.4

# ---- NDVI HEATMAP ----
plt.figure(figsize=(10,6))
plt.imshow(ndvi, cmap="RdYlGn")
plt.colorbar(label="NDVI Value")
plt.title("NDVI Vegetation Heatmap - Delhi")

plt.savefig("img/delhi_ndvi_heatmap.png")
plt.show()


# ---- VEGETATION MASK ----
plt.figure(figsize=(10,6))
plt.imshow(vegetation_mask, cmap="Greens")
plt.title("Detected Vegetation Areas")

plt.savefig("img/delhi_vegetation_mask.png")
plt.show()