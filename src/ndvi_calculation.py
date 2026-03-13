import rasterio
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open("data/delhi_satellite.tif") as src:

    red = src.read(1).astype(float)   # B4
    nir = src.read(4).astype(float)   # B8

    ndvi = (nir - red) / (nir + red + 1e-5)

plt.imshow(ndvi, cmap="RdYlGn")
plt.colorbar()
plt.title("NDVI Map")
plt.show()