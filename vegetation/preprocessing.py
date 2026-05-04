from __future__ import annotations

import numpy as np


def normalize_array(array: np.ndarray, method: str, per_channel: bool, eps: float) -> np.ndarray:
    if method == "minmax":
        if per_channel:
            mins = array.min(axis=(1, 2), keepdims=True)
            maxs = array.max(axis=(1, 2), keepdims=True)
        else:
            mins = array.min(keepdims=True)
            maxs = array.max(keepdims=True)
        return (array - mins) / (maxs - mins + eps)
    if method == "standard":
        if per_channel:
            means = array.mean(axis=(1, 2), keepdims=True)
            stds = array.std(axis=(1, 2), keepdims=True)
        else:
            means = array.mean(keepdims=True)
            stds = array.std(keepdims=True)
        return (array - means) / (stds + eps)
    raise ValueError(f"Unsupported normalization method: {method}")


def compute_ndvi(sentinel: np.ndarray, nir_index: int, red_index: int, eps: float) -> np.ndarray:
    nir = sentinel[nir_index]
    red = sentinel[red_index]
    return (nir - red) / (nir + red + eps)


def compute_ndsi(sentinel: np.ndarray, green_index: int, swir_index: int, eps: float) -> np.ndarray:
    green = sentinel[green_index]
    swir = sentinel[swir_index]
    return (green - swir) / (green + swir + eps)


def compute_ndwi(sentinel: np.ndarray, green_index: int, nir_index: int, eps: float) -> np.ndarray:
    green = sentinel[green_index]
    nir = sentinel[nir_index]
    return (green - nir) / (green + nir + eps)


def stack_modalities(sentinel: np.ndarray, dem: np.ndarray) -> np.ndarray:
    if dem.ndim == 2:
        dem = dem[None, :, :]
    return np.concatenate([sentinel, dem], axis=0)


def validate_patch_compatibility(height: int, width: int, patch_size: int) -> None:
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError("Tile size must be divisible by patch size")
