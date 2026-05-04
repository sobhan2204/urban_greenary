from __future__ import annotations

from typing import Any
import numpy as np
import torch
from torch.utils.data import Dataset

from vegetation.preprocessing import (
    normalize_array,
    compute_ndvi,
    compute_ndsi,
    stack_modalities,
    validate_patch_compatibility,
)
from vegetation.labeling import generate_pseudo_labels


def prepare_tile(tile: dict[str, Any], config: dict) -> dict[str, np.ndarray]:
    sentinel = tile["sentinel"].astype(np.float32)
    dem = tile["dem"].astype(np.float32)
    eps = config["preprocessing"]["normalization"]["epsilon"]
    ndvi_cfg = config["preprocessing"]["ndvi"]
    ndsi_cfg = config["preprocessing"]["ndsi"]

    ndvi = compute_ndvi(sentinel, ndvi_cfg["nir_band_index"], ndvi_cfg["red_band_index"], eps)
    ndsi = compute_ndsi(sentinel, ndsi_cfg["green_band_index"], ndsi_cfg["swir_band_index"], eps)

    norm_cfg = config["preprocessing"]["normalization"]
    sentinel_norm = normalize_array(sentinel, norm_cfg["method"], norm_cfg["per_channel"], eps)
    dem_norm = normalize_array(dem, norm_cfg["method"], norm_cfg["per_channel"], eps)

    stacked = stack_modalities(sentinel_norm, dem_norm)

    patch_size = config["model"]["multissl"]["patch_size"]
    validate_patch_compatibility(stacked.shape[1], stacked.shape[2], patch_size)

    return {"input": stacked, "ndvi": ndvi, "ndsi": ndsi, "dem": dem}


class PretrainDataset(Dataset):
    def __init__(self, tiles: list[dict[str, Any]], config: dict) -> None:
        self.tiles = tiles
        self.config = config

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = prepare_tile(self.tiles[idx], self.config)
        return torch.from_numpy(sample["input"]).float()


class SegmentationDataset(Dataset):
    def __init__(self, tiles: list[dict[str, Any]], config: dict) -> None:
        self.tiles = tiles
        self.config = config

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = prepare_tile(self.tiles[idx], self.config)
        labels = generate_pseudo_labels(sample["ndvi"], sample["ndsi"], self.config)
        inputs = torch.from_numpy(sample["input"]).float()
        targets = torch.from_numpy(labels).long()
        return inputs, targets
