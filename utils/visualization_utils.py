from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from utils.io_utils import ensure_dir


def build_rgb_composite(sentinel: np.ndarray, rgb_indices: list[int], eps: float) -> np.ndarray:
    rgb = sentinel[rgb_indices, :, :].astype(np.float32)
    mins = np.nanmin(rgb, axis=(1, 2), keepdims=True)
    maxs = np.nanmax(rgb, axis=(1, 2), keepdims=True)
    rgb_norm = (rgb - mins) / (maxs - mins + eps)
    rgb_norm = np.nan_to_num(rgb_norm, nan=0.0, posinf=0.0, neginf=0.0)
    rgb_norm = np.clip(rgb_norm, 0.0, 1.0)
    return np.transpose(rgb_norm, (1, 2, 0))


def overlay_mask(base_rgb: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha: float) -> np.ndarray:
    overlay = base_rgb.copy()
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color_arr
    return overlay


def save_image(image: np.ndarray, output_path: Path, dpi: int) -> None:
    ensure_dir(output_path.parent)
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0)
    plt.close()
