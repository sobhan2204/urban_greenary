from __future__ import annotations

from pathlib import Path
import numpy as np

from utils.visualization_utils import overlay_mask, save_image


def save_overlay_map(
    base_rgb: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    color: tuple[float, float, float],
    alpha: float,
    dpi: int,
) -> None:
    overlay = overlay_mask(base_rgb, mask, color, alpha)
    save_image(overlay, output_path, dpi)
