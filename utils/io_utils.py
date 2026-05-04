from __future__ import annotations

from pathlib import Path
import logging
import numpy as np
import torch


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_numpy(array: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    np.save(path, array)


def load_numpy(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def save_model(state: dict, path: Path) -> None:
    ensure_dir(path.parent)
    torch.save(state, path)


def load_model(path: Path, map_location: str | None = None) -> dict:
    return torch.load(path, map_location=map_location)
