from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from utils.config_loader import resolve_path
from utils.io_utils import load_model
from vegetation.model import build_segmenter


def load_trained_model(config: dict) -> torch.nn.Module:
    tile_size = config["data_collection"]["tile_size"]
    input_size = (tile_size["height"], tile_size["width"])
    in_channels = config["data_collection"]["sentinel"]["num_channels"] + config["data_collection"]["dem"]["num_channels"]

    model = build_segmenter(config, input_size, in_channels)
    model_path = resolve_path(config, "paths", "models", "vegetation_model")
    state = load_model(model_path, map_location=config["inference"]["device"])
    model.encoder.load_state_dict(state["encoder_state"])
    model.decoder.load_state_dict(state["decoder_state"])
    model.eval()
    return model


def predict_probabilities(model: torch.nn.Module, inputs: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = model(inputs)
        probs = F.softmax(logits, dim=1)
    return probs.cpu().numpy()
