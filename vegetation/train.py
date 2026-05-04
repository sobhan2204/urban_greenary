from __future__ import annotations

import argparse
from pathlib import Path
import logging
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.config_loader import resolve_config_path, load_config, resolve_path
from utils.io_utils import setup_logging, save_model
from utils.data_collection import collect_gangotri_tiles
from vegetation.data_loader import PretrainDataset, SegmentationDataset
from vegetation.model import build_pretrainer, build_segmenter, patchify


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_multissl(model: torch.nn.Module, loader: DataLoader, config: dict) -> None:
    device = torch.device(config["model"]["training"]["device"])
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["model"]["optimizer"]["lr"],
        weight_decay=config["model"]["optimizer"]["weight_decay"],
    )
    epochs = config["model"]["training"]["pretrain_epochs"]
    eps = config["preprocessing"]["normalization"]["epsilon"]
    patch_size = config["model"]["multissl"]["patch_size"]

    model.train()
    grad_clip = config["model"]["training"]["gradient_clip_norm"]
    use_amp = config["model"]["training"]["amp"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                preds, mask, _ = model(batch)
                target = patchify(batch, patch_size)
                mask_float = mask.float()
                loss = ((preds - target) ** 2).mean(dim=-1)
                loss = (loss * mask_float).sum() / (mask_float.sum() + eps)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        logging.info("Pretrain epoch %d loss %.4f", epoch + 1, total_loss / max(1, len(loader)))


def train_segmenter(model: torch.nn.Module, loader: DataLoader, config: dict) -> None:
    device = torch.device(config["model"]["training"]["device"])
    model.to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["model"]["optimizer"]["lr"],
        weight_decay=config["model"]["optimizer"]["weight_decay"],
    )
    epochs = config["model"]["training"]["finetune_epochs"]
    ignore_index = config["model"]["loss"]["ignore_index"]
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    model.train()
    grad_clip = config["model"]["training"]["gradient_clip_norm"]
    use_amp = config["model"]["training"]["amp"] and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    for epoch in range(epochs):
        total_loss = 0.0
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(inputs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        logging.info("Finetune epoch %d loss %.4f", epoch + 1, total_loss / max(1, len(loader)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    config = load_config(config_path)
    setup_logging("INFO")
    set_seed(config["project"]["seed"])

    tiles = collect_gangotri_tiles(config)
    batch_size = config["model"]["training"]["batch_size"]
    num_workers = config["model"]["training"]["num_workers"]

    pretrain_dataset = PretrainDataset(tiles, config)
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    tile_size = config["data_collection"]["tile_size"]
    input_size = (tile_size["height"], tile_size["width"])
    in_channels = config["data_collection"]["sentinel"]["num_channels"] + config["data_collection"]["dem"]["num_channels"]

    pretrainer = build_pretrainer(config, input_size, in_channels)
    train_multissl(pretrainer, pretrain_loader, config)

    encoder = pretrainer.encoder
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()

    segmenter = build_segmenter(config, input_size, in_channels, encoder=encoder)
    seg_dataset = SegmentationDataset(tiles, config)
    seg_loader = DataLoader(seg_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    train_segmenter(segmenter, seg_loader, config)

    model_path = resolve_path(config, "paths", "models", "vegetation_model")
    save_model({"encoder_state": encoder.state_dict(), "decoder_state": segmenter.decoder.state_dict()}, model_path)
    logging.info("Saved model to %s", model_path)


if __name__ == "__main__":
    main()
