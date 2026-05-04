from __future__ import annotations

from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F


def patchify(inputs: torch.Tensor, patch_size: int) -> torch.Tensor:
    batch, channels, height, width = inputs.shape
    num_h = height // patch_size
    num_w = width // patch_size
    patches = inputs.reshape(batch, channels, num_h, patch_size, num_w, patch_size)
    patches = patches.permute(0, 2, 4, 3, 5, 1).reshape(batch, num_h * num_w, -1)
    return patches


def unpatchify(patches: torch.Tensor, patch_size: int, height: int, width: int, channels: int) -> torch.Tensor:
    batch = patches.shape[0]
    num_h = height // patch_size
    num_w = width // patch_size
    patches = patches.reshape(batch, num_h, num_w, patch_size, patch_size, channels)
    patches = patches.permute(0, 5, 1, 3, 2, 4)
    return patches.reshape(batch, channels, height, width)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: int, drop: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(tokens), self.norm1(tokens), self.norm1(tokens), need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class MultiSSLEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: int,
        drop: float,
        patch_size: int,
        num_patches: int,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, drop) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, inputs: torch.Tensor, mask: torch.Tensor | None = None, mask_token: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        features = self.patch_embed(inputs)
        height, width = features.shape[-2:]
        tokens = features.flatten(2).transpose(1, 2)
        if mask is not None and mask_token is not None:
            mask_token = mask_token.expand(tokens.size(0), tokens.size(1), -1)
            tokens = torch.where(mask.unsqueeze(-1), mask_token, tokens)
        tokens = tokens + self.pos_embed
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        return tokens, (height, width)


class MultiSSLPretrainer(nn.Module):
    def __init__(self, encoder: MultiSSLEncoder, patch_dim: int, mask_ratio: float) -> None:
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.mask_token = nn.Parameter(torch.zeros(1, 1, encoder.pos_embed.shape[-1]))
        self.decoder = nn.Linear(encoder.pos_embed.shape[-1], patch_dim)

    def _random_mask(self, batch: int, num_patches: int, device: torch.device) -> torch.Tensor:
        return torch.rand(batch, num_patches, device=device) < self.mask_ratio

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, int]]:
        batch = inputs.size(0)
        num_patches = self.encoder.pos_embed.shape[1]
        mask = self._random_mask(batch, num_patches, inputs.device)
        tokens, hw = self.encoder(inputs, mask=mask, mask_token=self.mask_token)
        pred_patches = self.decoder(tokens)
        return pred_patches, mask, hw


class KernelAttentionBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(channels, channels, kernel_size, padding=padding, groups=channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate = nn.Conv2d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        attn = torch.sigmoid(self.gate(self.depthwise(inputs)))
        out = self.pointwise(inputs) * attn
        out = self.norm(out)
        return self.dropout(out)


class UKANDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        decoder_channels: list[int],
        num_classes: int,
        kernel_size: int,
        dropout: float,
        upsample_scale: int,
        upsample_mode: str,
    ) -> None:
        super().__init__()
        self.upsample_mode = upsample_mode
        layers = []
        current = in_channels
        for channels in decoder_channels:
            layers.append(
                nn.Sequential(
                    nn.Upsample(
                        scale_factor=upsample_scale,
                        mode=upsample_mode,
                        align_corners=False if upsample_mode != "nearest" else None,
                    ),
                    nn.Conv2d(current, channels, kernel_size=kernel_size, padding=kernel_size // 2),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(inplace=True),
                    KernelAttentionBlock(channels, kernel_size, dropout),
                )
            )
            current = channels
        self.decoder = nn.ModuleList(layers)
        self.head = nn.Conv2d(current, num_classes, kernel_size=1)

    def forward(self, features: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        x = features
        for layer in self.decoder:
            x = layer(x)
        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode=self.upsample_mode, align_corners=False)
        return self.head(x)


class MultiSSLSegmenter(nn.Module):
    def __init__(self, encoder: MultiSSLEncoder, decoder: UKANDecoder, input_size: tuple[int, int]) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.input_size = input_size

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens, hw = self.encoder(inputs)
        features = tokens.transpose(1, 2).reshape(inputs.size(0), -1, hw[0], hw[1])
        return self.decoder(features, self.input_size)


def build_encoder(config: dict, input_size: tuple[int, int], in_channels: int) -> MultiSSLEncoder:
    patch_size = config["model"]["multissl"]["patch_size"]
    num_patches = (input_size[0] // patch_size) * (input_size[1] // patch_size)
    return MultiSSLEncoder(
        in_channels=in_channels,
        embed_dim=config["model"]["multissl"]["embed_dim"],
        depth=config["model"]["multissl"]["depth"],
        num_heads=config["model"]["multissl"]["num_heads"],
        mlp_ratio=config["model"]["multissl"]["mlp_ratio"],
        drop=config["model"]["multissl"]["drop_rate"],
        patch_size=patch_size,
        num_patches=num_patches,
    )


def build_pretrainer(config: dict, input_size: tuple[int, int], in_channels: int) -> MultiSSLPretrainer:
    encoder = build_encoder(config, input_size, in_channels)
    patch_dim = in_channels * config["model"]["multissl"]["patch_size"] ** 2
    return MultiSSLPretrainer(encoder, patch_dim, config["model"]["multissl"]["mask_ratio"])


def build_segmenter(
    config: dict, input_size: tuple[int, int], in_channels: int, encoder: MultiSSLEncoder | None = None
) -> MultiSSLSegmenter:
    if encoder is None:
        encoder = build_encoder(config, input_size, in_channels)
    decoder = UKANDecoder(
        in_channels=encoder.pos_embed.shape[-1],
        decoder_channels=config["model"]["ukan"]["decoder_channels"],
        num_classes=config["model"]["ukan"]["num_classes"],
        kernel_size=config["model"]["ukan"]["kernel_size"],
        dropout=config["model"]["ukan"]["dropout"],
        upsample_scale=config["model"]["ukan"]["upsample_scale"],
        upsample_mode=config["model"]["ukan"]["upsample_mode"],
    )
    return MultiSSLSegmenter(encoder, decoder, input_size)
