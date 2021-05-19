"""
Adapted from https://github.com/openai/guided-diffusion/blob/b16b0a180ffac9da8a6a03f1e78de8e96669eee8/guided_diffusion/unet.py.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Predictor, TimeEmbedding


class Resize(nn.Module):
    def __init__(self, scale_factor: float):
        self.scale_factor = scale_factor

    def forward(self, x):
        if self.scale_factor == 1.0:
            return x
        if self.scale_factor < 1.0:
            down_factor = int(round(1 / self.scale_factor))
            assert (
                float(1 / down_factor - self.scale_factor) < 1e-5
            ), "scale factor must be integer or 1/integer"
            return F.avg_pool1d(x, down_factor)
        else:
            return F.interpolate(x, scale_factor=self.scale_factor)


class ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        emb_channels: int,
        out_channels: Optional[int] = None,
        scale_factor: float = 1.0,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels or channels
        self.scale_factor = scale_factor

        skip_conv = nn.Identity()
        if self.channels != self.out_channels:
            skip_conv = nn.Conv1d(self.channels, self.out_channels, 1)
        self.skip = nn.Sequential(
            Resize(scale_factor),
            skip_conv,
        )

        self.cond_layers = nn.Sequential(
            activation(),
            # Start with a small amount of conditioning.
            scale_module(nn.Linear(emb_channels, self.out_channels * 2), s=0.1),
        )
        self.pre_cond = nn.Sequential(
            norm_act(channels),
            Resize(scale_factor),
            nn.Conv1d(self.channels, self.out_channels, 3, padding=1),
            normalization(channels),
        )
        self.post_cond = nn.Sequential(
            activation(),
            scale_module(nn.Conv1d(self.out_channels, self.out_channels, 3, padding=1)),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.pre_cond(x)
        cond_ab = self.cond_layers(cond)[..., None]
        cond_a, cond_b = torch.split(cond_ab, self.out_channels, dim=1)
        h = h * (cond_a + 1) + cond_b
        h = self.post_cond(h)
        return self.skip(x) + h


def norm_act(ch: int) -> nn.Module:
    return nn.Sequential(normalization(ch), activation())


def activation() -> nn.Module:
    return nn.GELU()


def normalization(ch: int) -> nn.Module:
    return nn.GroupNorm(num_groups=32, num_channels=ch)


def scale_module(module: nn.Module, s: float = 0.0) -> nn.Module:
    for p in module.parameters():
        with torch.no_grad():
            p.mul_(s)
    return module
