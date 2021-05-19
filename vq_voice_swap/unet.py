"""
Adapted from https://github.com/openai/guided-diffusion/blob/b16b0a180ffac9da8a6a03f1e78de8e96669eee8/guided_diffusion/unet.py.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .model import Predictor, TimeEmbedding


class UNetPredictor(Predictor):
    def __init__(
        self,
        base_channels: int,
        channel_mult: Tuple[int] = (1, 1, 2, 2, 2, 4, 4, 8),
        depth_mult: int = 2,
        cond_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
        in_channels: int = 1,
        out_channels: int = 1,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.depth_mult = depth_mult
        self.cond_channels = cond_channels
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = out_channels

        embed_dim = base_channels * 4
        self.time_embed = TimeEmbedding(embed_dim)
        if num_classes is not None:
            self.class_embed = nn.Embedding(num_classes, embed_dim)
        if cond_channels is not None:
            self.cond_proj = nn.Conv1d(cond_channels, base_channels)

        self.in_conv = nn.Conv1d(in_channels, base_channels, 1)

        skip_channels = [base_channels]
        cur_channels = base_channels

        self.down_blocks = nn.ModuleList([])
        for depth, mult in enumerate(channel_mult):
            for _ in range(depth_mult):
                self.down_blocks.append(
                    ResBlock(
                        channels=cur_channels,
                        emb_channels=embed_dim,
                        out_channels=mult * base_channels,
                    )
                )
                cur_channels = mult * base_channels
                skip_channels.append(cur_channels)
            if depth != len(channel_mult) - 1:
                self.down_blocks.append(
                    ResBlock(
                        channels=cur_channels,
                        emb_channels=embed_dim,
                        scale_factor=0.5,
                    ),
                )
                skip_channels.append(cur_channels)

        self.middle_blocks = nn.ModuleList(
            [
                ResBlock(
                    channels=cur_channels,
                    emb_channels=embed_dim,
                )
                for _ in range(2)
            ]
        )

        self.up_blocks = nn.ModuleList([])
        for depth, mult in list(enumerate(channel_mult))[::-1]:
            for _ in range(depth_mult + 1):
                in_ch = skip_channels.pop()
                self.up_blocks.append(
                    ResBlock(
                        channels=cur_channels + in_ch,
                        emb_channels=embed_dim,
                        out_channels=mult * base_channels,
                    )
                )
                cur_channels = mult * base_channels
            if depth:
                self.up_blocks.append(
                    ResBlock(
                        channels=cur_channels, emb_channels=embed_dim, scale_factor=2.0
                    )
                )

        self.out = nn.Sequential(
            norm_act(base_channels),
            nn.Conv1d(base_channels, out_channels, 3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        ts: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        assert (labels is None) == (
            self.num_classes is None
        ), "must provide labels if and only if model is class conditional"
        assert (cond is None) == (
            self.cond_channels is None
        ), "must provide cond sequence if and only if model is conditional"

        emb = self.time_embed(ts)
        if labels is not None:
            emb = emb + self.class_embed(labels)

        h = self.in_conv(x)
        if cond is not None:
            h = h + F.interpolate(self.cond_proj(cond), h.shape[-1])

        skips = [h]
        for block in self.down_blocks:
            if use_checkpoint:
                h = checkpoint(block, h, emb)
            else:
                h = block(h, emb)
            skips.append(h)
        for block in self.middle_blocks:
            if use_checkpoint:
                h = checkpoint(block, h, emb)
            else:
                h = block(h, emb)
        for i, block in enumerate(self.up_blocks):
            # No skip connection for upsampling block.
            if i % (self.depth_mult + 2) != self.depth_mult + 1:
                h = torch.cat([h, skips.pop()], axis=1)
            if use_checkpoint:
                h = checkpoint(block, h, emb)
            else:
                h = block(h, emb)

        h = self.out(h)
        return h


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
            normalization(self.out_channels),
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


class Resize(nn.Module):
    def __init__(self, scale_factor: float):
        super().__init__()
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def norm_act(ch: int) -> nn.Module:
    return nn.Sequential(normalization(ch), activation())


def activation() -> nn.Module:
    return nn.GELU()


def normalization(ch: int) -> nn.Module:
    return nn.GroupNorm(num_groups=min(32, ch), num_channels=ch)


def scale_module(module: nn.Module, s: float = 0.0) -> nn.Module:
    for p in module.parameters():
        with torch.no_grad():
            p.mul_(s)
    return module
