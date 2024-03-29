"""
Flexible audio sequence classification models.
"""

import math
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .base import Savable
from .unet import ResBlock, UNetPredictor, activation, norm_act, scale_module
from .wavegrad import TimeEmbedding


class Classifier(Savable):
    """
    A module which adds an N-way linear layer head to a classifier stem.
    """

    def __init__(self, num_labels: int, **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.stem = ClassifierStem(**kwargs)
        self.out = nn.Sequential(
            activation(), scale_module(nn.Linear(self.stem.out_channels, num_labels))
        )

    def forward(
        self, x: torch.Tensor, ts: torch.Tensor, use_checkpoint: bool = False, **kwargs
    ) -> torch.Tensor:
        h = self.stem(x, ts, use_checkpoint=use_checkpoint, **kwargs)
        h = self.out(h)
        return h

    def save_kwargs(self) -> Dict[str, Any]:
        return dict(
            num_labels=self.num_labels,
            base_channels=self.stem.base_channels,
            channel_mult=self.stem.channel_mult,
            output_mult=self.stem.output_mult,
            depth_mult=self.stem.depth_mult,
        )


class ClassifierStem(nn.Module):
    """
    A module which takes [N x 1 x T] sequences and produces feature vectors of
    the shape [N x C].
    """

    def __init__(
        self,
        base_channels: int = 32,
        channel_mult: int = (1, 1, 2, 2, 2, 4, 4, 8, 8),
        output_mult: int = 16,
        depth_mult: int = 2,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.output_mult = output_mult
        self.depth_mult = depth_mult
        self.out_channels = base_channels * output_mult

        embed_dim = base_channels * 4
        self.embed_dim = embed_dim
        self.time_embed = TimeEmbedding(embed_dim)
        self.time_embed_extra = nn.Sequential(
            activation(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.in_conv = nn.Conv1d(1, base_channels, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([])
        cur_channels = base_channels
        for ch_mult in channel_mult:
            for _ in range(depth_mult):
                self.blocks.append(
                    ResBlock(
                        channels=cur_channels,
                        out_channels=ch_mult * base_channels,
                        emb_channels=embed_dim,
                    )
                )
                cur_channels = ch_mult * base_channels
            self.blocks.append(
                ResBlock(
                    channels=cur_channels,
                    out_channels=cur_channels,
                    emb_channels=embed_dim,
                    scale_factor=0.5,
                )
            )

        self.out = nn.Sequential(
            norm_act(cur_channels),
            AttentionPool1d(
                cur_channels,
                head_channels=min(cur_channels, 64),
                out_channels=self.out_channels,
            ),
        )

    def conditional_embedding(self, ts: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.time_embed_extra(self.time_embed(ts))

    def forward(
        self, x: torch.Tensor, ts: torch.Tensor, use_checkpoint: bool = False, **kwargs
    ) -> torch.Tensor:
        emb = self.conditional_embedding(ts, **kwargs)
        h = self.in_conv(x)
        for block in self.blocks:
            if use_checkpoint:
                h = checkpoint(block, h, emb)
            else:
                h = block(h, emb)
        return self.out(h)

    def load_from_predictor(self, pred: UNetPredictor) -> int:
        dsts = [self.in_conv, self.time_embed, self.time_embed_extra, *self.blocks]
        srcs = [pred.in_conv, pred.time_embed, pred.time_embed_extra, *pred.down_blocks]
        total = 0
        for dst, src in zip(dsts, srcs):
            dst.load_state_dict(src.state_dict())
            total += sum(np.prod(x.shape) for x in src.state_dict().values())
        return total


class AttentionPool1d(nn.Module):
    """
    Adapted from: https://github.com/openai/guided-diffusion/blob/b16b0a180ffac9da8a6a03f1e78de8e96669eee8/guided_diffusion/unet.py#L22
    """

    def __init__(
        self,
        channels: int,
        head_channels: int = 64,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        assert (
            channels % head_channels == 0
        ), f"head channels ({head_channels}) must divide output channels ({out_channels})"
        self.qkv_proj = nn.Conv1d(channels, 3 * channels, 1)
        self.c_proj = nn.Conv1d(channels, out_channels or channels, 1)
        self.num_heads = channels // head_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.cat([torch.zeros_like(x[..., :1]), x], dim=-1)  # NC(T+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[..., 0]


class QKVAttention(nn.Module):
    """
    Adapted from: https://github.com/openai/guided-diffusion/blob/b16b0a180ffac9da8a6a03f1e78de8e96669eee8/guided_diffusion/unet.py#L361
    """

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor) -> torch.Tensor:
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )
        weight = torch.softmax(weight, dim=-1)
        a = torch.einsum(
            "bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length)
        )
        return a.reshape(bs, -1, length)
