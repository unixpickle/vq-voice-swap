import os

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .model import TimeEmbedding
from .unet import activation, norm_act, normalization, scale_module
from .util import atomic_save


class Classifier(nn.Module):
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

    def save(self, path):
        """
        Save this model and its hyperparameters to a file.
        """
        state = {
            "kwargs": {
                "num_labels": self.num_labels,
                "base_channels": self.stem.base_channels,
                "channel_mult": self.stem.channel_mult,
                "depth_mult": self.stem.depth_mult,
            },
            "state_dict": self.state_dict(),
        }
        atomic_save(state, path)

    @classmethod
    def load(cls, path):
        """
        Load a fresh model instance from a file.
        """
        state = torch.load(path, map_location="cpu")
        obj = cls(**state["kwargs"])
        obj.load_state_dict(state["state_dict"])
        return obj


class ClassifierStem(nn.Module):
    """
    A module which takes [N x 1 x T] sequences and produces feature vectors of
    the shape [N x C].
    """

    def __init__(
        self,
        base_channels: int = 32,
        channel_mult: int = (1, 1, 2, 2, 2, 4, 4, 8, 16),
        depth_mult: int = 2,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.channel_mult = channel_mult
        self.depth_mult = depth_mult
        self.out_channels = base_channels * channel_mult[-1]

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
                    ClassifierResBlock(
                        in_channels=cur_channels,
                        out_channels=ch_mult * base_channels,
                        emb_channels=embed_dim,
                    )
                )
                cur_channels = ch_mult * base_channels
            self.blocks.append(
                ClassifierResBlock(
                    in_channels=cur_channels,
                    out_channels=cur_channels,
                    emb_channels=embed_dim,
                    stride=2,
                )
            )

        self.out = nn.Sequential(
            norm_act(self.out_channels),
            nn.Conv1d(self.out_channels, self.out_channels, kernel_size=3, padding=1),
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
        h = self.out(h)
        return h.mean(dim=-1)


class ClassifierResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        dilation: int = 2,
        stride: int = 1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        if in_channels != out_channels or stride != 1:
            self.skip = nn.Conv1d(
                in_channels, out_channels, 3, padding=1, stride=stride
            )
        else:
            self.skip = nn.Identity()

        # Adapted from the UNet ResBlock.
        self.cond_layers = nn.Sequential(
            activation(),
            scale_module(nn.Linear(emb_channels, out_channels * 2), s=0.1),
        )
        self.pre_cond = nn.Sequential(
            norm_act(in_channels),
            nn.Conv1d(in_channels, out_channels, 3, stride=stride, padding=1),
            normalization(out_channels),
        )
        self.post_cond = nn.Sequential(
            activation(),
            scale_module(
                nn.Conv1d(
                    out_channels,
                    out_channels,
                    3,
                    padding=dilation,
                    dilation=dilation,
                )
            ),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.pre_cond(x)
        cond_ab = self.cond_layers(cond)[..., None]
        cond_a, cond_b = torch.split(cond_ab, self.out_channels, dim=1)
        h = h * (cond_a + 1) + cond_b
        h = self.post_cond(h)
        return self.skip(x) + h
