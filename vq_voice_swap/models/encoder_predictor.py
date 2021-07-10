"""
Models to predict the outputs of an Encoder from noised audio.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Savable
from .unet import UNetPredictor


class EncoderPredictor(Savable):
    """
    A model which predicts a series of categorical variables.

    :param base_channels: channel multiplier for the model.
    :param downsample_rate: downsampling factor for the latents.
    :param num_latents: dictionary size we are predicting.
    :param bottleneck_dim: the bottleneck layer dimension.
    """

    def __init__(
        self,
        base_channels: int,
        downsample_rate: int,
        num_latents: int,
        bottleneck_dim: int = 64,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.downsample_rate = downsample_rate
        self.num_latents = num_latents
        self.bottleneck_dim = bottleneck_dim
        self.unet = UNetPredictor(base_channels, out_channels=bottleneck_dim)
        self.out = nn.Conv1d(bottleneck_dim, num_latents, 1)

    def forward(
        self, x: torch.Tensor, ts: torch.Tensor, use_checkpoint: bool = False
    ) -> torch.Tensor:
        """
        Predict the codes for a given sequence.

        :param x: an [N x C x T] Tensor.
        :param ts: an [N] Tensor of timesteps.
        :param use_checkpoint: if true, use gradient checkpointing.
        :return: an [N x D x T//R] Tensor of logits, where D is the number of
                 categorical latents, and R is the downsampling rate.
        """
        h = self.unet(x, ts, use_checkpoint=use_checkpoint)
        h = F.interpolate(
            h, size=(h.shape[-1] // self.downsample_rate,), mode="nearest"
        )
        h = self.out(h)
        return h

    def losses(
        self, x: torch.Tensor, ts: torch.Tensor, targets: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        losses = F.cross_entropy(self(x, ts, **kwargs), targets, reduction="none")
        return losses.mean(-1)

    def save_kwargs(self) -> Dict[str, Any]:
        return dict(
            base_channels=self.base_channels,
            downsample_rate=self.downsample_rate,
            num_latents=self.num_latents,
            bottleneck_dim=self.bottleneck_dim,
        )
