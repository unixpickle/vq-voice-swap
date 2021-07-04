"""
Encoders from https://arxiv.org/abs/1901.08810.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .base import Encoder


class ConvMFCCEncoder(Encoder):
    """
    The convolutional model with MFCC features at regular intervals.

    Requires torchaudio upon initialization.
    """

    def __init__(
        self,
        base_channels: int,
        out_channels: int = 64,
        input_ulaw: bool = True,
        input_rate: int = 16000,
        mfcc_rate: int = 100,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.out_channels = out_channels
        self.input_ulaw = input_ulaw
        self.input_rate = input_rate
        self.mfcc_rate = mfcc_rate
        self.mid_channels = base_channels * 12

        assert mfcc_rate % 2 == 0, "must be able to downsample MFCCs once"
        assert input_rate % mfcc_rate == 0, "must evenly downsample input sequences"

        from torchaudio.transforms import MFCC

        self.mfcc = MFCC(
            sample_rate=input_rate,
            n_mfcc=13,
            log_mels=True,
            melkwargs=dict(
                n_fft=(input_rate // self.mfcc_rate) * 2,
                hop_length=input_rate // self.mfcc_rate,
                n_mels=40,
            ),
        )

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(13 * 3, self.mid_channels, 3, padding=1),
                    nn.GELU(),
                ),
                ResConv(self.mid_channels, self.mid_channels, 3, padding=1),
                nn.Sequential(
                    nn.Conv1d(
                        self.mid_channels, self.mid_channels, 4, stride=2, padding=1
                    ),
                    nn.GELU(),
                ),
                *[
                    ResConv(self.mid_channels, self.mid_channels, 3, padding=1)
                    for _ in range(2)
                ],
                *[ResConv(self.mid_channels, self.mid_channels, 1) for _ in range(4)],
                nn.Conv1d(self.mid_channels, self.out_channels, 1),
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        use_checkpoint: bool = False,
    ) -> torch.Tensor:
        assert x.shape[1] == 1, "input must only have one channel"
        if self.input_ulaw:
            # MFCC layer expects linear waveform.
            x = invert_ulaw(x)
        h = self.mfcc(x[:, 0, :])
        deriv = deltas(h)
        accel = deltas(deriv)
        h = torch.cat([h, deriv, accel], dim=1)
        for block in self.blocks:
            if use_checkpoint:
                h = checkpoint(block, h)
            else:
                h = block(h)
        return h

    @property
    def downsample_rate(self) -> int:
        return self.input_rate // (self.mfcc_rate // 2)


class ResConv(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv1d(*args, **kwargs)

    def forward(self, x):
        h = self.conv(x)
        h = F.gelu(h)
        return x + h


def deltas(seq: torch.Tensor) -> torch.Tensor:
    right_shifted = torch.cat([seq[..., :1], seq[..., :-1]], dim=-1)
    left_shifted = torch.cat([seq[..., 1:], seq[..., -1:]], dim=-1)

    d1 = right_shifted - seq
    d2 = seq - left_shifted
    return (d1 + d2) / 2


def invert_ulaw(x: torch.Tensor, mu: float = 255.0) -> torch.Tensor:
    return x.sign() * (1 / mu) * ((1 + mu) ** x.abs() - 1)
