from abc import abstractmethod
import functools
import math

import torch
import torch.nn as nn


class Predictor(nn.Module):
    @abstractmethod
    def forward(self, xs: torch.Tensor, ts: torch.Tensor, **kwargs):
        """
        Apply the epsilon predictor to a batch of noised inputs.
        """

    def condition(self, **kwargs):
        return functools.partial(self, **kwargs)


class WaveGradPredictor(Predictor):
    """
    A model similar to that in GAN-TTS (https://arxiv.org/abs/1909.11646v2)
    and WaveGrad (https://arxiv.org/abs/2009.00713).
    """

    def __init__(self, cond_channels: int = 512):
        super().__init__()
        self.cond_channels = cond_channels
        self.d_blocks = nn.ModuleList(
            [
                nn.Conv1d(1, 32, 5, padding=2),
                DBlock(32, 128, 2),
                DBlock(128, 128, 2),
                DBlock(128, 256, 2),
                DBlock(256, 512, 2),
            ]
        )
        self.film_blocks = nn.ModuleList(
            [TimestepFILM(ch) for ch in [32, 128, 128, 256, 512]]
        )
        self.u_conv_1 = nn.Conv1d(cond_channels, 768, 3, padding=1)
        self.u_blocks = nn.ModuleList(
            [
                UBlock(768, 512, 512, 4),
                UBlock(512, 512, 256, 2),
                UBlock(512, 256, 128, 2),
                UBlock(256, 128, 128, 2),
                UBlock(128, 128, 32, 2),
            ]
        )
        self.u_conv_2 = nn.Conv1d(128, 1, 3, padding=1)

    def forward(self, x, t, cond=None):
        assert x.shape[2] % 64 == 0, "timesteps must be divisible by 64"

        # Model doesn't need to be conditional
        if cond is None:
            cond = torch.zeros(x.shape[0], self.cond_channels, x.shape[2] // 64).to(x)

        d_outputs = []
        d_input = x
        for block, film_block in zip(self.d_blocks, self.film_blocks):
            d_input = block(d_input)
            d_outputs.append(film_block(d_input, t))

        u_input = self.u_conv_1(cond)
        for block in self.u_blocks:
            u_input = block(u_input, d_outputs.pop())
        out = self.u_conv_2(u_input)
        return out


class WaveGradEncoder(nn.Module):
    """
    An encoder-only version of WaveGradPredictor that can be used to downsample
    waveforms.
    """

    def __init__(self, cond_channels: int = 512):
        super().__init__()
        self.cond_channels = cond_channels
        self.d_blocks = nn.Sequential(
            [
                nn.Conv1d(1, 32, 5, padding=2),
                DBlock(32, 128, 2),
                DBlock(128, 128, 2),
                DBlock(128, 256, 2),
                DBlock(256, 512, 2),
                DBlock(512, cond_channels, 4),
            ]
        )

    def forward(self, x):
        return self.d_blocks(x)


class UBlock(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels, upsample_rate):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_rate = upsample_rate

        self.res_transform = nn.Sequential(
            nn.Upsample(scale_factor=upsample_rate),
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
        )
        self.block_1 = nn.Sequential(
            nn.ReLU(),
            nn.Upsample(scale_factor=upsample_rate),
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
        )
        self.z_linear_1 = nn.Conv1d(cond_channels, out_channels, 1)
        self.block_2 = nn.Sequential(
            nn.ReLU(), nn.Conv1d(out_channels, out_channels, 3, dilation=2, padding=2)
        )
        self.z_linear_2 = nn.Conv1d(cond_channels, out_channels, 1)
        self.block_3 = nn.Sequential(
            nn.ReLU(), nn.Conv1d(out_channels, out_channels, 3, dilation=4, padding=4)
        )
        self.z_linear_3 = nn.Conv1d(cond_channels, out_channels, 1)
        self.block_4 = nn.Sequential(
            nn.ReLU(), nn.Conv1d(out_channels, out_channels, 3, dilation=8, padding=8)
        )

    def forward(self, h, z):
        res_out = self.res_transform(h)
        output = self.block_1(h)
        output = output + self.z_linear_1(z)
        output = self.block_2(output)
        output = output + res_out
        res_out = output
        output = self.block_3(self.z_linear_2(z) + output)
        output = self.block_4(self.z_linear_3(z) + output)
        return output + res_out


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample_rate):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_rate = downsample_rate

        self.res_transform = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.AvgPool1d(2, stride=2),
        )
        self.block_1 = nn.Sequential(
            nn.AvgPool1d(2, stride=2),
            nn.ReLU(),
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 3, dilation=2, padding=2),
        )

    def forward(self, h):
        return self.block_1(h) + self.res_transform(h)


class TimestepFILM(nn.Module):
    def __init__(self, ch, internal_ch=512):
        super().__init__()
        assert internal_ch % 2 == 0, "positional embedding must have dim divisible by 2"
        self.internal_ch = internal_ch
        self.ch = ch
        self.mlp = nn.Sequential(
            nn.Linear(internal_ch, internal_ch),
            nn.ReLU(),
            nn.Linear(internal_ch, ch * 2),
        )

    def forward(self, x, t):
        half = self.internal_ch // 2
        freqs = torch.exp(
            -math.log(10.0)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(x)
        args = t[:, None].to(x.dtype) * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

        gamma_beta = self.mlp(embedding)
        while len(gamma_beta.shape) < len(x.shape):
            gamma_beta = gamma_beta[..., None]
        gamma, beta = torch.split(gamma_beta, gamma_beta.shape[1] // 2, dim=1)

        return x * gamma + beta
