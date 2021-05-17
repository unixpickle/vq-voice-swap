from abc import abstractmethod
import functools
import math
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


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

    def __init__(
        self,
        cond_mult: int = 16,
        base_channels: int = 32,
        num_labels: Optional[int] = None,
    ):
        super().__init__()
        self.cond_channels = cond_mult * base_channels
        self.base_channels = base_channels
        self.d_blocks = nn.ModuleList(
            [
                nn.Conv1d(1, base_channels, 5, padding=2),
                DBlock(base_channels, base_channels * 4, 4),
                DBlock(base_channels * 4, base_channels * 4, 2),
                DBlock(base_channels * 4, base_channels * 8, 2),
                DBlock(base_channels * 8, base_channels * 16, 2),
            ]
        )
        self.u_conv_1 = nn.Conv1d(self.cond_channels, base_channels * 24, 3, padding=1)
        self.u_blocks = nn.ModuleList(
            [
                UBlock(
                    base_channels * 24,
                    base_channels * 16,
                    base_channels * 16,
                    2,
                    num_labels=num_labels,
                ),
                UBlock(
                    base_channels * 16,
                    base_channels * 16,
                    base_channels * 8,
                    2,
                    num_labels=num_labels,
                ),
                UBlock(
                    base_channels * 16,
                    base_channels * 8,
                    base_channels * 4,
                    2,
                    num_labels=num_labels,
                ),
                UBlock(
                    base_channels * 8,
                    base_channels * 4,
                    base_channels * 4,
                    2,
                    num_labels=num_labels,
                ),
                UBlock(
                    base_channels * 4,
                    base_channels * 4,
                    base_channels,
                    4,
                    num_labels=num_labels,
                ),
            ]
        )
        self.u_ln = NCTLayerNorm(base_channels * 4)
        self.u_conv_2 = nn.Conv1d(base_channels * 4, 1, 3, padding=1)
        for p in self.u_conv_2.parameters():
            with torch.no_grad():
                p.zero_()

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_checkpoint=False,
    ):
        assert x.shape[2] % 64 == 0, "timesteps must be divisible by 64"

        # Model doesn't need to be conditional
        if cond is None:
            cond = torch.zeros(x.shape[0], self.cond_channels, x.shape[2] // 64).to(x)

        d_outputs = []
        d_input = x
        for block in self.d_blocks:
            if use_checkpoint:
                if not d_input.requires_grad:
                    d_input = d_input.clone().requires_grad_(True)
                d_input = checkpoint(block, d_input)
            else:
                d_input = block(d_input)
            d_outputs.append(d_input)

        u_input = self.u_conv_1(cond)
        for block in self.u_blocks:

            def run_fn(u_input, d_output, block=block, t=t, labels=labels):
                return block(u_input, d_output, t, labels=labels)

            if use_checkpoint:
                u_input = checkpoint(run_fn, u_input, d_outputs.pop())
            else:
                u_input = run_fn(u_input, d_outputs.pop())
        out = self.u_ln(u_input)
        out = self.u_conv_2(out)
        return out


class WaveGradEncoder(nn.Module):
    """
    An encoder-only version of WaveGradPredictor that can be used to downsample
    waveforms.
    """

    def __init__(self, cond_mult: int = 16, base_channels: int = 32):
        super().__init__()
        self.cond_channels = cond_mult * base_channels
        self.d_blocks = nn.Sequential(
            nn.Conv1d(1, base_channels, 5, padding=2),
            DBlock(base_channels, base_channels * 4, 4, extra_blocks=1),
            DBlock(base_channels * 4, base_channels * 4, 2, extra_blocks=1),
            DBlock(base_channels * 4, base_channels * 8, 2, extra_blocks=1),
            DBlock(base_channels * 8, base_channels * 16, 2, extra_blocks=1),
            DBlock(base_channels * 16, self.cond_channels, 2, extra_blocks=1),
        )

    def forward(self, x, use_checkpoint=False):
        if use_checkpoint:
            if not x.requires_grad:
                x = x.clone().requires_grad_()
            return checkpoint_sequential(self.d_blocks, len(self.d_blocks), x)
        else:
            return self.d_blocks(x)


class UBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_channels: int,
        upsample_rate: int,
        num_labels: Optional[int] = None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cond_channels = cond_channels
        self.upsample_rate = upsample_rate

        def make_film():
            return FILM(cond_channels, out_channels, num_labels=num_labels)

        self.film_1 = make_film()
        self.film_2 = make_film()
        self.film_3 = make_film()

        self.res_transform = nn.Sequential(
            nn.Upsample(scale_factor=upsample_rate),
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
        )
        self.block_1 = nn.Sequential(
            NCTLayerNorm(in_channels),
            nn.GELU(),
            nn.Upsample(scale_factor=upsample_rate),
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
        )
        self.block_2 = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 3, dilation=2, padding=2),
        )
        self.block_3 = nn.Sequential(
            NCTLayerNorm(out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 3, dilation=4, padding=4),
        )
        self.block_4 = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 3, dilation=8, padding=8),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 3, dilation=16, padding=16),
        )

    def forward(
        self,
        h: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        res_out = self.res_transform(h)
        output = self.block_1(h)
        output = self.block_2(self.film_1(output, z, t, labels=labels))
        output = output + res_out
        res_out = output
        output = self.block_3(self.film_2(output, z, t, labels=labels))
        output = self.block_4(self.film_3(output, z, t, labels=labels))
        return output + res_out


class DBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample_rate: int,
        extra_blocks: int = 0,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.downsample_rate = downsample_rate
        self.extra_blocks = extra_blocks

        self.res_transform = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.AvgPool1d(downsample_rate, stride=downsample_rate),
        )
        self.block_1 = nn.Sequential(
            NCTLayerNorm(in_channels),
            nn.AvgPool1d(downsample_rate, stride=downsample_rate),
            nn.GELU(),
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, 3, dilation=2, padding=2),
        )
        self.extra = nn.ModuleList(
            [
                nn.Sequential(
                    NCTLayerNorm(out_channels),
                    nn.GELU(),
                    nn.Conv1d(out_channels, out_channels, 3, padding=1),
                    nn.GELU(),
                    nn.Conv1d(out_channels, out_channels, 3, dilation=4, padding=4),
                    nn.GELU(),
                    nn.Conv1d(out_channels, out_channels, 3, dilation=8, padding=8),
                )
                for _ in range(extra_blocks)
            ]
        )

    def forward(self, h: torch.Tensor):
        res = self.block_1(h) + self.res_transform(h)
        for block in self.extra:
            res = res + block(res)
        return res


class FILM(nn.Module):
    """
    A FiLM layer that conditions on a timestep and (possibly) a label.

    The timestep is a floating point in the range [0, 1], whereas the labels
    are integers in the range [0, num_labels).

    The output of a FiLM layer is a tuple (alpha, beta), where alpha is a
    multiplier and beta is a bias.
    """

    def __init__(
        self, cond_channels: int, out_channels: int, num_labels: Optional[int] = None
    ):
        super().__init__()
        self.cond_channels = cond_channels
        self.out_channels = out_channels
        self.hidden_channels = out_channels * 2
        self.num_labels = num_labels
        self.time_emb = nn.Linear(self.hidden_channels, self.hidden_channels)
        self.cond_emb = nn.Sequential(
            NCTLayerNorm(cond_channels),
            nn.Conv1d(cond_channels, self.hidden_channels, 3, padding=1),
        )
        if num_labels is not None:
            self.label_emb = nn.Embedding(num_labels, self.hidden_channels)
            # Random initial label embeddings appears to hurt performance.
            with torch.no_grad():
                self.label_emb.weight.zero_()
        else:
            self.label_emb = None
        self.out_layer = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(self.hidden_channels, out_channels * 2, 3, padding=1),
        )
        # Start off with little conditioning signal.
        with torch.no_grad():
            self.out_layer[1].weight.mul_(0.1)
            self.out_layer[1].bias.mul_(0.0)

    def forward(
        self,
        inputs: torch.Tensor,
        cond: torch.Tensor,
        t: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        half = self.hidden_channels // 2
        min_coeff = 0.1
        max_coeff = 100.0
        freqs = (
            torch.exp(
                -math.log(max_coeff / min_coeff)
                * torch.arange(start=0, end=half, dtype=torch.float32)
                / (half - 1)
            ).to(cond)
            * max_coeff
        )
        args = t[:, None].to(cond.dtype) * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        embedding = self.time_emb(embedding)
        assert (labels is None) == (self.label_emb is None)
        if labels is not None:
            embedding = embedding + self.label_emb(labels)
        while len(embedding.shape) < len(cond.shape):
            embedding = embedding[..., None]
        embedding = embedding + self.cond_emb(cond)
        alpha_beta = self.out_layer(embedding)
        alpha, beta = torch.split(alpha_beta, self.out_channels, dim=1)
        return inputs * (1 + alpha) + beta


class NCTLayerNorm(nn.Module):
    """
    LayerNorm that normalizes channels in NCT tensors.
    """

    def __init__(self, ch: int):
        super().__init__()
        self.ln = nn.LayerNorm((ch,))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1).contiguous()
        x = self.ln(x)
        x = x.permute(0, 2, 1).contiguous()
        return x
