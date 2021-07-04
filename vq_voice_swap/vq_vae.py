from typing import Any, Dict, Optional

import torch
from torch._C import Value

from .diffusion_model import DiffusionModel
from .models import make_encoder
from .vq import VQ, VQLoss


class VQVAE(DiffusionModel):
    """
    A waveform VQ-VAE with a diffusion decoder.
    """

    def __init__(
        self, base_channels: int, enc_name: str = "unet", cond_mult: int = 16, **kwargs
    ):
        encoder = make_encoder(
            enc_name=enc_name, base_channels=base_channels, cond_mult=cond_mult
        )
        kwargs["cond_channels"] = base_channels * cond_mult
        super().__init__(base_channels=base_channels, **kwargs)
        self.enc_name = enc_name
        self.cond_mult = cond_mult
        self.encoder = encoder
        self.vq = VQ(self.cond_channels, 512)

    def losses(
        self,
        vq_loss: VQLoss,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        jitter: float = 0.0,
        **extra_kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute losses for training the VQVAE.

        :param vq_loss: the vector-quantization loss function.
        :param inputs: the input [N x 1 x T] audio Tensor.
        :param labels: an [N] Tensor of integer labels.
        :param jitter: jitter regularization to use.
        :return: a dict containing the following keys:
                 - "vq_loss": loss for the vector quantization layer.
                 - "mse": mean loss for all batch elements.
                 - "ts": a 1-D float tensor of the timesteps per batch entry.
                 - "mses": a 1-D tensor of the mean MSE losses per batch entry.
        """
        encoder_out = self.encoder(inputs, **extra_kwargs)
        if jitter:
            encoder_out = jitter_seq(encoder_out, jitter)
        vq_out = self.vq(encoder_out)
        vq_loss = vq_loss(encoder_out, vq_out["embedded"], self.vq.dictionary)

        ts = torch.rand(inputs.shape[0]).to(inputs)
        epsilon = torch.randn_like(inputs)
        noised_inputs = self.diffusion.sample_q(inputs, ts, epsilon=epsilon)
        predictions = self.predictor(
            noised_inputs, ts, cond=vq_out["passthrough"], labels=labels, **extra_kwargs
        )
        mses = ((predictions - epsilon) ** 2).flatten(1).mean(1)
        mse = mses.mean()

        return {"vq_loss": vq_loss, "mse": mse, "ts": ts, "mses": mses}

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encode a waveform as discrete symbols.

        :param inputs: an [N x 1 x T] audio Tensor.
        :return: an [N x T1] Tensor of latent codes.
        """
        with torch.no_grad():
            return self.vq(self.encoder(inputs))["idxs"]

    def decode(
        self,
        codes: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        steps: int = 100,
        progress: bool = False,
        constrain: bool = False,
    ) -> torch.Tensor:
        """
        Sample the decoder using encoded audio and corresponding labels.

        :param codes: an [N x T1] Tensor of latent codes or an [N x C x T1]
                      Tensor of latent code embeddings.
        :param labels: an [N] Tensor of integer labels.
        :param steps: number of diffusion steps.
        :param progress: if True, show a progress bar with tqdm.
        :param key: the key from predictions() to use as a predictor.
        :return: an [N x 1 x T] Tensor of audio.
        """
        if len(codes.shape) == 2:
            cond_seq = self.vq.embed(codes)
        elif len(codes.shape) == 3:
            cond_seq = codes
        else:
            raise ValueError(f"unsupported codes shape: {codes.shape}")

        x_T = torch.randn(
            codes.shape[0], 1, codes.shape[-1] * self.encoder.downsample_rate
        ).to(codes.device)
        return self.diffusion.ddpm_sample(
            x_T,
            lambda xs, ts, **kwargs: self.predictor(
                xs, ts, cond=cond_seq, labels=labels, **kwargs
            ),
            steps=steps,
            progress=progress,
            constrain=constrain,
        )

    @property
    def downsample_rate(self) -> int:
        """
        Get the minimum divisor required for input sequences.
        """
        # Naive lowest common multiple.
        x, y = super().downsample_rate, self.encoder.downsample_rate
        return next(i for i in range(x * y) if i % x == 0 and i % y == 0)

    def save_kwargs(self) -> Dict[str, Any]:
        res = super().save_kwargs()
        res.update(dict(enc_name=self.enc_name, cond_mult=self.cond_mult))
        return res


def jitter_seq(seq: torch.Tensor, p: float) -> torch.Tensor:
    """
    Apply temporal jitter to a latent sequence.

    This regularization technique was proposed in
    https://arxiv.org/abs/1901.08810.

    :param seq: an [N x C x T] Tensor.
    :param p: probability of a timestep being replaced.
    """
    right_shifted = torch.cat([seq[..., :1], seq[..., :-1]], dim=-1)
    left_shifted = torch.cat([seq[..., 1:], seq[..., -1:]], dim=-1)
    nums = torch.rand(seq.shape[0], 1, seq.shape[-1]).to(seq.device)

    return torch.where(
        nums < p / 2,
        right_shifted,
        torch.where(nums < p, left_shifted, seq),
    )
