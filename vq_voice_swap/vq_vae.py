from typing import Any, Optional

import torch

from .diffusion_model import DiffusionModel
from .models import WaveGradEncoder, predictor_downsample_rate
from .vq import VQ, VQLoss


class VQVAE(DiffusionModel):
    """
    A waveform VQ-VAE with a diffusion decoder.
    """

    def __init__(self, base_channels: int, **kwargs):
        encoder = WaveGradEncoder(base_channels=base_channels)
        kwargs["cond_channels"] = encoder.cond_channels
        super().__init__(base_channels=base_channels, **kwargs)
        self.encoder = encoder
        self.vq = VQ(self.encoder.cond_channels, 512)
        self.vq_loss = VQLoss()

    def losses(
        self,
        inputs: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        **extra_kwargs: Any
    ):
        """
        Compute losses for training the VQVAE.

        :param inputs: the input [N x 1 x T] audio Tensor.
        :param labels: an [N] Tensor of integer labels.
        :return: a dict containing the following keys:
                 - "vq_loss": loss for the vector quantization layer.
                 - "mse": mean loss for all batch elements.
                 - "ts": a 1-D float tensor of the timesteps per batch entry.
                 - "mses": a 1-D tensor of the mean MSE losses per batch entry.
        """
        encoder_out = self.encoder(inputs, **extra_kwargs)
        vq_out = self.vq(encoder_out)
        vq_loss = self.vq_loss(encoder_out, vq_out["embedded"])

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
    ) -> torch.Tensor:
        """
        Sample the decoder using encoded audio and corresponding labels.

        :param codes: an [N x T1] Tensor of latent codes.
        :param labels: an [N] Tensor of integer labels.
        :param steps: number of diffusion steps.
        :param progress: if True, show a progress bar with tqdm.
        :param key: the key from predictions() to use as a predictor.
        :return: an [N x 1 x T] Tensor of audio.
        """
        cond_seq = self.vq.embed(codes)
        x_T = torch.randn(
            codes.shape[0], 1, codes.shape[1] * self.downsample_rate()
        ).to(codes.device)
        return self.diffusion.ddpm_sample(
            x_T,
            lambda xs, ts, **kwargs: self.predictor(
                xs, ts, cond=cond_seq, labels=labels, **kwargs
            ),
            steps=steps,
            progress=progress,
        )

    def downsample_rate(self):
        """
        Get the number of audio samples per latent code.
        """
        return predictor_downsample_rate(self.pred_nae)
