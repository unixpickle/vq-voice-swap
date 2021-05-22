from typing import Callable

import torch
from tqdm.auto import tqdm

from .schedule import Schedule


class Diffusion:
    """
    A PyTorch implementation of continuous-time diffusion.
    """

    def __init__(self, schedule: Schedule):
        self.schedule = schedule

    def sample_q(
        self, x_0: torch.Tensor, ts: torch.Tensor, epsilon: torch.Tensor = None
    ):
        """
        Sample from q(x_t | x_0) for a batch of x_0.
        """
        if epsilon is None:
            epsilon = torch.randn_like(x_0)
        alphas = broadcast_as(self.schedule(ts), x_0)
        return alphas.sqrt() * x_0 + (1 - alphas).sqrt() * epsilon

    def eps_to_x0(
        self, x_t: torch.Tensor, ts: torch.Tensor, epsilon_prediction: torch.Tensor
    ):
        """
        Evaluate the mean of p(x_0 | x_t), provided the model's epsilon
        prediction for x_t.
        """
        alphas = broadcast_as(self.schedule(ts), x_t)
        return (x_t - (1 - alphas).sqrt() * epsilon_prediction) * alphas.rsqrt()

    def x0_to_eps(self, x_t: torch.Tensor, ts: torch.Tensor, x_0: torch.Tensor):
        """
        Compute the inverse of eps_to_x0() with respect to epsilon, computing
        the epsilon which would have given an x_0 prediction.
        """
        alphas = broadcast_as(self.schedule(ts), x_t)
        return (x_t - x_0 * alphas.sqrt()) * (1 - alphas).rsqrt()

    def ddpm_previous(
        self,
        x_t: torch.Tensor,
        ts: torch.Tensor,
        step: float,
        epsilon_prediction: torch.Tensor,
        noise: torch.Tensor = None,
        sigma_large: bool = False,
        cond_fn: Callable = None,
    ):
        """
        Sample the previous timestep using reverse diffusion.
        """
        if noise is None:
            noise = torch.randn_like(x_t)
        alphas_t = broadcast_as(self.schedule(ts), x_t)
        alphas_prev = broadcast_as(self.schedule(ts - step), x_t)
        alphas = alphas_t / alphas_prev
        betas = 1 - alphas

        if not sigma_large:
            sigmas = betas * (1 - alphas_prev) / (1 - alphas_t)
        else:
            sigmas = betas

        mean_pred = alphas.rsqrt() * (
            x_t - betas * (1 - alphas_t).rsqrt() * epsilon_prediction
        )
        if cond_fn is not None:
            mean_pred = mean_pred + sigmas * cond_fn(mean_pred, ts - step)

        return mean_pred + sigmas.sqrt() * noise

    def ddpm_sample(
        self,
        x_T: torch.Tensor,
        predictor: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        steps: int,
        progress: bool = False,
        sigma_large: bool = False,
        constrain: bool = False,
        cond_fn: Callable = None,
    ):
        """
        Sample x_0 from x_t using reverse diffusion.
        """
        x_t = x_T
        ts = [(i + 1) / steps for i in range(steps)]
        t_step = 1 / steps

        its = enumerate(ts[::-1])
        if progress:
            its = tqdm(its)

        for i, t in its:
            ts = torch.tensor([t] * x_T.shape[0]).to(x_T)
            with torch.no_grad():
                eps = predictor(x_t, ts)
                if constrain:
                    x0 = self.eps_to_x0(x_t, ts, eps)
                    x0 = (x0 - x0.mean(dim=-1, keepdim=True)).clamp(-1, 1)
                    eps = self.x0_to_eps(x_t, ts, x0)
                x_t = self.ddpm_previous(
                    x_t=x_t,
                    ts=ts,
                    step=t_step,
                    epsilon_prediction=eps,
                    noise=torch.zeros_like(x_T) if i + 1 == steps else None,
                    sigma_large=sigma_large,
                    cond_fn=cond_fn,
                )

        return x_t


def broadcast_as(ts, tensor):
    while len(ts.shape) < len(tensor.shape):
        ts = ts[:, None]
    return ts.to(tensor) + torch.zeros_like(tensor)
