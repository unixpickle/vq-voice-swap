from abc import ABC, abstractmethod
import math

import torch


class Schedule(ABC):
    @abstractmethod
    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        """
        Evaluate alpha for the noise schedule to a timestep in [0, 1].
        """


class ExpSchedule(Schedule):
    """
    A noise schedule defined as exp(-k*t^2), which is nearly equivalent to
    using betas linearly interpolated from a tiny value to a larger value.
    """

    def __init__(self, alpha_final: float = 1e-5):
        super().__init__()
        self.alpha_final = alpha_final

        # alpha(t) = exp(-k*t^2)
        # alpha(1.0) = exp(-k)
        # k = -ln(alpha(1.0))
        self.k = -math.log(alpha_final)

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self.k * (t ** 2))


class CosSchedule(Schedule):
    """
    The squared cosine schedule cos(t*pi/2)^2, introduced by
    https://arxiv.org/abs/2102.09672.
    """

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        return torch.cos(t * math.pi / 2) ** 2
