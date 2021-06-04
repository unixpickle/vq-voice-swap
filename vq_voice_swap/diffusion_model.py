from typing import Any, Dict, Optional

import torch

from .diffusion import Diffusion, make_schedule
from .models import Savable, make_predictor


class DiffusionModel(Savable):
    """
    A diffusion model and its corresponding diffusion process.
    """

    def __init__(
        self,
        pred_name: str,
        base_channels: int,
        schedule_name: str = "exp",
        num_labels: Optional[int] = None,
        cond_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.pred_name = pred_name
        self.base_channels = base_channels
        self.schedule_name = schedule_name
        self.num_labels = num_labels
        self.cond_channels = cond_channels
        self.dropout = dropout

        self.predictor = make_predictor(
            pred_name,
            base_channels=base_channels,
            cond_channels=cond_channels,
            num_labels=num_labels,
            dropout=dropout,
        )
        self.diffusion = Diffusion(make_schedule(schedule_name))

    def forward(self, *args, **kwargs) -> torch.Tensor:
        return self.predictor(*args, **kwargs)

    def save_kwargs(self) -> Dict[str, Any]:
        return dict(
            pred_name=self.pred_name,
            base_channels=self.base_channels,
            schedule_name=self.schedule_name,
            num_labels=self.num_labels,
            cond_channels=self.cond_channels,
            dropout=self.dropout,
        )
