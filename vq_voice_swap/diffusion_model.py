from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

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

        # Fix bug in some checkpoints where dropout is a tuple.
        self.dropout = dropout[0] if isinstance(dropout, tuple) else dropout

        self.predictor = make_predictor(
            pred_name,
            base_channels=base_channels,
            cond_channels=cond_channels,
            num_labels=num_labels,
            dropout=self.dropout,
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

    def load_from_pretrained(self, model: nn.Module) -> int:
        """
        Load the available parameters from a model into self.
        This only copies the union of self and model.

        :return: the total number of parameters copied. In particular, this is
                 the sum of the product of the shapes of the parameters.
        """
        src_params = dict(model.named_parameters())
        dst_params = dict(self.named_parameters())
        total = 0
        for name, dst in dst_params.items():
            if name in src_params:
                dst.copy_(src_params[name])
                total += np.prod(dst.shape)
        return total
