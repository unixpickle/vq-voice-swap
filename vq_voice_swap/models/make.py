from typing import Optional

from .base import Predictor
from .unet import UNetPredictor
from .wavegrad import WaveGradPredictor


def make_predictor(
    pred_name: str,
    base_channels: int = 32,
    num_labels: Optional[int] = None,
    cond_channels: Optional[int] = None,
) -> Predictor:
    """
    Create a Predictor model from a human-readable name.
    """
    if pred_name == "wavegrad":
        cond_mult = cond_channels // base_channels if cond_channels else 16
        return WaveGradPredictor(
            base_channels=base_channels,
            cond_mult=cond_mult,
            num_labels=num_labels,
        )
    elif pred_name == "unet":
        return UNetPredictor(
            base_channels=base_channels,
            cond_channels=cond_channels,
            num_labels=num_labels,
        )
    else:
        raise ValueError(f"unknown predictor: {pred_name}")


def predictor_downsample_rate(pred_name: str) -> int:
    """
    Get the downsample rate of a named Predictor, to ensure that input
    sequences are evenly divisible by it.
    """
    if pred_name == "wavegrad":
        return 2 ** 6
    elif pred_name == "unet":
        return 2 ** 8
    else:
        raise ValueError(f"unknown predictor: {pred_name}")