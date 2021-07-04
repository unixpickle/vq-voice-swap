from typing import Optional

from .base import Encoder, Predictor
from .conv_encoder import ConvMFCCEncoder
from .unet import UNetEncoder, UNetPredictor
from .wavegrad import WaveGradEncoder, WaveGradPredictor


def make_predictor(
    pred_name: str,
    base_channels: int = 32,
    num_labels: Optional[int] = None,
    cond_channels: Optional[int] = None,
    dropout: float = 0.0,
) -> Predictor:
    """
    Create a Predictor model from a human-readable name.
    """
    if pred_name == "wavegrad":
        assert not dropout, "dropout not supported for wavegrad"
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
            dropout=dropout,
        )
    else:
        raise ValueError(f"unknown predictor: {pred_name}")


def make_encoder(
    enc_name: str,
    base_channels: int = 32,
    cond_mult: int = 16,
) -> Encoder:
    """
    Create an Encoder model from a human-readable name.
    """
    if enc_name == "wavegrad":
        return WaveGradEncoder(cond_mult=cond_mult, base_channels=base_channels)
    elif enc_name == "unet":
        return UNetEncoder(
            base_channels=base_channels, out_channels=base_channels * cond_mult
        )
    elif enc_name == "unet128":
        # Like unet, but with downsample rate 128 rather than 256.
        return UNetEncoder(
            base_channels=base_channels,
            channel_mult=(1, 1, 2, 2, 2, 4, 4, 8),
            out_channels=base_channels * cond_mult,
        )
    elif enc_name == "unet128-dilated":
        return UNetEncoder(
            base_channels=base_channels,
            channel_mult=(1, 1, 2, 2, 2, 4, 4, 8),
            out_dilations=(4, 8, 16, 32),
            out_channels=base_channels * cond_mult,
        )
    elif enc_name == "conv-mfcc-ulaw":
        return ConvMFCCEncoder(
            base_channels=base_channels, out_channels=base_channels * cond_mult
        )
    elif enc_name == "conv-mfcc-linear":
        return ConvMFCCEncoder(
            base_channels=base_channels,
            out_channels=base_channels * cond_mult,
            input_ulaw=False,
        )
    else:
        raise ValueError(f"unknown encoder: {enc_name}")
