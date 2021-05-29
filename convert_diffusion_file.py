"""
Convert raw diffusion model checkpoints to the new format.
"""

import argparse

import torch

from vq_voice_swap.diffusion_model import DiffusionModel


def main():
    args = arg_parser().parse_args()

    model = DiffusionModel(
        pred_name=args.predictor,
        base_channels=args.base_channels,
        schedule_name=args.schedule,
    )
    model.predictor.load_state_dict(torch.load(args.in_path, map_location="cpu"))
    model.save(args.out_path)


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--predictor", default="wavegrad", type=str)
    parser.add_argument("--base-channels", default=32, type=int)
    parser.add_argument("--schedule", default="exp", type=str)
    parser.add_argument("in_path")
    parser.add_argument("out_path")
    return parser


if __name__ == "__main__":
    main()