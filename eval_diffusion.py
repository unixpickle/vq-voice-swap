"""
Evaluate how well a diffusion model performs.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

from vq_voice_swap.dataset import create_data_loader
from vq_voice_swap.diffusion_model import DiffusionModel
from vq_voice_swap.loss_tracker import LossTracker
from vq_voice_swap.vq_vae import make_predictor


def main():
    args = arg_parser().parse_args()

    data_loader, _ = create_data_loader(
        directory=args.data_dir, batch_size=args.batch_size
    )

    print("loading model from checkpoint...")
    model = DiffusionModel.load(args.checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    tracker = LossTracker(avg_size=1_000_000)

    num_samples = 0
    for data_batch in data_loader:
        audio_seq = data_batch["samples"][:, None].to(device)
        ts = torch.rand(args.batch_size, device=device)
        noise = torch.randn_like(audio_seq)
        samples = model.diffusion.sample_q(audio_seq, ts, epsilon=noise)
        with torch.no_grad():
            noise_pred = model.predictor(samples, ts)
        losses = ((noise - noise_pred) ** 2).flatten(1).mean(dim=1)

        tracker.add(ts, losses)
        log_dict = tracker.log_dict()

        num_samples += len(ts)

        msg = " ".join([f"{key}={value:.06f}" for key, value in log_dict.items()])
        print(f"{num_samples} samples: {msg}")


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("data_dir", type=str)
    return parser


if __name__ == "__main__":
    main()
