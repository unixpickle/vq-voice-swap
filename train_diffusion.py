"""
Train an unconditional diffusion model on waveforms.
"""

import argparse
import os

import numpy as np
import torch
from torch.optim import AdamW

from vq_voice_swap.dataset import create_data_loader
from vq_voice_swap.diffusion import Diffusion
from vq_voice_swap.ema import ModelEMA
from vq_voice_swap.logger import Logger
from vq_voice_swap.loss_tracker import LossTracker
from vq_voice_swap.schedule import ExpSchedule
from vq_voice_swap.util import atomic_save, count_params, repeat_dataset
from vq_voice_swap.vq_vae import make_predictor


def main():
    args = arg_parser().parse_args()

    diffusion = Diffusion(ExpSchedule())
    model = make_predictor(args.predictor, base_channels=args.base_channels)

    if os.path.exists(args.checkpoint_path):
        print("loading from checkpoint...")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))
        resume = True
    else:
        resume = False

    print(f"total parameters: {count_params(model)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ema = ModelEMA(
        model,
        rates={
            "": args.ema_rate,
        },
    )

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tracker = LossTracker()
    logger = Logger(args.log_file, resume=resume)

    data_loader, _ = create_data_loader(
        directory=args.data_dir, batch_size=args.batch_size
    )
    for i, data_batch in enumerate(repeat_dataset(data_loader)):
        audio_seq = data_batch["samples"][:, None].to(device)
        ts = torch.rand(args.batch_size, device=device)
        noise = torch.randn_like(audio_seq)
        samples = diffusion.sample_q(audio_seq, ts, epsilon=noise)
        noise_pred = model(samples, ts, use_checkpoint=args.grad_checkpoint)
        losses = ((noise - noise_pred) ** 2).flatten(1).mean(dim=1)
        loss = losses.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update()

        step = i + 1
        tracker.add(ts, losses)
        logger.log(step, mse=loss.item(), **tracker.log_dict())

        if step % args.save_interval == 0:
            atomic_save(model.state_dict(), args.checkpoint_path)
            atomic_save(ema.model.state_dict(), args.ema_path)


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--predictor", default="wavegrad", type=str)
    parser.add_argument("--base-channels", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--ema-rate", default=0.9999, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--checkpoint-path", default="model_diffusion.pt", type=str)
    parser.add_argument("--ema-path", default="model_diffusion_ema.pt", type=str)
    parser.add_argument("--save-interval", default=1000, type=int)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--log-file", default="train_diffusion_log.txt")
    parser.add_argument("data_dir", type=str)
    return parser


if __name__ == "__main__":
    main()
