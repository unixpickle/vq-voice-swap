"""
Train an unconditional diffusion model on waveforms.
"""

import argparse
import os

import numpy as np
import torch
from torch.optim import AdamW

from vq_voice_swap.dataset import create_data_loader
from vq_voice_swap.diffusion_model import DiffusionModel
from vq_voice_swap.ema import ModelEMA
from vq_voice_swap.logger import Logger
from vq_voice_swap.loss_tracker import LossTracker
from vq_voice_swap.util import count_params, repeat_dataset


def main():
    args = arg_parser().parse_args()

    if os.path.exists(args.checkpoint_path):
        print("loading from checkpoint...")
        model = DiffusionModel.load(args.checkpoint_path)
        resume = True
    else:
        print("creating new model")
        model = DiffusionModel(
            pred_name=args.predictor,
            base_channels=args.base_channels,
            schedule_name=args.schedule,
            dropout=args.dropout,
        )
        resume = False

    print(f"total parameters: {count_params(model)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ema = ModelEMA(model, rates={"": args.ema_rate})

    if os.path.exists(args.ema_path):
        print("loading EMA from checkpoint...")
        ema.model = DiffusionModel.load(args.ema_path).to(device)

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
        samples = model.diffusion.sample_q(audio_seq, ts, epsilon=noise)
        noise_pred = model.predictor(samples, ts, use_checkpoint=args.grad_checkpoint)
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
            model.save(args.checkpoint_path)
            ema.model.save(args.ema_path)


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--predictor", default="wavegrad", type=str)
    parser.add_argument("--base-channels", default=32, type=int)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--schedule", default="exp", type=str)
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
