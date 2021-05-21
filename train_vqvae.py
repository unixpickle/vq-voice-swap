"""
Train an VQ-VAE + diffusion model on waveforms.
"""

import argparse
import os

import numpy as np
import torch
from torch.optim import AdamW

from vq_voice_swap.dataset import create_data_loader
from vq_voice_swap.ema import ModelEMA
from vq_voice_swap.logger import Logger
from vq_voice_swap.loss_tracker import LossTracker
from vq_voice_swap.util import count_params, repeat_dataset
from vq_voice_swap.vq_vae import CascadeVQVAE


def main():
    args = arg_parser().parse_args()

    data_loader, num_labels = create_data_loader(
        directory=args.data_dir, batch_size=args.batch_size
    )

    if os.path.exists(args.checkpoint_path):
        print("loading from checkpoint...")
        resume = True
        model = CascadeVQVAE.load(args.checkpoint_path)
        assert model.num_labels == num_labels
    else:
        print("creating new model...")
        resume = False
        model = CascadeVQVAE(
            args.predictor, num_labels, base_channels=args.base_channels
        )

    if args.pretrained_base:
        print(f"loading and fusing pre-trained base model: {args.pretrained_base}")
        model.base_predictor.load_state_dict(torch.load(args.pretrained_base))
        for p in model.base_predictor.parameters():
            p.requires_grad_(False)

    print(f"total parameters: {count_params(model)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ema = ModelEMA(
        model,
        rates={
            "": args.ema_rate,
            "vq.": args.vq_ema_rate,
        },
    )

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    trackers = {key: LossTracker(prefix=f"{key}_") for key in ["base", "label", "cond"]}
    logger = Logger(args.log_file, resume=resume)

    for i, data_batch in enumerate(repeat_dataset(data_loader)):
        audio_seq = data_batch["samples"][:, None].to(device)
        labels = data_batch["label"].to(device)
        losses = model.losses(audio_seq, labels, use_checkpoint=args.grad_checkpoint)
        loss = losses["vq_loss"] + losses["mse"]

        opt.zero_grad()
        loss.backward()
        opt.step()
        ema.update()

        model.vq.revive_dead_entries()

        step = i + 1
        log_dict = {}
        for key, mses in losses["mses_dict"].items():
            trackers[key].add(losses["ts"], mses)
            log_dict.update(trackers[key].log_dict())
        logger.log(
            step, vq_loss=losses["vq_loss"].item(), mse=losses["mse"].item(), **log_dict
        )
        if step % args.save_interval == 0:
            model.save(args.checkpoint_path)
            ema.model.save(args.ema_path)


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--predictor", default="wavegrad", type=str)
    parser.add_argument("--base-channels", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--ema-rate", default=0.9999, type=float)
    parser.add_argument("--vq-ema-rate", default=0.99, type=float)
    parser.add_argument("--checkpoint-path", default="model_vqvae.pt", type=str)
    parser.add_argument("--pretrained-base", default=None, type=str)
    parser.add_argument("--ema-path", default="model_vqvae_ema.pt", type=str)
    parser.add_argument("--save-interval", default=500, type=int)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--log-file", default="train_log.txt")
    parser.add_argument("data_dir", type=str)
    return parser


if __name__ == "__main__":
    main()
