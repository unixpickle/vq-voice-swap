"""
Train an unconditional diffusion model on waveforms.
"""

import argparse
import os

import torch
from torch.optim import AdamW

from vq_voice_swap.dataset import create_data_loader
from vq_voice_swap.ema import ModelEMA
from vq_voice_swap.logger import Logger
from vq_voice_swap.loss_tracker import LossTracker
from vq_voice_swap.vq_vae import WaveGradVQVAE


def main():
    args = arg_parser().parse_args()

    data_loader, num_labels = create_data_loader(
        directory=args.data_dir, batch_size=args.batch_size
    )

    if os.path.exists(args.checkpoint_path):
        print("loading from checkpoint...")
        resume = True
        model = WaveGradVQVAE.load(args.checkpoint_path)
        assert model.num_labels == num_labels
    else:
        print("creating new model...")
        resume = False
        model = WaveGradVQVAE(num_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ema = ModelEMA(
        model,
        rates={
            "": args.ema_rate,
            "vq.": args.vq_ema_rate,
        },
    )

    opt = AdamW(model.parameters(), lr=args.lr)
    lt = LossTracker()
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
        lt.add(losses["ts"], losses["mses"])
        logger.log(
            step,
            vq_loss=losses["vq_loss"].item(),
            mse=losses["mse"].item(),
            **lt.log_dict()
        )
        if step % args.save_interval == 0:
            model.save(args.checkpoint_path)
            ema.model.save(args.ema_path)


def repeat_dataset(data_loader):
    while True:
        yield from data_loader


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--ema-rate", default=0.9999, type=float)
    parser.add_argument("--vq-ema-rate", default=0.99, type=float)
    parser.add_argument("--checkpoint-path", default="model_vqvae.pt", type=str)
    parser.add_argument("--ema-path", default="model_vqvae_ema.pt", type=str)
    parser.add_argument("--save-interval", default=500, type=int)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--log-file", default="train_log.txt")
    parser.add_argument("data_dir", type=str)
    return parser


if __name__ == "__main__":
    main()
