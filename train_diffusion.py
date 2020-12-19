"""
Train an unconditional diffusion model on waveforms.
"""

import argparse
import os

import torch
from torch.optim import AdamW

from vq_voice_swap.dataset import create_data_loader
from vq_voice_swap.diffusion import Diffusion
from vq_voice_swap.predictor import WaveGradModel
from vq_voice_swap.schedule import ExpSchedule


def main():
    args = arg_parser().parse_args()

    diffusion = Diffusion(ExpSchedule())
    model = WaveGradModel()

    if os.path.exists(args.checkpoint_path):
        print("loading from checkpoint...")
        model.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = AdamW(model.parameters(), lr=args.lr)

    data_loader, _ = create_data_loader(
        directory=args.data_dir, batch_size=args.batch_size
    )
    for i, data_batch in enumerate(data_loader):
        audio_seq = data_batch["samples"][:, None].to(device)
        ts = torch.rand(args.batch_size, device=device)
        noise = torch.randn_like(audio_seq)
        samples = diffusion.sample_q(audio_seq, ts, epsilon=noise)
        loss = ((noise - model(samples, ts)) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

        step = i + 1
        print(f"step {step}: loss={loss.item()}")
        if step % args.save_interval == 0:
            tmp_file = args.checkpoint_path + ".tmp"
            torch.save(model.state_dict(), tmp_file)
            os.rename(tmp_file, args.checkpoint_path)


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--checkpoint-path", default="model_diffusion.pt", type=str)
    parser.add_argument("--save-interval", default=1000, type=int)
    parser.add_argument("data_dir", type=str)
    return parser


if __name__ == "__main__":
    main()
