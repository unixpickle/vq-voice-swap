"""
Train an unconditional diffusion model on waveforms.
"""

import argparse

import torch

from vq_voice_swap.dataset import ChunkWriter
from vq_voice_swap.diffusion import Diffusion
from vq_voice_swap.model import WaveGradPredictor
from vq_voice_swap.schedule import ExpSchedule


def main():
    args = arg_parser().parse_args()

    diffusion = Diffusion(ExpSchedule())
    model = WaveGradPredictor()

    model.load_state_dict(torch.load(args.checkpoint_path, map_location="cpu"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    x_T = torch.randn(1, 1, 64000, device=device)
    sample = diffusion.ddpm_sample(x_T, model, args.sample_steps, progress=True)

    writer = ChunkWriter(args.sample_path, 16000)
    writer.write(sample.view(-1).cpu().numpy())
    writer.close()


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sample-steps", default=100, type=int)
    parser.add_argument("--checkpoint-path", default="model_diffusion.pt", type=str)
    parser.add_argument("--sample-path", default="sample.wav", type=str)
    return parser


if __name__ == "__main__":
    main()
