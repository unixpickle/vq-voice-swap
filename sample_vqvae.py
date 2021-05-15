"""
Encode and decode a sample from a VQ-VAE.
"""

import argparse

import torch

from vq_voice_swap.dataset import ChunkReader, ChunkWriter
from vq_voice_swap.vq_vae import CascadeWaveGradVQVAE


def main():
    args = arg_parser().parse_args()

    print("loading model from checkpoint...")
    model = CascadeWaveGradVQVAE.load(args.checkpoint_path)
    assert args.label < model.num_labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"loading waveform from {args.input_file}...")
    reader = ChunkReader(args.input_file, sample_rate=args.sample_rate)
    try:
        chunk = reader.read(args.seconds * args.sample_rate)
    finally:
        reader.close()
    in_seq = torch.from_numpy(chunk[None, None]).to(device)

    print("encoding audio sequence...")
    encoded = model.encode(in_seq)

    print("decoding audio samples...")
    labels = torch.tensor([args.label]).long().to(device)
    sample = model.decode(encoded, labels, steps=args.sample_steps, progress=True)
    sample = sample.clamp(-1, 1).cpu().numpy().flatten()

    print(f"saving result to {args.output_file}...")
    writer = ChunkWriter(args.output_file, sample_rate=args.sample_rate)
    try:
        writer.write(sample)
    finally:
        writer.close()


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--sample-steps", type=int, default=100)
    parser.add_argument("--seconds", type=int, default=4)
    parser.add_argument("--label", type=int, default=0)
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("input_file", type=str)
    parser.add_argument("output_file", type=str)
    return parser


if __name__ == "__main__":
    main()
