"""
Evaluate how much a VQ-VAE leverages labels by measuring how much worse the
loss becomes when the label is randomized.
"""

import argparse

import numpy as np
import torch

from vq_voice_swap.dataset import create_data_loader
from vq_voice_swap.vq_vae import WaveGradVQVAE


def main():
    args = arg_parser().parse_args()

    data_loader, num_labels = create_data_loader(
        directory=args.data_dir, batch_size=args.batch_size
    )

    print("loading model from checkpoint...")
    model = WaveGradVQVAE.load(args.checkpoint_path)
    assert model.num_labels == num_labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    true_mses = []
    rand_mses = []
    num_samples = 0
    for data_batch in data_loader:
        audio_seq = data_batch["samples"][:, None].to(device)
        labels = data_batch["label"].to(device)
        rand_labels = torch.randint_like(labels, high=num_labels)

        with torch.no_grad():
            encoder_out = model.encoder(audio_seq)
            vq_out = model.vq(encoder_out)["embedded"]

            ts = torch.rand(audio_seq.shape[0]).to(audio_seq)
            epsilon = torch.randn_like(audio_seq)
            noised_inputs = model.diffusion.sample_q(audio_seq, ts, epsilon=epsilon)
            true_predictions = model.predictor(
                noised_inputs, ts, cond=vq_out, labels=labels
            )
            true_mse = ((true_predictions - epsilon) ** 2).mean()
            true_mses.append(true_mse.item())
            rand_predictions = model.predictor(
                noised_inputs, ts, cond=vq_out, labels=rand_labels
            )
            rand_mse = ((rand_predictions - epsilon) ** 2).mean()
            rand_mses.append(rand_mse.item())

        num_samples += len(labels)
        print(
            f"{num_samples} samples: true_mse={np.mean(true_mse)} rand_mse={np.mean(rand_mse)}"
        )


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
