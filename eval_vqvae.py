"""
Evaluate how much a VQ-VAE leverages labels by measuring how much worse the
loss becomes when the label is randomized.
"""

import argparse

import torch
import torch.nn as nn

from vq_voice_swap.dataset import create_data_loader
from vq_voice_swap.loss_tracker import LossTracker
from vq_voice_swap.vq_vae import ConcreteVQVAE


def main():
    args = arg_parser().parse_args()

    data_loader, num_labels = create_data_loader(
        directory=args.data_dir, batch_size=args.batch_size
    )

    print("loading model from checkpoint...")
    model = ConcreteVQVAE.load(args.checkpoint_path)
    assert model.num_labels == num_labels

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    trackers = {
        key: LossTracker(avg_size=1_000_000, prefix=f"{key}_") for key in ["cond"]
    }
    output_stats = [
        OutputStats(module, key) for key, module in (("cond", model.cond_predictor),)
    ]

    num_samples = 0
    for data_batch in data_loader:
        audio_seq = data_batch["samples"][:, None].to(device)
        labels = data_batch["label"].to(device)
        with torch.no_grad():
            losses = model.losses(audio_seq, labels)

        log_dict = {}
        for key, mses in losses["mses_dict"].items():
            trackers[key].add(losses["ts"], mses)
            log_dict.update(trackers[key].log_dict())
        for stat in output_stats:
            log_dict.update(stat.log_dict())

        num_samples += len(labels)

        msg = " ".join([f"{key}={value:.06f}" for key, value in log_dict.items()])
        print(f"{num_samples} samples: {msg}")


class OutputStats:
    def __init__(self, module: nn.Module, key: str):
        self.module = module
        self.key = key
        self.stds = LossTracker(prefix=f"{key}_std_")

        def hook(_module, inputs, output):
            self.stds.add(inputs[1], output.flatten(1).std(dim=1))

        self.module.register_forward_hook(hook)

    def log_dict(self):
        return self.stds.log_dict()


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
