"""
Train a voice classifier on noised inputs.
"""

import argparse
import os

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from vq_voice_swap.classifier import Classifier
from vq_voice_swap.dataset import create_data_loader
from vq_voice_swap.diffusion import Diffusion
from vq_voice_swap.logger import Logger
from vq_voice_swap.loss_tracker import LossTracker
from vq_voice_swap.schedule import ExpSchedule
from vq_voice_swap.util import count_params, repeat_dataset


def main():
    args = arg_parser().parse_args()

    data_loader, num_labels = create_data_loader(
        directory=args.data_dir, batch_size=args.batch_size
    )

    diffusion = Diffusion(ExpSchedule())

    if os.path.exists(args.checkpoint_path):
        print("loading from checkpoint...")
        model = Classifier.load(args.checkpoint_path)
        assert model.num_labels == num_labels
        resume = True
    else:
        print("creating new model...")
        model = Classifier(num_labels=num_labels, base_channels=args.base_channels)
        resume = False

    print(f"total parameters: {count_params(model)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tracker = LossTracker()
    logger = Logger(args.log_file, resume=resume)

    for i, data_batch in enumerate(repeat_dataset(data_loader)):
        audio_seq = data_batch["samples"][:, None].to(device)
        ts = torch.rand(args.batch_size, device=device)
        noise = torch.randn_like(audio_seq)
        samples = diffusion.sample_q(audio_seq, ts, epsilon=noise)
        logits = model(samples, ts, use_checkpoint=args.grad_checkpoint)
        targets = data_batch["label"].to(device)
        nlls = -F.log_softmax(logits, dim=-1)[range(len(targets)), targets]
        loss = nlls.mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        step = i + 1
        tracker.add(ts, nlls)
        logger.log(step, nll=loss.item(), **tracker.log_dict())

        if step % args.save_interval == 0:
            model.save(args.checkpoint_path)


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--base-channels", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight-decay", default=0.01, type=float)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--checkpoint-path", default="model_classifier.pt", type=str)
    parser.add_argument("--save-interval", default=1000, type=int)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--log-file", default="train_classifier_log.txt")
    parser.add_argument("data_dir", type=str)
    return parser


if __name__ == "__main__":
    main()
