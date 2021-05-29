"""
Generate feature statistics for a batch of samples using a classifier's
features. Also computes a class score similar to Inception Score.
"""

import argparse
import os
from typing import Iterable, Iterator, List, Optional

import numpy as np
import torch

from vq_voice_swap.classifier import Classifier
from vq_voice_swap.dataset import ChunkReader, create_data_loader, lookup_audio_duration


def main():
    args = arg_parser().parse_args()
    segments = load_segments(args)

    classifier = Classifier.load(args.checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)

    features = []
    logits = []
    for batch in batch_segments(args.batch_size, segments):
        ts = torch.zeros(len(batch)).to(device)
        batch = batch.to(device)
        with torch.no_grad():
            fv = classifier.stem(batch, ts)
            features.extend(fv.cpu().numpy())
            logits.extend(classifier.out(fv).cpu().numpy())

    # TODO: compute statistics for feature vectors
    # TODO: compute IS-like metric based on logits
    # TODO: save information to a npz file


def batch_segments(
    batch_size: int, segs: Iterator[torch.Tensor]
) -> Iterator[torch.Tensor]:
    batch = []
    for seg in segs:
        batch.append(seg)
        if len(batch) == batch_size:
            yield torch.stack(batch)[:, 1]
            batch = []
    if len(batch):
        yield torch.stack(batch)[:, 1]


def load_segments(args) -> Iterator[torch.Tensor]:
    if (args.data_dir is None and args.sample_dir is None) or (
        args.data_dir is not None and args.sample_dir is not None
    ):
        raise argparse.ArgumentError(
            message="must specify --data-dir or --sample-dir, but not both"
        )
    if args.data_dir is not None:
        loader = create_data_loader(args.data_dir, batch_size=1)
        return segments_from_loader(args.num_samples, loader)
    else:
        files = [
            os.path.join(args.sample_dir, x)
            for x in os.listdir(args.sample_dir)
            if not x.startswith(".") and x.endswith(".wav")
        ]
        if args.num_samples:
            files = files[: args.num_samples]
        return segments_from_files(files)


def segments_from_loader(
    limit: Optional[int], loader: Iterable[dict]
) -> Iterator[torch.Tensor]:
    i = 0
    for batch in loader:
        yield batch["samples"].view(-1)
        i += 1
        if limit and i >= limit:
            break


def segments_from_files(files: List[str]) -> Iterator[torch.Tensor]:
    for file in files:
        duration = lookup_audio_duration(file)
        cr = ChunkReader(file, sample_rate=16000)
        chunk = cr.read(16000 * int(duration + 2))  # make sure we read the whole file
        yield torch.from_numpy(chunk)


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint-path", default="model_classifier.pt", type=str)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-samples", default=None, type=None)
    parser.add_argument("--sample-dir", default=None, type=str)
    parser.add_argument("--data-dir", default=None, type=str)
    return parser


if __name__ == "__main__":
    main()
