"""
Generate feature statistics for a batch of samples using a classifier's
features. Also computes a class score similar to Inception Score.
"""

import argparse
import multiprocessing as mp
import os
from typing import Iterable, Iterator, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from vq_voice_swap.dataset import ChunkReader, create_data_loader, lookup_audio_duration
from vq_voice_swap.models import Classifier


def main():
    args = arg_parser().parse_args()
    segments = load_segments(args)

    classifier = Classifier.load(args.checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)

    features = []
    probs = []
    for batch in batch_segments(args.batch_size, tqdm(segments)):
        ts = torch.zeros(len(batch)).to(device)
        batch = batch.to(device)
        with torch.no_grad():
            fv = classifier.stem(batch, ts)
            features.extend(fv.cpu().numpy())
            probs.extend(F.softmax(classifier.out(fv), dim=-1).cpu().numpy())

    features = np.stack(features, axis=0)
    probs = np.stack(probs, axis=0)

    mean = np.mean(features, axis=0)
    cov = np.cov(features, rowvar=False)

    # Based on inception score.
    # https://github.com/openai/improved-gan/blob/4f5d1ec5c16a7eceb206f42bfc652693601e1d5c/inception_score/model.py#L49
    kl = probs * (np.log(probs) - np.log(np.expand_dims(np.mean(probs, 0), 0)))
    kl = np.mean(np.sum(kl, 1))
    score = np.exp(kl)
    print(f"classifier score: {score}")

    np.savez(args.output_path, mean=mean, cov=cov, probs=probs, class_score=score)


def batch_segments(
    batch_size: int, segs: Iterator[torch.Tensor]
) -> Iterator[torch.Tensor]:
    batch = []
    for seg in segs:
        batch.append(seg)
        if len(batch) == batch_size:
            yield torch.stack(batch)[:, None]
            batch = []
    if len(batch):
        yield torch.stack(batch)[:, None]


def load_segments(args) -> Iterator[torch.Tensor]:
    if (args.data_dir is None and args.sample_dir is None) or (
        args.data_dir is not None and args.sample_dir is not None
    ):
        raise argparse.ArgumentError(
            message="must specify --data-dir or --sample-dir, but not both"
        )
    if args.data_dir is not None:
        loader, _ = create_data_loader(args.data_dir, batch_size=1)
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
    ctx = mp.get_context("spawn")
    with ctx.Pool(4) as pool:
        for x in pool.imap_unordered(_read_audio_file, files):
            yield torch.from_numpy(x)


def _read_audio_file(path: str) -> np.ndarray:
    duration = lookup_audio_duration(path)  # may not be precise
    cr = ChunkReader(path, sample_rate=16000)
    return cr.read(16000 * int(duration + 2))


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint-path", default="model_classifier.pt", type=str)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-samples", default=None, type=int)
    parser.add_argument("--sample-dir", default=None, type=str)
    parser.add_argument("--data-dir", default=None, type=str)
    parser.add_argument("output_path", type=str)
    return parser


if __name__ == "__main__":
    main()
