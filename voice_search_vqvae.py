"""
Find the class label that minimizes the reconstruction error of an audio clip.

This can be seen as searching for the voice that best matches the actual
speaker of a clip.
"""

import argparse

import torch
from tqdm.auto import tqdm

from vq_voice_swap.dataset import ChunkReader, ChunkWriter
from vq_voice_swap.vq_vae import VQVAE


def main():
    args = arg_parser().parse_args()

    print("loading model from checkpoint...")
    model = VQVAE.load(args.checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"loading waveform from {args.input_file}...")
    reader = ChunkReader(
        args.input_file, sample_rate=args.sample_rate, encoding=args.encoding
    )
    try:
        chunk = reader.read(args.seconds * args.sample_rate)
    finally:
        reader.close()
    in_seq = torch.from_numpy(chunk[None, None]).to(device)

    print("encoding audio sequence...")
    encoded = model.vq.embed(model.encode(in_seq)).detach()

    print("evaluating all losses...")
    labels = (
        torch.tensor(
            [i for i in range(model.num_labels) for _ in range(args.num_timesteps)]
        )
        .long()
        .to(device)
    )
    ts = torch.linspace(
        0.0, 1.0, steps=args.num_timesteps, dtype=torch.float32, device=device
    ).repeat(model.num_labels)
    losses = (
        evaluate_losses(model, in_seq, labels, ts, encoded, args.batch_size)
        .reshape([-1, args.num_timesteps])
        .mean(-1)
        .cpu()
        .numpy()
        .tolist()
    )

    print(f"top {min(args.top_k, len(losses))} sorted losses")
    print("-------")
    id_loss = sorted(enumerate(losses), key=lambda x: x[1])
    for id, loss in id_loss[: args.top_k]:
        print(f"{id}\t\t{loss:.6f}")


def evaluate_losses(
    model: VQVAE,
    targets: torch.Tensor,
    labels: torch.Tensor,
    ts: torch.Tensor,
    encoded: torch.Tensor,
    batch_size: int,
):
    results = []

    # Fix a noise seed for every example to reduce variance
    epsilon = torch.randn_like(targets)

    for i in tqdm(range(0, len(labels), batch_size)):
        labels_mb = labels[i : i + batch_size]
        ts_mb = ts[i : i + batch_size]
        encoded_mb = encoded.repeat(len(ts_mb), 1, 1)
        targets_mb = targets.repeat(len(ts_mb), 1, 1)
        epsilon_mb = epsilon.repeat(len(ts_mb), 1, 1)

        noised_inputs = model.diffusion.sample_q(targets_mb, ts_mb, epsilon=epsilon_mb)
        with torch.no_grad():
            predictions = model.predictor(
                noised_inputs, ts_mb, cond=encoded_mb, labels=labels_mb
            )
            mses = ((predictions - epsilon) ** 2).flatten(1).mean(1)
            results.append(mses)

    return torch.cat(results)


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seconds", type=int, default=4)
    parser.add_argument("--encoding", type=str, default="linear")
    parser.add_argument("--num-timesteps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--input-file", type=str, default=None, required=True)
    parser.add_argument("checkpoint_path", type=str)
    return parser


if __name__ == "__main__":
    main()
