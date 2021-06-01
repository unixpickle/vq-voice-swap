"""
Train an unconditional diffusion model on waveforms.
"""

import argparse
import math
import os

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from vq_voice_swap.models import Classifier
from vq_voice_swap.dataset import ChunkWriter
from vq_voice_swap.diffusion_model import DiffusionModel


def main():
    args = arg_parser().parse_args()

    schedule = eval(args.schedule)

    model = DiffusionModel.load(args.checkpoint_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    if args.classifier_path:
        classifier = Classifier.load(args.classifier_path).to(device)
        classifier.eval()

        def cond_fn(x, ts):
            if args.classifier_class is not None:
                target_class = (
                    torch.tensor([args.classifier_class] * len(x)).long().to(device)
                )
            else:
                target_class = torch.randint_like(ts, high=classifier.num_labels).long()
            with torch.enable_grad():
                x = x.detach().clone().requires_grad_()
                logits = classifier(x, ts, use_checkpoint=args.grad_checkpoint)
                logprobs = F.log_softmax(logits, dim=-1)
                grads = torch.autograd.grad(
                    logprobs[range(len(x)), target_class].sum(), x
                )[0]
                return grads.detach() * args.classifier_scale

    else:
        cond_fn = None

    if args.num_samples is None:
        generate_one_sample(
            args,
            model,
            device,
            constrain=args.constrain,
            cond_fn=cond_fn,
            schedule=schedule,
        )
    else:
        generate_many_samples(
            args,
            model,
            device,
            constrain=args.constrain,
            cond_fn=cond_fn,
            schedule=schedule,
        )


def generate_one_sample(args, model, device, **kwargs):
    x_T = torch.randn(1, 1, 64000, device=device)
    sample = model.diffusion.ddpm_sample(
        x_T, model.predictor, args.sample_steps, progress=True, **kwargs
    )

    writer = ChunkWriter(args.sample_path, 16000)
    writer.write(sample.view(-1).cpu().numpy())
    writer.close()


def generate_many_samples(args, model, device, **kwargs):
    os.mkdir(args.sample_path)

    num_batches = int(math.ceil(args.num_samples / args.batch_size))
    count = 0

    for _ in tqdm(range(num_batches)):
        x_T = torch.randn(args.batch_size, 1, 64000, device=device)
        sample = model.diffusion.ddpm_sample(
            x_T, model.predictor, args.sample_steps, progress=False, **kwargs
        )
        for seq in sample:
            if count == args.num_samples:
                break
            sample_path = os.path.join(args.sample_path, f"sample_{count:06}.wav")
            writer = ChunkWriter(sample_path, 16000)
            writer.write(seq.view(-1).cpu().numpy())
            writer.close()
            count += 1


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--checkpoint-path", default="model_diffusion.pt", type=str)
    parser.add_argument("--sample-steps", default=100, type=int)
    parser.add_argument("--batch-size", default=1, type=int)
    parser.add_argument("--constrain", action="store_true")
    parser.add_argument("--sample-path", default="sample.wav", type=str)
    parser.add_argument("--num-samples", default=None, type=int)
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--classifier-path", default=None, type=str)
    parser.add_argument("--classifier-scale", default=1.0, type=float)
    parser.add_argument("--classifier-class", default=None, type=int)
    parser.add_argument("--schedule", default="lambda t: t", type=str)
    return parser


if __name__ == "__main__":
    main()
