"""
Train an unconditional diffusion model on waveforms.
"""

import argparse
import math
import os
from functools import partial

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from vq_voice_swap.dataset import ChunkWriter
from vq_voice_swap.diffusion_model import DiffusionModel
from vq_voice_swap.models import Classifier


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

        def cond_fn(x, ts, labels=None):
            if labels is None:
                labels = sample_labels(args, classifier.num_labels, len(ts), ts.device)
            with torch.enable_grad():
                x = x.detach().clone().requires_grad_()
                logits = classifier(x, ts, use_checkpoint=args.grad_checkpoint)
                logprobs = F.log_softmax(logits, dim=-1)
                grads = torch.autograd.grad(logprobs[range(len(x)), labels].sum(), x)[0]
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


def generate_one_sample(args, model, device, cond_fn=None, **kwargs):
    x_T = torch.randn(1, 1, 64000, device=device)
    cond_pred, cond_fn = condition_on_sampled_labels(args, model, cond_fn, 1, device)
    sample = model.diffusion.ddpm_sample(
        x_T, cond_pred, args.sample_steps, progress=True, cond_fn=cond_fn, **kwargs
    )

    writer = ChunkWriter(args.sample_path, 16000, encoding=args.encoding)
    writer.write(sample.view(-1).cpu().numpy())
    writer.close()


def generate_many_samples(args, model, device, cond_fn=None, **kwargs):
    os.mkdir(args.sample_path)

    num_batches = int(math.ceil(args.num_samples / args.batch_size))
    count = 0

    for _ in tqdm(range(num_batches)):
        x_T = torch.randn(args.batch_size, 1, 64000, device=device)
        cond_pred, cond_fn_1 = condition_on_sampled_labels(
            args, model, cond_fn, args.batch_size, device
        )
        sample = model.diffusion.ddpm_sample(
            x_T,
            cond_pred,
            args.sample_steps,
            progress=False,
            cond_fn=cond_fn_1,
            **kwargs,
        )
        for seq in sample:
            if count == args.num_samples:
                break
            sample_path = os.path.join(args.sample_path, f"sample_{count:06}.wav")
            writer = ChunkWriter(sample_path, 16000, encoding=args.encoding)
            writer.write(seq.view(-1).cpu().numpy())
            writer.close()
            count += 1


def condition_on_sampled_labels(args, model, cond_fn, batch_size, device):
    if model.num_labels is None:
        return model.predictor, cond_fn
    labels = sample_labels(args, model.num_labels, batch_size, device)
    if cond_fn is not None:
        cond_fn = partial(cond_fn, labels=labels)
    return partial(model.predictor, labels=labels), cond_fn


def sample_labels(args, num_labels, batch_size, device):
    if args.target_class is not None:
        out = torch.tensor([args.target_class] * batch_size)
    else:
        out = torch.randint(low=0, high=num_labels, size=(batch_size,))
    return out.to(dtype=torch.long, device=device)


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
    parser.add_argument("--target-class", default=None, type=int)
    parser.add_argument("--schedule", default="lambda t: t", type=str)
    parser.add_argument("--encoding", default="linear", type=str)
    return parser


if __name__ == "__main__":
    main()
