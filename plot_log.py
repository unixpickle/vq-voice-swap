"""
Train an unconditional diffusion model on waveforms.
"""

import argparse

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np

from vq_voice_swap.logger import read_log


def main():
    args = arg_parser().parse_args()
    entries = [(step, x["mse"]) for step, x in read_log(args.log_file)]
    xs, ys = list(zip(*entries))
    ys = np.concatenate(
        [
            np.cumsum(ys)[: args.smoothing - 1] / (np.arange(args.smoothing - 1) + 1),
            np.convolve(ys, np.ones([args.smoothing]) / args.smoothing, mode="valid"),
        ]
    )
    plt.plot(xs, ys)
    plt.xlabel("step")
    plt.ylabel("mse")
    plt.savefig(args.out_file)


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--smoothing", type=int, default=100)
    parser.add_argument("log_file", type=str)
    parser.add_argument("out_file", type=str)
    return parser


if __name__ == "__main__":
    main()
