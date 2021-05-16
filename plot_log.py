"""
Plot the MSE over a run from its log file.
"""

import argparse

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

from vq_voice_swap.logger import read_log
from vq_voice_swap.smoothing import moving_average


def main():
    args = arg_parser().parse_args()
    entries = [(step, x["mse"]) for step, x in read_log(args.log_file)]
    xs, ys = list(zip(*entries))
    ys = moving_average(ys, args.smoothing)
    plt.plot(xs, ys)
    plt.ylim(0, args.max_y)
    plt.xlabel("step")
    plt.ylabel("mse")
    plt.savefig(args.out_file)


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--smoothing", type=int, default=100)
    parser.add_argument("--max-y", type=float, default=1.0)
    parser.add_argument("log_file", type=str)
    parser.add_argument("out_file", type=str)
    return parser


if __name__ == "__main__":
    main()
