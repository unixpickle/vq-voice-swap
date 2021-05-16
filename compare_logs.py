"""
Compare a value throughout training for multiple runs, plotted on the same
figure.
"""

import argparse
import os
import re

import matplotlib

matplotlib.use("agg")
import matplotlib.pyplot as plt

from vq_voice_swap.logger import read_log
from vq_voice_swap.smoothing import moving_average


def main():
    args = arg_parser().parse_args()

    for filename in args.log_files:
        name, _ = os.path.splitext(os.path.basename(filename))
        for field in args.fields:
            entries = [(step, field_value(x, field)) for step, x in read_log(filename)]
            entries = [(x, y) for x, y in entries if y is not None]
            xs, ys = tuple(zip(*entries))
            ys = moving_average(ys, args.smoothing)
            plt.plot(xs, ys, label=f"{name} {field}")
    plt.ylim(0, args.max_y)
    if args.max_x is not None:
        plt.xlim(0, args.max_x)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(args.out_file)


def field_value(log_entry, field_expr):
    values = [v for k, v in log_entry.items() if re.match(field_expr, k)]
    if len(values) == 0:
        return None
    return sum(values) / len(values)


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--smoothing", type=int, default=1)
    parser.add_argument("--max-x", type=float, default=None)
    parser.add_argument("--max-y", type=float, default=1.0)
    parser.add_argument("--fields", type=str, nargs="+", default="base_q.")
    parser.add_argument("log_files", nargs="+", type=str)
    parser.add_argument("out_file", type=str)
    return parser


if __name__ == "__main__":
    main()
