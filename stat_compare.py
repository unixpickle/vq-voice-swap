"""
Compare the statistics of two batch statistics, similar to the Frechet
Inception Distance.
"""

import argparse
import numpy as np
from scipy import linalg


def main():
    args = arg_parser().parse_args()
    stat1 = np.load(args.stat_1)
    stat2 = np.load(args.stat_2)
    print(frechet_distance(stat1["mean"], stat1["cov"], stat2["mean"], stat2["cov"]))


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # https://github.com/bioinf-jku/TTUR/blob/73ab375cdf952a12686d9aa7978567771084da42/fid.py#L132
    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; adding %s to diagonal of cov estimates"
            % eps
        )
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def arg_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("stat_1", type=str)
    parser.add_argument("stat_2", type=str)
    return parser


if __name__ == "__main__":
    main()
