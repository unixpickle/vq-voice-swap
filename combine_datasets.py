"""
Combine two LibriSpeech-like datasets into one directory with a shared index
and symbolic links to the sub-datasets.
"""

import argparse
import json
import os
import sys

from vq_voice_swap.dataset import LibriSpeech


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", type=str, nargs="+")
    parser.add_argument("output", type=str)
    args = parser.parse_args()

    if os.path.exists(args.output):
        print(f"error: output directory already exists: {args.output}")
        sys.exit(1)
    os.mkdir(args.output)

    full_index = {}
    for i, subdir in enumerate(args.directories):
        print(f"creating dataset for {subdir}...")
        dataset = LibriSpeech(subdir)
        prefix = f"{i:02}_"
        full_index.update({prefix + k: v for k, v in dataset.index.items()})
        for speaker_id in dataset.index.keys():
            os.symlink(
                os.path.join(subdir, speaker_id),
                os.path.join(args.output, prefix + speaker_id),
            )

    with open(os.path.join(args.output, "index.json"), "w") as f:
        json.dump(full_index, f)


if __name__ == "__main__":
    main()
