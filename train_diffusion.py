"""
Train an diffusion model on waveforms.
"""

from vq_voice_swap.train_loop import DiffusionTrainLoop


def main():
    DiffusionTrainLoop().loop()


if __name__ == "__main__":
    main()
