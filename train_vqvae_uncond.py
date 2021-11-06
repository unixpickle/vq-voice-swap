"""
Fine-tune a pre-trained VQVAE to not always take VQ codes
or classes as input. Adds a new zero class that is
unconditional.
"""

from vq_voice_swap.train_loop import VQVAEUncondTrainLoop


def main():
    VQVAEUncondTrainLoop().loop()


if __name__ == "__main__":
    main()
