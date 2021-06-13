"""
Train an VQ-VAE + diffusion model on waveforms.
"""

from vq_voice_swap.train_loop import VQVAETrainLoop


def main():
    VQVAETrainLoop().loop()


if __name__ == "__main__":
    main()
