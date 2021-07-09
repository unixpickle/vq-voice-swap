"""
Add classes to a pre-trained VQVAE from a new dataset.
"""

from vq_voice_swap.train_loop import VQVAEAddClassesTrainLoop


def main():
    VQVAEAddClassesTrainLoop().loop()


if __name__ == "__main__":
    main()
