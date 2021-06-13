"""
Train a voice classifier on noised inputs.
"""

from vq_voice_swap.train_loop import ClassifierTrainLoop


def main():
    ClassifierTrainLoop().loop()


if __name__ == "__main__":
    main()
