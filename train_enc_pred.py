"""
Train a model to predict the latents from a VQVAE encoder from noised audio
samples.
"""

from vq_voice_swap.train_loop import EncoderPredictorTrainLoop


def main():
    EncoderPredictorTrainLoop().loop()


if __name__ == "__main__":
    main()
