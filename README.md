# vq-voice-swap

This is an ongoing experiment with diffusion models for speech generation. It includes scripts for training and evaluating diffusion models on speech datasets like [LibriSpeech](https://www.openslr.org/12).

This project initially started out as an experiment in using [VQ-VAE](https://arxiv.org/abs/1711.00937) + a [diffusion model](https://arxiv.org/abs/2006.11239) for speaker conversion. The initial results were discouraging: while the VQ-VAE encoding/decoding worked well, the speaker conversion aspect did not work, possibly because the diffusion model wasn't good enough to capture speaker information on its own.

# What's Included

This codebase includes data loaders for LibriSpeech, scripts to train diffusion models and classifiers, and some initial experiments with VQ-VAE.

You can train unconditional diffusion models on speech data using [train_diffusion.py](train_diffusion.py). The resulting diffusion models can then be sampled via [sample_diffusion.py](sample_diffusion.py).

You can train classifiers using [train_classifier.py](train_classifier.py). Classifiers can be used with `sample_diffusion.py` to potentially improve sample quality, as done in [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233).

# Evaluations

For generative modeling, loss or log-likelihood don't necessarily correspond well with sample quality. To evaluate sample quality, it is preferrable to use perception-aware evaluation metrics, typically by leveraging pre-trained models.

I have prepared evaluation metrics similar to [Inception Score](https://github.com/openai/improved-gan/tree/master/inception_score) and [FID](https://github.com/bioinf-jku/TTUR), but for speech segments rather than images:

 * **Class score** is similar to [Inception Score](https://github.com/openai/improved-gan/tree/master/inception_score), and higher is better. This should be taken as a measure of individual sample quality and speaker coverage.
 * **Frechet score** is similar to [FID](https://github.com/bioinf-jku/TTUR), and lower is better. Frechet score measures both fidelity and diversity, or more generally how well two distributions match.

These evals use a [pre-trained speaker classifier](http://data.aqnichol.com/vq-voice-swap/eval/) to extract features. For evaluating a diffusion model, you must first generate a directory of 10k samples using the `sample_diffusion.py` script. Next, download the [pre-trained classifier](http://data.aqnichol.com/vq-voice-swap/eval/model_classifier.pt), and run [stat_generate.py](stat_generate.py) on your samples to gather statistics and compute a class score. Then you can generate or [download](http://data.aqnichol.com/vq-voice-swap/eval/train_clean_360.npz) statistics for the training set, and run [stat_compare.py](stat_compare.py) to compute the Frechet score.

# Results

Here are all of the models I've trained (and released), with their corresponding evals. Each model links to a directory with samples, evaluation statistics, and the model checkpoints:

 * [unet32](http://data.aqnichol.com/vq-voice-swap/unet32): a 10M parameter UNet model with the default noise schedule. For this model, I sampled with 50 steps using a sample-time schedule `t = s^2` where `s` is linearly spaced.
  * Class score: 47.1
  * Frechet score: 2494
 * [unet64](http://data.aqnichol.com/vq-voice-swap/unet64/): a 50M parameter model which is otherwise similar to the unet32, but with some learning rate annealing at the end of training.
  * Class score: 69.0
  * Frechet score: 1834
 * [unet64/early_stopped](http://data.aqnichol.com/vq-voice-swap/unet64/early_stopped/): like unet64, but *without* learning rate annealing. Surprisingly, the Frechet score is much better, suggesting some kind of overfitting.
  * Class score: 51.5
  * Frechet score: 855
