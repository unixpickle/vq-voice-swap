# vq-voice-swap

Using [VQ-VAE](https://arxiv.org/abs/1711.00937) + a [diffusion model](https://arxiv.org/abs/2006.11239) for speaker conversion. Unfortunately, while the VQ-VAE encoding/decoding works well, the speaker conversion aspect does not work, possibly because the diffusion model doesn't have long enough context or enough modeling power.
