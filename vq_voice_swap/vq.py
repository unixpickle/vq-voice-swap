"""
Implementation of the vector quantization step in a VQ-VAE.

Code adapted from: https://github.com/unixpickle/vq-vae-2/blob/6874db74dbc8e7a24c33303c0aa12d66d803c725/vq_vae_2/vq.py
"""

import random
from typing import Callable, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQLoss(nn.Module):
    """
    A special loss function for training a VQ layer.
    """

    def __init__(self, commitment: float = 0.25):
        super().__init__()
        self.commitment = commitment

    def forward(self, inputs: torch.Tensor, embedded: torch.Tensor) -> torch.Tensor:
        """
        Compute the codebook and commitment losses for a VQ layer.

        :param inputs: inputs to the VQ layer.
        :param embedded: outputs from the VQ layer.
        :return: a scalar Tensor loss term.
        """
        codebook_loss = ((inputs.detach() - embedded) ** 2).mean()
        comm_loss = ((inputs - embedded.detach()) ** 2).mean()
        return codebook_loss + self.commitment * comm_loss


class VQ(nn.Module):
    """
    A vector quantization layer.

    Inputs are Tensors of shape [N x C x ...].
    Outputs include an embedded version of the input Tensor of the same shape,
    a quantized, discrete [N x ...] Tensor, and other losses.

    :param num_channels: the depth of the input Tensors.
    :param num_codes: the number of codebook entries.
    :param dead_rate: the number of forward passes after which a dictionary
                      entry is considered dead if it has not been used.
    """

    def __init__(self, num_channels: int, num_codes: int, dead_rate: int = 100):
        super().__init__()
        self.num_channels = num_channels
        self.num_codes = num_codes
        self.dead_rate = dead_rate

        self.dictionary = nn.Parameter(torch.randn(num_codes, num_channels))
        self.register_buffer("usage_count", dead_rate * torch.ones(num_codes).long())
        self._last_batch = None  # used for revival

    def embed(self, idxs: torch.Tensor) -> torch.Tensor:
        """
        Convert encoded indices into embeddings.

        :param idxs: an [N x ...] Tensor.
        :return: an [N x C x ...] Tensor with gradients to the dictionary.
        """
        batch_size = idxs.shape[0]
        new_shape = (batch_size, self.num_channels, *idxs.shape[1:])
        idxs = idxs.view(batch_size, -1)
        embedded = F.embedding(idxs, self.dictionary)
        embedded = embedded.permute(0, 2, 1).reshape(new_shape)
        return embedded

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Apply vector quantization.

        If the module is in training mode, this will also update the usage
        tracker and re-initialize dead dictionary entries.

        :param inputs: an [N x C x ...] Tensor.
        :return: a dict containing the following keys:
                 - "embedded": a Tensor like inputs whose gradients flow to
                               the dictionary.
                 - "passthrough": a Tensor like inputs whose gradients are
                                  passed through to inputs.
                 - "idxs": an [N x ...] integer Tensor of code indices.
        """
        idxs_shape = (inputs.shape[0], *inputs.shape[2:])
        x, unflatten_fn = flatten_channels(inputs)

        diffs = embedding_distances(self.dictionary, x)
        idxs = torch.argmin(diffs, dim=-1)
        embedded = self.embed(idxs)
        passthrough = embedded.detach() + (x - x.detach())

        if self.training:
            self._update_tracker(idxs)
            self._last_batch = x.detach()

        return {
            "embedded": unflatten_fn(embedded),
            "passthrough": unflatten_fn(passthrough),
            "idxs": idxs.reshape(idxs_shape),
        }

    def revive_dead_entries(self):
        """
        Use the dictionary usage tracker to re-initialize entries that aren't
        being used often.

        Uses statistics from the previous call to forward() to revive centers.
        Thus, forward() must have been called at least once.
        """
        assert (
            self._last_batch is not None
        ), "cannot revive dead entries until a batch has been run"
        inputs = self._last_batch

        counts = self.usage_count.detach().cpu().numpy()
        new_dictionary = None
        inputs_numpy = None
        for i, count in enumerate(counts):
            if count:
                continue
            if new_dictionary is None:
                new_dictionary = self.dictionary.detach().cpu().numpy()
            if inputs_numpy is None:
                inputs_numpy = inputs.detach().cpu().numpy()
            new_dictionary[i] = random.choice(inputs_numpy)
            counts[i] = self.dead_rate
        if new_dictionary is not None:
            dict_tensor = torch.from_numpy(new_dictionary).to(self.dictionary)
            counts_tensor = torch.from_numpy(counts).to(self.usage_count)
            with torch.no_grad():
                self.dictionary.copy_(dict_tensor)
            self.usage_count.copy_(counts_tensor)

    def _update_tracker(self, idxs):
        raw_idxs = set(idxs.detach().cpu().numpy().flatten())
        update = -np.ones([self.num_codes], dtype=np.int)
        for idx in raw_idxs:
            update[idx] = self.dead_rate
        self.usage_count.add_(torch.from_numpy(update).to(self.usage_count))
        self.usage_count.clamp_(0, self.dead_rate)


def embedding_distances(dictionary: torch.Tensor, tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute distances between every embedding in a
    dictionary and every vector in a Tensor.

    This will not generate a huge intermediate Tensor,
    unlike the naive implementation.

    :param dictionary: a [D x C] Tensor.
    :param tensor: an [N x C] Tensor.

    :return: an [N x D] Tensor of distances.
    """
    dict_norms = torch.sum(torch.pow(dictionary, 2), dim=-1)
    tensor_norms = torch.sum(torch.pow(tensor, 2), dim=-1)

    # Work-around for https://github.com/pytorch/pytorch/issues/18862.
    exp_tensor = tensor[..., None].view(-1, tensor.shape[-1], 1)
    exp_dict = dictionary[None].expand(exp_tensor.shape[0], *dictionary.shape)
    dots = torch.bmm(exp_dict, exp_tensor)[..., 0]
    dots = dots.view(*tensor.shape[:-1], dots.shape[-1])

    return -2 * dots + dict_norms + tensor_norms[..., None]


def flatten_channels(
    x: torch.Tensor,
) -> Tuple[torch.Tensor, Callable[[torch.Tensor], torch.Tensor]]:
    """
    Turn an [N x C x ...] Tensor into a [B x C] Tensor.

    :return: a tuple (new_tensor, reverse_fn). The reverse_fn can be applied
             to a [B x C] Tensor to get back an [N x C x ...] Tensor.
    """
    in_shape = x.shape
    batch, channels = in_shape[:2]
    x = x.view(batch, channels, -1)
    x = x.permute(0, 2, 1)
    permuted_shape = x.shape
    x = x.reshape(-1, channels)

    def reverse_fn(y):
        return y.reshape(permuted_shape).permute(0, 2, 1).reshape(in_shape)

    return x, reverse_fn
