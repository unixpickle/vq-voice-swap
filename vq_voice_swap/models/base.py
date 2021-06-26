from abc import abstractmethod
import functools
import os
import tempfile
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn as nn


class Predictor(nn.Module):
    @abstractmethod
    def forward(self, xs: torch.Tensor, ts: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Apply the epsilon predictor to a batch of noised inputs.
        """

    def condition(self, **kwargs) -> Callable:
        return functools.partial(self, **kwargs)


class Savable(nn.Module):
    """
    A module which saves constructor kwargs to reconstruct itself.
    """

    @abstractmethod
    def save_kwargs(self) -> Dict[str, Any]:
        """
        Get kwargs for restoring this model.
        """

    def save_dict(self) -> Dict[str, Any]:
        """
        Save a dict compatible with load_dict().
        """
        return {
            "kwargs": self.save_kwargs(),
            "state_dict": self.state_dict(),
        }

    @classmethod
    def load_dict(cls, state: Dict[str, Any]) -> Any:
        """
        Construct an object saved with save_dict().
        """
        obj = cls(**state["kwargs"])
        obj.load_state_dict(state["state_dict"])
        return obj

    def save(self, path: str):
        """
        Save this model to a file for loading with load().
        """
        atomic_save(self.save_dict(), path)

    @classmethod
    def load(cls, path: str):
        """
        Load a fresh model instance from a file created with save().
        """
        state = torch.load(path, map_location="cpu")
        return cls.load_dict(state)

    def load_from_pretrained(self, model: nn.Module) -> int:
        """
        Load the available parameters from a model into self.
        This only copies the union of self and model.

        :return: the total number of parameters copied. In particular, this is
                 the sum of the product of the shapes of the parameters.
        """
        src_params = dict(model.named_parameters())
        dst_params = dict(self.named_parameters())
        total = 0
        for name, dst in dst_params.items():
            if name in src_params:
                with torch.no_grad():
                    if dst.shape != src_params[name].shape:
                        raise RuntimeError(
                            f"Parameter {name} has shape {dst.shape} in destination "
                            f"but {src_params[name].shape} in source."
                        )
                    dst.copy_(src_params[name])
                total += np.prod(dst.shape)
        return total


def atomic_save(state: Any, path: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = os.path.join(tmp_dir, "out.pt")
        torch.save(state, tmp_file)
        os.rename(tmp_file, path)