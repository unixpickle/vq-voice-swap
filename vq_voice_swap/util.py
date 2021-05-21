import os
import tempfile
from typing import Any, Iterable, Iterator

import numpy as np
import torch
import torch.nn as nn


def repeat_dataset(data_loader: Iterable) -> Iterator:
    while True:
        yield from data_loader


def count_params(model: nn.Module) -> int:
    return sum(np.prod(x.shape) for x in model.parameters())


def atomic_save(state: Any, path: str):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_file = os.path.join(tmp_dir, "out.pt")
        torch.save(state, tmp_file)
        os.rename(tmp_file, path)