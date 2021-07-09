from typing import Iterable, Iterator

import torch.nn as nn


def repeat_dataset(data_loader: Iterable) -> Iterator:
    while True:
        yield from data_loader


def count_params(model: nn.Module) -> int:
    return sum(x.numel() for x in model.parameters())
