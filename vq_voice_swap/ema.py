import copy
from typing import Dict

import torch
import torch.nn as nn


class ModelEMA:
    """
    An exponential moving average of model parameters.

    :param source_model: the non-EMA model whose parameters are copied.
    :param rates: a dict mapping parameter names (or prefixes of names)
                  to EMA rates. Any parameter not explicitly named in this
                  dict will use the rate from its longest prefix in the dict.
    """

    def __init__(self, source_model: nn.Module, rates: Dict[str, float]):
        self.source_model = source_model
        self.rates = rates
        self.model = copy.deepcopy(source_model)

    def update(self):
        """
        Update the EMA parameters based on the current source parameters.
        """
        for (name, source), target in zip(
            self.source_model.named_parameters(), self.model.parameters()
        ):
            rate = 1 - lookup_longest_prefix(self.rates, name)
            with torch.no_grad():
                target.add_(rate * (source - target))


def lookup_longest_prefix(values: Dict[str, float], name: str) -> float:
    longest = None
    for k in values.keys():
        if name.startswith(k) and (longest is None or len(k) > len(longest)):
            longest = k
    if longest is None:
        raise KeyError(f"no rate prefix found for parameter: {name}")
    return values[longest]
