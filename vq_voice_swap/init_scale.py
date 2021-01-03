import torch
import torch.nn as nn


def init_scale(module: nn.Module, *args, **kwargs):
    """
    Rescale the weights of every submodule of a module so that each submodule's
    output has unit variance.
    """
    hooks = {}

    def hook_fn(self, inputs, outputs):
        std = outputs.std().item()
        if std < 1e-5:
            # Assume this is intentionally zero.
            return outputs
        self.weight.mul_(1 / std)
        hooks[self].remove()
        del hooks[self]
        return self(*inputs)

    def create_hook(module):
        if hasattr(module, "weight"):
            hooks[module] = module.register_forward_hook(hook_fn)

    module.apply(create_hook)

    with torch.no_grad():
        module(*args, **kwargs)
    for hook in hooks.values():
        hook.remove()
