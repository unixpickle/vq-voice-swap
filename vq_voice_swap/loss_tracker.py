from typing import List, Optional

import numpy as np
import torch


class LossTracker:
    """
    Track loss averages throughout training to log separate
    quantiles of the diffusion loss function.
    """

    def __init__(self, quantiles: int = 4, avg_size: int = 1000):
        self.quantiles = quantiles
        self.avg_size = avg_size
        self.history = [[] for _ in range(quantiles)]

    def add(self, ts: torch.Tensor, mses: torch.Tensor):
        ts_list = ts.detach().cpu().numpy().tolist()
        mses_list = mses.detach().cpu().numpy().tolist()
        for t, mse in zip(ts_list, mses_list):
            quantile = int(t * (self.quantiles - 1e-8))
            history = self.history[quantile]
            if len(history) == self.avg_size:
                del history[0]
            history.append(mse)

    def quantile_averages(self) -> List[Optional[float]]:
        return [float(np.mean(x)) if len(x) else None for x in self.history]

    def log_str(self) -> str:
        avgs = self.quantile_averages()
        return " ".join(f"q{i}={avg}" for i, avg in enumerate(avgs) if avg is not None)
