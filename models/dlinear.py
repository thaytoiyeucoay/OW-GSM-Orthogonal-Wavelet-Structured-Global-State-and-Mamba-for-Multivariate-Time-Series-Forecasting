"""DLinear baseline.

Official reference: https://github.com/cure-lab/LTSF-Linear
"""

from __future__ import annotations

import torch
from torch import nn

from .common import SeriesDecomposition


class DLinear(nn.Module):
    """Decomposition-Linear baseline."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        moving_avg: int = 25,
        individual: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.individual = individual
        self.decomposition = SeriesDecomposition(moving_avg)

        if individual:
            self.seasonal_linear = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(n_channels)]
            )
            self.trend_linear = nn.ModuleList(
                [nn.Linear(seq_len, pred_len) for _ in range(n_channels)]
            )
        else:
            self.seasonal_linear = nn.Linear(seq_len, pred_len)
            self.trend_linear = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomposition(x)
        seasonal = seasonal.permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)

        if self.individual:
            seasonal_out = []
            trend_out = []
            for channel in range(self.n_channels):
                seasonal_out.append(self.seasonal_linear[channel](seasonal[:, channel, :]))
                trend_out.append(self.trend_linear[channel](trend[:, channel, :]))
            return torch.stack(seasonal_out, dim=-1) + torch.stack(trend_out, dim=-1)

        out = self.seasonal_linear(seasonal) + self.trend_linear(trend)
        return out.permute(0, 2, 1)
