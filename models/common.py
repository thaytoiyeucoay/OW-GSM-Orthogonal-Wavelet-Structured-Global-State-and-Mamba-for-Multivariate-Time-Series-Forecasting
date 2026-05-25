"""Shared building blocks for forecasting baseline models."""

from __future__ import annotations

import torch
from torch import nn


def check_attention_heads(d_model: int, n_heads: int) -> int:
    """Return a valid number of attention heads for ``d_model``."""

    if d_model % n_heads == 0:
        return n_heads
    for candidate in range(min(n_heads, d_model), 0, -1):
        if d_model % candidate == 0:
            return candidate
    return 1


class MovingAverage(nn.Module):
    """Centered moving average with replicated boundary padding."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.pool = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end = x[:, -1:, :].repeat(1, pad, 1)
        padded = torch.cat([front, x, end], dim=1)
        return self.pool(padded.permute(0, 2, 1)).permute(0, 2, 1)


class SeriesDecomposition(nn.Module):
    """Moving-average decomposition into seasonal and trend components."""

    def __init__(self, kernel_size: int = 25) -> None:
        super().__init__()
        self.moving_average = MovingAverage(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_average(x)
        seasonal = x - trend
        return seasonal, trend
