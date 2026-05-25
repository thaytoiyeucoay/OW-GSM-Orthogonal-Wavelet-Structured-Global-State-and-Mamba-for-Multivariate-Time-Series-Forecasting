"""FEDformer baseline.

Official reference: https://github.com/MAZiqing/FEDformer
"""

from __future__ import annotations

import torch
from torch import nn

from .common import SeriesDecomposition


class FourierBlock(nn.Module):
    """Frequency-domain channel-wise filtering block."""

    def __init__(self, seq_len: int, n_channels: int, modes: int = 32) -> None:
        super().__init__()
        max_modes = seq_len // 2 + 1
        self.modes = min(modes, max_modes)
        scale = 1.0 / max(1, n_channels)
        self.weight = nn.Parameter(scale * torch.randn(self.modes, n_channels, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x_ft = torch.fft.rfft(x, dim=1)
        out_ft = torch.zeros_like(x_ft)
        weight = torch.view_as_complex(self.weight.contiguous())
        modes = min(self.modes, x_ft.size(1))
        out_ft[:, :modes, :] = x_ft[:, :modes, :] * weight[:modes].unsqueeze(0)
        return torch.fft.irfft(out_ft, n=seq_len, dim=1)


class FEDformer(nn.Module):
    """Frequency enhanced decomposition baseline."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        moving_avg: int = 25,
        modes: int = 32,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        self.decomposition = SeriesDecomposition(moving_avg)
        self.fourier = FourierBlock(seq_len, n_channels, modes=modes)
        self.channel_mixer = nn.Linear(n_channels, n_channels)
        self.seasonal_head = nn.Linear(seq_len, pred_len)
        self.trend_head = nn.Linear(seq_len, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomposition(x)
        seasonal = self.fourier(seasonal)
        seasonal = self.channel_mixer(seasonal)
        seasonal = self.dropout(seasonal).permute(0, 2, 1)
        trend = trend.permute(0, 2, 1)
        out = self.seasonal_head(seasonal) + self.trend_head(trend)
        return out.permute(0, 2, 1)
