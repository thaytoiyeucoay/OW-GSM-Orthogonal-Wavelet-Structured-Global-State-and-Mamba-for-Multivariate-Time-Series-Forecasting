"""TimeKAN baseline.

Official reference: https://github.com/huangst21/TimeKAN
"""

from __future__ import annotations

import torch
from torch import nn


class KANLinear(nn.Module):
    """Lightweight RBF-KAN layer used by the TimeKAN baseline."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 8,
        grid_range: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features)
        self.spline = nn.Linear(in_features * grid_size, out_features)
        centers = torch.linspace(grid_range[0], grid_range[1], grid_size)
        self.register_buffer("centers", centers)
        self.gamma = 1.0 / ((centers[1] - centers[0]).item() ** 2) if grid_size > 1 else 1.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        basis = torch.exp(-self.gamma * (x.unsqueeze(-1) - self.centers) ** 2)
        basis = basis.flatten(start_dim=-2)
        return self.base(x) + self.spline(basis)


class TimeKAN(nn.Module):
    """Temporal KAN baseline with channel-independent sequence projection."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        d_model: int = 128,
        grid_size: int = 8,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        del n_channels
        self.temporal_kan = KANLinear(seq_len, d_model, grid_size=grid_size)
        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, channels = x.shape
        x = x.permute(0, 2, 1)
        mean = x.mean(dim=-1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + 1e-5)
        x = (x - mean) / std
        z = self.temporal_kan(x.reshape(batch * channels, -1))
        z = z + self.ffn(z)
        out = self.head(z).view(batch, channels, -1)
        out = out * std + mean
        return out.permute(0, 2, 1)
