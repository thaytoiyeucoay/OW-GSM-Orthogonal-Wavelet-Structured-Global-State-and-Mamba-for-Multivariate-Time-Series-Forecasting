"""iTransformer baseline.

Official reference: https://github.com/thuml/iTransformer
"""

from __future__ import annotations

import torch
from torch import nn

from .common import check_attention_heads


class ITransformer(nn.Module):
    """Inverted Transformer baseline with variables as tokens."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_norm: bool = True,
        **_: object,
    ) -> None:
        super().__init__()
        n_heads = check_attention_heads(d_model, n_heads)
        self.use_norm = use_norm
        self.value_embedding = nn.Linear(seq_len, d_model)
        self.variable_embedding = nn.Parameter(torch.zeros(1, n_channels, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.projection = nn.Linear(d_model, pred_len)
        nn.init.trunc_normal_(self.variable_embedding, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            mean = x.mean(dim=1, keepdim=True).detach()
            std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            x = (x - mean) / std
        else:
            mean = std = None

        x = x.permute(0, 2, 1)
        x = self.value_embedding(x) + self.variable_embedding[:, : x.size(1), :]
        x = self.encoder(x)
        out = self.projection(x).permute(0, 2, 1)

        if self.use_norm:
            out = out * std + mean
        return out
