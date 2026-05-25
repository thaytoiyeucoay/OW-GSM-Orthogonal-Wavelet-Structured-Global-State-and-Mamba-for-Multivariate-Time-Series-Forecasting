"""TimeXer baseline.

Official reference: https://github.com/thuml/TimeXer
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .common import check_attention_heads


class TimeXer(nn.Module):
    """Transformer baseline with patch-level and variate-level interaction."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        n_heads = check_attention_heads(d_model, n_heads)
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = max(1, (max(seq_len, patch_len) - patch_len) // stride + 1)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.patch_position = nn.Parameter(torch.zeros(1, self.n_patches, d_model))

        patch_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.patch_encoder = nn.TransformerEncoder(patch_layer, num_layers=max(1, e_layers // 2))

        self.global_token = nn.Parameter(torch.zeros(1, 1, d_model))
        variate_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.variate_encoder = nn.TransformerEncoder(variate_layer, num_layers=e_layers)
        self.cross_gate = nn.Linear(d_model * 2, d_model)
        self.head = nn.Linear(d_model, pred_len)
        nn.init.trunc_normal_(self.patch_position, std=0.02)
        nn.init.trunc_normal_(self.global_token, std=0.02)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        x = x.permute(0, 2, 1)
        if seq_len < self.patch_len:
            x = F.pad(x, (0, self.patch_len - seq_len), mode="replicate")
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if patches.size(2) < self.n_patches:
            missing = self.n_patches - patches.size(2)
            patches = torch.cat([patches, patches[:, :, -1:, :].repeat(1, 1, missing, 1)], dim=2)
        patches = patches[:, :, : self.n_patches, :]
        return patches.reshape(batch * channels, self.n_patches, self.patch_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, channels = x.shape
        patches = self._patchify(x)
        z = self.patch_embedding(patches) + self.patch_position
        z = self.patch_encoder(z).mean(dim=1).view(batch, channels, -1)

        global_token = self.global_token.expand(batch, -1, -1)
        tokens = torch.cat([global_token, z], dim=1)
        tokens = self.variate_encoder(tokens)
        global_context = tokens[:, :1, :]
        channel_tokens = tokens[:, 1:, :]
        gate = torch.sigmoid(
            self.cross_gate(torch.cat([channel_tokens, global_context.expand_as(channel_tokens)], dim=-1))
        )
        channel_tokens = channel_tokens + gate * global_context
        return self.head(channel_tokens).permute(0, 2, 1)
