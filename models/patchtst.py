"""PatchTST baseline.

Official reference: https://github.com/yuqinie98/PatchTST
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .common import check_attention_heads


class PatchTST(nn.Module):
    """PatchTST-style channel-independent Transformer."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        **_: object,
    ) -> None:
        super().__init__()
        del n_channels
        n_heads = check_attention_heads(d_model, n_heads)
        self.patch_len = patch_len
        self.stride = stride
        self.n_patches = max(1, (max(seq_len, patch_len) - patch_len) // stride + 1)
        self.patch_embedding = nn.Linear(patch_len, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
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
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Dropout(dropout),
            nn.Linear(self.n_patches * d_model, pred_len),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        x = x.permute(0, 2, 1)
        if seq_len < self.patch_len:
            x = F.pad(x, (0, self.patch_len - seq_len), mode="replicate")
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if patches.size(2) < self.n_patches:
            missing = self.n_patches - patches.size(2)
            pad_patch = patches[:, :, -1:, :].repeat(1, 1, missing, 1)
            patches = torch.cat([patches, pad_patch], dim=2)
        patches = patches[:, :, : self.n_patches, :]
        return patches.reshape(batch * channels, self.n_patches, self.patch_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, channels = x.shape
        patches = self._patchify(x)
        z = self.patch_embedding(patches) + self.position_embedding
        z = self.encoder(z)
        out = self.head(z)
        return out.view(batch, channels, -1).permute(0, 2, 1)
