"""OW-GSM architecture.

This module follows the paper design:

1. Last-value centering and RevIN.
2. Learnable approximately orthogonal wavelet split.
3. Trend branch with Global State Register.
4. Detail branch with patch normalization and bidirectional Mamba-style scan.
5. Adaptive gated fusion and per-channel MLP decoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class OWGSMConfig:
    input_dim: int
    seq_len: int = 720
    pred_len: int = 96
    d_model: int = 32
    dropout: float = 0.4
    revin_affine: bool = True
    wavelet_kernel: int = 16
    gsr_tokens: int = 4
    patch_size: int = 8
    mamba_conv_kernel: int = 4
    mamba_expand: int = 2
    input_jitter: float = 0.005


class RevIN(nn.Module):
    """Reversible instance normalization."""

    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        self._mean: Optional[torch.Tensor] = None
        self._std: Optional[torch.Tensor] = None
        if affine:
            self.gamma = nn.Parameter(torch.ones(1, 1, num_features))
            self.beta = nn.Parameter(torch.zeros(1, 1, num_features))

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "norm":
            self._mean = x.mean(dim=1, keepdim=True).detach()
            self._std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
            x = (x - self._mean) / self._std
            if self.affine:
                x = x * self.gamma + self.beta
            return x

        if mode == "denorm":
            if self._mean is None or self._std is None:
                raise RuntimeError("RevIN denorm was called before norm.")
            if self.affine:
                x = (x - self.beta) / (self.gamma + self.eps)
            return x * self._std + self._mean

        raise ValueError("mode must be either 'norm' or 'denorm'.")


class LightMambaBlock(nn.Module):
    """Lightweight gated state-space style block for local detail modeling.

    The implementation keeps the OW-GSM interface self-contained. It uses a
    depthwise convolution and input-dependent gates to emulate the local,
    linear-time selective scanning behavior required by the detail branch.
    """

    def __init__(
        self,
        d_model: int,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        d_inner = d_model * expand
        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.depthwise_conv = nn.Conv1d(
            d_inner,
            d_inner,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=d_inner,
        )
        self.dt_proj = nn.Linear(d_inner, d_inner)
        self.skip = nn.Parameter(torch.ones(d_inner))
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        length = x.size(1)
        x = self.norm(x)
        x, gate = self.in_proj(x).chunk(2, dim=-1)
        x = self.depthwise_conv(x.transpose(1, 2))[:, :, :length].transpose(1, 2)
        x = F.silu(x * F.softplus(self.dt_proj(x)))
        x = x * F.silu(gate) * self.skip
        return residual + self.dropout(self.out_proj(x))


class LearnableWaveletSplitter(nn.Module):
    """Learnable low-pass and high-pass filters with stride-2 decimation."""

    def __init__(self, kernel_size: int = 16) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        sqrt2_over_k = math.sqrt(2.0) / kernel_size
        low_init = torch.full((kernel_size,), sqrt2_over_k)
        high_init = torch.tensor([sqrt2_over_k * ((-1) ** idx) for idx in range(kernel_size)])
        self.low_filter = nn.Parameter(low_init)
        self.high_filter = nn.Parameter(high_init)

    @property
    def g_theta(self) -> nn.Parameter:
        return self.low_filter

    @property
    def h_theta(self) -> nn.Parameter:
        return self.high_filter

    def full_resolution(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, channels = x.shape
        x_flat = x.permute(0, 2, 1).reshape(batch * channels, 1, length)
        pad = self.kernel_size // 2
        x_padded = F.pad(x_flat, (pad, pad), mode="replicate")
        low = F.conv1d(x_padded, self.low_filter.view(1, 1, -1))[:, :, :length]
        high = F.conv1d(x_padded, self.high_filter.view(1, 1, -1))[:, :, :length]
        low = low.squeeze(1).reshape(batch, channels, length).permute(0, 2, 1)
        high = high.squeeze(1).reshape(batch, channels, length).permute(0, 2, 1)
        return low, high

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low, high = self.full_resolution(x)
        return low[:, ::2, :], high[:, ::2, :]

    def regularization_loss(self) -> torch.Tensor:
        """Equation 7: norm, zero-mean, and filter orthogonality constraints."""

        sqrt2 = math.sqrt(2.0)
        low_norm = (self.low_filter.sum() - sqrt2).pow(2)
        high_zero_mean = self.high_filter.sum().pow(2)
        filter_orthogonality = torch.dot(self.low_filter, self.high_filter).pow(2)
        return low_norm + high_zero_mean + filter_orthogonality


class GlobalStateRegister(nn.Module):
    """Global prototype retrieval for trend features."""

    def __init__(self, d_model: int, n_tokens: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.global_tokens = nn.Parameter(torch.randn(n_tokens, d_model) * 0.02)
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.gate_proj = nn.Linear(d_model * 2, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, dim = x.shape
        memory = self.global_tokens.unsqueeze(0).expand(batch, -1, -1)
        query = self.query_proj(x)
        key = self.key_proj(memory)
        value = self.value_proj(memory)
        attention = torch.bmm(query, key.transpose(1, 2)) / math.sqrt(dim)
        attention = attention.softmax(dim=-1)
        self.last_attention = attention.detach()
        context = torch.bmm(self.dropout(attention), value)
        gate = torch.sigmoid(self.gate_proj(torch.cat([x, context], dim=-1)))
        return x + gate * context


class DetailEncoder(nn.Module):
    """Patch-tokenized bidirectional detail encoder."""

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        patch_size: int = 8,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.stride = max(1, patch_size // 2)
        self.n_patches = max(1, (max(seq_len, patch_size) - patch_size) // self.stride + 1)
        self.patch_embed = nn.Linear(patch_size, d_model)
        self.position_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_model))
        self.mamba = LightMambaBlock(d_model, d_conv=d_conv, expand=expand, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.trunc_normal_(self.position_embed, std=0.02)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, channels = x.shape
        x = x.permute(0, 2, 1)
        if length < self.patch_size:
            x = F.pad(x, (0, self.patch_size - length), mode="replicate")
        patches = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        if patches.size(2) < self.n_patches:
            missing = self.n_patches - patches.size(2)
            patches = torch.cat([patches, patches[:, :, -1:, :].repeat(1, 1, missing, 1)], dim=2)
        patches = patches[:, :, : self.n_patches, :]
        return patches.reshape(batch * channels, self.n_patches, self.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, channels = x.shape
        patches = self._patchify(x)
        patch_mean = patches.mean(dim=-1, keepdim=True)
        patch_std = torch.sqrt(patches.var(dim=-1, keepdim=True, unbiased=False) + 1e-5)
        patches = (patches - patch_mean) / patch_std

        z = self.patch_embed(patches) + self.position_embed
        z = self.dropout(z)
        z_forward = self.mamba(z)
        z_backward = torch.flip(self.mamba(torch.flip(z, dims=[1])), dims=[1])
        z = 0.5 * (z_forward + z_backward)
        z = self.norm(z.mean(dim=1))
        return z.view(batch, channels, -1)


class OWGSM(nn.Module):
    """Orthogonal Wavelet-Structured Global State and Mamba forecaster."""

    def __init__(
        self,
        input_dim: int,
        seq_len: int = 720,
        pred_len: int = 96,
        d_model: int = 32,
        dropout: float = 0.4,
        affine: bool = True,
        wavelet_kernel: int = 16,
        gsr_tokens: int = 4,
        patch_size: int = 8,
        mamba_conv_kernel: int = 4,
        mamba_expand: int = 2,
        input_jitter: float = 0.005,
    ) -> None:
        super().__init__()
        self.config = OWGSMConfig(
            input_dim=input_dim,
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            dropout=dropout,
            revin_affine=affine,
            wavelet_kernel=wavelet_kernel,
            gsr_tokens=gsr_tokens,
            patch_size=patch_size,
            mamba_conv_kernel=mamba_conv_kernel,
            mamba_expand=mamba_expand,
            input_jitter=input_jitter,
        )
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.input_jitter = input_jitter
        wavelet_len = (seq_len + 1) // 2

        self.revin = RevIN(input_dim, affine=affine)
        self.wavelet = LearnableWaveletSplitter(kernel_size=wavelet_kernel)
        self.trend_embed = nn.Linear(wavelet_len, d_model)
        self.trend_global = GlobalStateRegister(d_model, n_tokens=gsr_tokens, dropout=dropout)
        self.detail_encoder = DetailEncoder(
            wavelet_len,
            d_model=d_model,
            patch_size=patch_size,
            d_conv=mamba_conv_kernel,
            expand=mamba_expand,
            dropout=dropout,
        )
        self.fusion_gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * 2, pred_len),
                )
                for _ in range(input_dim)
            ]
        )
        self.last_components: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor, return_components: bool = False):
        _, _, channels = x.shape
        if channels != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} channels, got {channels}.")

        # Work on relative changes, then restore the last observed value at the end.
        last_value = x[:, -1:, :].detach()
        centered = x - last_value
        if self.training and self.input_jitter > 0:
            centered = centered + torch.randn_like(centered) * self.input_jitter

        # Remove instance-level non-stationary statistics before decomposition.
        normalized = self.revin(centered, mode="norm")

        # Split the signal into low-frequency approximation and high-frequency detail bands.
        approximation, detail = self.wavelet(normalized)

        # Trend branch: retrieve stable global prototypes for slowly varying components.
        trend = self.trend_embed(approximation.permute(0, 2, 1))
        trend = self.trend_global(trend)

        # Detail branch: tokenize local fluctuations and scan them in both directions.
        detail_features = self.detail_encoder(detail)

        # Adaptive fusion keeps trend as the anchor and injects detail only when useful.
        gate = self.fusion_gate(torch.cat([trend, detail_features], dim=-1))
        fused = trend + gate * detail_features

        # Each channel has its own lightweight decoder to avoid noisy cross-channel projection.
        outputs = [decoder(fused[:, channel, :]) for channel, decoder in enumerate(self.projections)]
        prediction = torch.stack(outputs, dim=-1)
        prediction = self.revin(prediction, mode="denorm") + last_value

        self.last_components = {
            "approximation": approximation,
            "detail": detail,
            "trend": trend,
            "detail_features": detail_features,
            "gate": gate,
            "fused": fused,
        }
        if return_components:
            return prediction, self.last_components
        return prediction

    def feature_disentanglement_loss(self) -> torch.Tensor:
        if "trend" not in self.last_components or "detail_features" not in self.last_components:
            return self.wavelet.low_filter.new_tensor(0.0)
        trend = self.last_components["trend"]
        detail = self.last_components["detail_features"]
        return F.cosine_similarity(trend, detail, dim=-1).abs().mean()

    def auxiliary_loss(
        self,
        wavelet_weight: float = 0.01,
        feature_weight: float = 0.1,
    ) -> torch.Tensor:
        # Paper objective: encourage wavelet-like filters and trend/detail disentanglement.
        return (
            wavelet_weight * self.wavelet.regularization_loss()
            + feature_weight * self.feature_disentanglement_loss()
        )


HybridFusionModel = OWGSM


class HorizonWeightedMSE(nn.Module):
    """Equation 6: later horizons receive larger weights."""

    def __init__(self, pred_len: int, alpha: float = 0.5) -> None:
        super().__init__()
        steps = torch.arange(1, pred_len + 1, dtype=torch.float32)
        weights = 1.0 + alpha * steps / pred_len
        self.register_buffer("weights", weights.view(1, pred_len, 1))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return (((pred - target) ** 2) * self.weights[:, : pred.size(1), :]).mean()


def compute_filter_stats(wavelet: LearnableWaveletSplitter) -> dict[str, float]:
    low = wavelet.low_filter.detach().cpu()
    high = wavelet.high_filter.detach().cpu()
    raw_cross = torch.dot(low, high).abs().item()
    normalized_cross = (torch.dot(low, high).abs() / (low.norm() * high.norm() + 1e-10)).item()
    return {
        "abs_dot_raw": round(raw_cross, 6),
        "abs_dot_normalized": round(normalized_cross, 6),
        "abs_sum_low_minus_sqrt2": round(abs(low.sum().item() - math.sqrt(2.0)), 6),
        "abs_sum_high": round(abs(high.sum().item()), 6),
    }


def compute_spectral_ratios(
    model: OWGSM,
    data_loader,
    device: torch.device,
    n_batches: int = 50,
    cutoff_fraction: float = 0.25,
) -> dict[str, float]:
    model.eval()
    trend_ratios: list[float] = []
    detail_ratios: list[float] = []

    with torch.no_grad():
        for batch_index, (x, _) in enumerate(data_loader):
            if batch_index >= n_batches:
                break
            x = x.to(device)
            last_value = x[:, -1:, :].detach()
            normalized = model.revin(x - last_value, mode="norm")
            low, high = model.wavelet.full_resolution(normalized)
            cutoff_bin = max(1, int((low.size(1) // 2 + 1) * cutoff_fraction))

            low_power = torch.fft.rfft(low, dim=1).abs().pow(2)
            high_power = torch.fft.rfft(high, dim=1).abs().pow(2)
            low_total = low_power.sum(dim=1) + 1e-10
            high_total = high_power.sum(dim=1) + 1e-10
            low_energy = low_power[:, : cutoff_bin + 1, :].sum(dim=1)
            high_energy = high_power[:, cutoff_bin + 1 :, :].sum(dim=1)
            trend_ratios.append((low_energy / low_total).mean().item())
            detail_ratios.append((high_energy / high_total).mean().item())

    if not trend_ratios:
        return {}

    return {
        "trend_low_frequency_percent": round(float(np.mean(trend_ratios)) * 100.0, 2),
        "detail_high_frequency_percent": round(float(np.mean(detail_ratios)) * 100.0, 2),
        "trend_low_frequency_std": round(float(np.std(trend_ratios)) * 100.0, 2),
        "detail_high_frequency_std": round(float(np.std(detail_ratios)) * 100.0, 2),
    }
