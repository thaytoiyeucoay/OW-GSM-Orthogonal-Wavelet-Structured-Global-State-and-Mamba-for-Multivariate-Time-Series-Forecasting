"""PyTorch datasets for forecasting windows."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class ForecastingWindowDataset(Dataset):
    """Sliding-window dataset returning ``(encoder_input, target_window)``."""

    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        seq_len: int,
        pred_len: int,
        stride: int = 1,
    ) -> None:
        if seq_len <= 0 or pred_len <= 0:
            raise ValueError("seq_len and pred_len must be positive.")
        if len(x_data) != len(y_data):
            raise ValueError("x_data and y_data must have the same time length.")

        self.x_data = np.asarray(x_data, dtype=np.float32)
        self.y_data = np.asarray(y_data, dtype=np.float32)
        self.seq_len = int(seq_len)
        self.pred_len = int(pred_len)
        self.stride = int(stride)
        self.num_samples = max(
            0, (len(self.x_data) - self.seq_len - self.pred_len) // self.stride + 1
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.stride
        x_end = start + self.seq_len
        y_end = x_end + self.pred_len
        x = self.x_data[start:x_end].copy()
        y = self.y_data[x_end:y_end].copy()
        return torch.from_numpy(x), torch.from_numpy(y)
