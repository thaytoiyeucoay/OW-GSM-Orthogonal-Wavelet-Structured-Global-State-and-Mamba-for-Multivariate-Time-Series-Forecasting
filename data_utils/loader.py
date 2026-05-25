"""Forecasting data loading and normalization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from .datasets import ForecastingWindowDataset
from .registry import canonical_dataset_name, find_dataset_file
from .splits import SplitIndices, get_split_indices


@dataclass
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    n_channels: int
    n_targets: int
    mean: np.ndarray
    std: np.ndarray
    split: SplitIndices
    dataset_name: str
    data_path: Path
    feature_mode: str
    target_name: str
    target_index: Optional[int]


def load_numeric_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".txt":
        values = np.loadtxt(path, delimiter=",").astype(np.float32)
        numeric = pd.DataFrame(values)
    else:
        frame = pd.read_csv(path, low_memory=False)
        numeric = frame.select_dtypes(include=[np.number]).copy()
        if numeric.empty:
            raise ValueError(f"No numeric columns found in {path}.")

    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    if numeric.isna().any().any():
        numeric = numeric.interpolate(method="linear", axis=0, limit_direction="both")
        numeric = numeric.ffill().bfill()
    return numeric.astype(np.float32)


def select_features(
    numeric: pd.DataFrame,
    feature_mode: str,
    target: str = "OT",
) -> tuple[pd.DataFrame, pd.DataFrame, str, Optional[int]]:
    mode = feature_mode.upper()
    if mode not in {"M", "S", "MS"}:
        raise ValueError("feature_mode must be one of 'M', 'S', or 'MS'.")

    column_lookup = {str(column): column for column in numeric.columns}
    if str(target) in column_lookup:
        target_column = column_lookup[str(target)]
    elif target == "OT":
        target_column = numeric.columns[-1]
    else:
        raise ValueError(f"Target column '{target}' not found.")

    target_name = str(target_column)
    target_index = int(list(numeric.columns).index(target_column))

    if mode == "M":
        return numeric, numeric, target_name, None
    if mode == "S":
        selected = numeric[[target_column]]
        return selected, selected, target_name, 0

    return numeric, numeric[[target_column]], target_name, target_index


def load_forecasting_data(
    dataset_name: str,
    seq_len: int,
    pred_len: int,
    batch_size: int = 32,
    root_path: str | Path = ".",
    feature_mode: str = "M",
    target: str = "OT",
    split_policy: str = "standard",
    stride: int = 1,
    num_workers: int = 0,
    drop_last: bool = True,
    pin_memory: bool = False,
) -> DataBundle:
    """Build train/val/test loaders with the same preprocessing for every model."""

    # Resolve aliases such as exchange_rate -> Exchange and locate the CSV file.
    canonical = canonical_dataset_name(dataset_name)
    data_path = find_dataset_file(canonical, root_path=root_path)
    numeric = load_numeric_frame(data_path)

    # Choose M/S/MS feature mode before slicing windows.
    x_frame, y_frame, target_name, target_index = select_features(
        numeric, feature_mode=feature_mode, target=target
    )

    # Create chronological splits with lookback buffers for validation/test windows.
    split = get_split_indices(canonical, len(numeric), seq_len, split_policy=split_policy)

    # Fit normalization on training rows only to avoid validation/test leakage.
    all_values = numeric.to_numpy(dtype=np.float32)
    train_values = all_values[split.train[0] : split.train[1]]
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0) + 1e-5
    normalized_all = (all_values - mean) / std
    normalized = pd.DataFrame(normalized_all, columns=numeric.columns)

    x_values = normalized[x_frame.columns].to_numpy(dtype=np.float32)
    y_values = normalized[y_frame.columns].to_numpy(dtype=np.float32)

    train_x = x_values[split.train[0] : split.train[1]]
    train_y = y_values[split.train[0] : split.train[1]]
    val_x = x_values[split.val[0] : split.val[1]]
    val_y = y_values[split.val[0] : split.val[1]]
    test_x = x_values[split.test[0] : split.test[1]]
    test_y = y_values[split.test[0] : split.test[1]]

    # Convert continuous arrays into sliding-window forecasting datasets.
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        ForecastingWindowDataset(train_x, train_y, seq_len, pred_len, stride=stride),
        shuffle=True,
        drop_last=drop_last,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        ForecastingWindowDataset(val_x, val_y, seq_len, pred_len, stride=stride),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        ForecastingWindowDataset(test_x, test_y, seq_len, pred_len, stride=stride),
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        n_channels=x_values.shape[1],
        n_targets=y_values.shape[1],
        mean=mean,
        std=std,
        split=split,
        dataset_name=canonical,
        data_path=data_path,
        feature_mode=feature_mode.upper(),
        target_name=target_name,
        target_index=target_index,
    )


def load_data(dataset_name: str, config: dict, root_path: str = "./"):
    """Backward-compatible wrapper used by older scripts."""

    bundle = load_forecasting_data(
        dataset_name=dataset_name,
        seq_len=int(config["seq_len"]),
        pred_len=int(config["pred_len"]),
        batch_size=int(config.get("batch_size", 32)),
        root_path=root_path,
        feature_mode=str(config.get("features", "M")),
        target=str(config.get("target", "OT")),
        split_policy=str(config.get("split_policy", "standard")),
        stride=int(config.get("stride", 1)),
        num_workers=int(config.get("num_workers", 0)),
        drop_last=bool(config.get("drop_last", True)),
        pin_memory=bool(config.get("pin_memory", False)),
    )

    return (
        bundle.train_loader,
        bundle.val_loader,
        bundle.test_loader,
        bundle.n_channels,
        bundle.mean,
        bundle.std,
    )


def describe_bundle(bundle: DataBundle) -> str:
    return (
        f"{bundle.dataset_name}: path={bundle.data_path}, "
        f"features={bundle.feature_mode}, channels={bundle.n_channels}, "
        f"targets={bundle.n_targets}, target={bundle.target_name}, "
        f"train={len(bundle.train_loader.dataset)}, "
        f"val={len(bundle.val_loader.dataset)}, "
        f"test={len(bundle.test_loader.dataset)}"
    )
