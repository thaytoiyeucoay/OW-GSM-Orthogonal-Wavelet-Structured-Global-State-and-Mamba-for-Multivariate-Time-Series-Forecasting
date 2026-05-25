"""Data loading package for long-horizon forecasting experiments."""

from .datasets import ForecastingWindowDataset
from .loader import DataBundle, describe_bundle, load_data, load_forecasting_data
from .registry import (
    DATASET_ALIASES,
    DATASET_FILES,
    DATASET_FREQUENCIES,
    canonical_dataset_name,
    find_dataset_file,
)
from .splits import SplitIndices, get_split_indices

__all__ = [
    "DATASET_ALIASES",
    "DATASET_FILES",
    "DATASET_FREQUENCIES",
    "DataBundle",
    "ForecastingWindowDataset",
    "SplitIndices",
    "canonical_dataset_name",
    "describe_bundle",
    "find_dataset_file",
    "get_split_indices",
    "load_data",
    "load_forecasting_data",
]
