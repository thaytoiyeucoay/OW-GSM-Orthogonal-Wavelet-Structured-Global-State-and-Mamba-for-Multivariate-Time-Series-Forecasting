"""Baseline registry and factory.

Each baseline implementation lives in its own module. This file only exposes
the shared lookup table used by the training CLI.
"""

from __future__ import annotations

from typing import Callable

from torch import nn

from .dlinear import DLinear
from .fedformer import FEDformer
from .itransformer import ITransformer
from .patchtst import PatchTST
from .timekan import TimeKAN
from .timexer import TimeXer


BASELINE_REPOS = {
    "itransformer": "https://github.com/thuml/iTransformer",
    "patchtst": "https://github.com/yuqinie98/PatchTST",
    "dlinear": "https://github.com/cure-lab/LTSF-Linear",
    "timekan": "https://github.com/huangst21/TimeKAN",
    "timexer": "https://github.com/thuml/TimeXer",
    "fedformer": "https://github.com/MAZiqing/FEDformer",
}

MODEL_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "dlinear": DLinear,
    "fedformer": FEDformer,
    "itransformer": ITransformer,
    "patchtst": PatchTST,
    "timekan": TimeKAN,
    "timexer": TimeXer,
}


def canonical_baseline_name(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


def list_baselines() -> list[str]:
    return sorted(MODEL_REGISTRY)


def build_baseline_model(
    name: str,
    seq_len: int,
    pred_len: int,
    n_channels: int,
    **kwargs: object,
) -> nn.Module:
    key = canonical_baseline_name(name)
    if key not in MODEL_REGISTRY:
        valid = ", ".join(list_baselines())
        raise ValueError(f"Unknown baseline '{name}'. Valid baselines: {valid}")
    return MODEL_REGISTRY[key](
        seq_len=seq_len,
        pred_len=pred_len,
        n_channels=n_channels,
        **kwargs,
    )
