"""Chronological split policies."""

from __future__ import annotations

from dataclasses import dataclass

from .registry import canonical_dataset_name


@dataclass(frozen=True)
class SplitIndices:
    """Inclusive-exclusive row ranges after adding lookback buffers."""

    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]
    train_end: int
    val_end: int
    test_end: int
    total_length: int


def get_split_indices(
    dataset_name: str,
    total_length: int,
    seq_len: int,
    split_policy: str = "standard",
) -> SplitIndices:
    """Return standard chronological splits.

    ``standard`` uses the common 12/4/4-month protocol for ETT datasets and
    70/10/20 for the other bundled benchmarks.
    """

    canonical = canonical_dataset_name(dataset_name)
    policy = split_policy.lower()

    if policy not in {"standard", "ratio"}:
        raise ValueError("split_policy must be either 'standard' or 'ratio'.")

    if policy == "standard" and canonical in {"ETTh1", "ETTh2"}:
        train_end = 12 * 30 * 24
        val_end = train_end + 4 * 30 * 24
        test_end = val_end + 4 * 30 * 24
    elif policy == "standard" and canonical in {"ETTm1", "ETTm2"}:
        train_end = 12 * 30 * 24 * 4
        val_end = train_end + 4 * 30 * 24 * 4
        test_end = val_end + 4 * 30 * 24 * 4
    else:
        train_end = int(total_length * 0.70)
        val_end = int(total_length * 0.80)
        test_end = total_length

    if test_end > total_length:
        raise ValueError(
            f"Dataset {canonical} has {total_length} rows, but split requires {test_end}."
        )

    return SplitIndices(
        train=(0, train_end),
        val=(max(0, train_end - seq_len), val_end),
        test=(max(0, val_end - seq_len), test_end),
        train_end=train_end,
        val_end=val_end,
        test_end=test_end,
        total_length=total_length,
    )
