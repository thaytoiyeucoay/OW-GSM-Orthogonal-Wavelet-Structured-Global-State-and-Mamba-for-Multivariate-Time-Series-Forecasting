"""Dataset names, aliases, and path discovery."""

from __future__ import annotations

from pathlib import Path


DATASET_ALIASES = {
    "etth1": "ETTh1",
    "etth2": "ETTh2",
    "ettm1": "ETTm1",
    "ettm2": "ETTm2",
    "exchange": "Exchange",
    "exchange_rate": "Exchange",
    "weather": "Weather",
}

DATASET_FILES = {
    "ETTh1": "ETTh1.csv",
    "ETTh2": "ETTh2.csv",
    "ETTm1": "ETTm1.csv",
    "ETTm2": "ETTm2.csv",
    "Exchange": "exchange.csv",
    "Weather": "weather.csv",
}

DATASET_FREQUENCIES = {
    "ETTh1": "h",
    "ETTh2": "h",
    "ETTm1": "15min",
    "ETTm2": "15min",
    "Exchange": "d",
    "Weather": "10min",
}


def canonical_dataset_name(dataset_name: str) -> str:
    key = dataset_name.strip().lower().replace("-", "_")
    if key not in DATASET_ALIASES:
        valid = ", ".join(sorted(DATASET_ALIASES))
        raise ValueError(f"Unknown dataset '{dataset_name}'. Valid names: {valid}")
    return DATASET_ALIASES[key]


def find_dataset_file(dataset_name: str, root_path: str | Path = ".") -> Path:
    canonical = canonical_dataset_name(dataset_name)
    root = Path(root_path)
    expected_file = DATASET_FILES[canonical]

    candidates = [
        root / "dataset" / expected_file,
        root / "datasets" / expected_file,
        root / "data" / expected_file,
        root / expected_file,
        root / "dataset" / canonical / expected_file,
        root / "dataset" / "ETT-small" / expected_file,
        root / "data" / "ETT-small" / expected_file,
    ]

    for path in candidates:
        if path.exists():
            return path

    searched = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {canonical} data file. Searched:\n{searched}")
