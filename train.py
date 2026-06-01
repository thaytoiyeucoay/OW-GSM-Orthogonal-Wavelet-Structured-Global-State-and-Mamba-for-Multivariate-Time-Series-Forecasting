"""Training entry point for OW-GSM and baselines.

OW-GSM architecture code lives in ``models/owgsm.py``. This script keeps the
paper-aligned experiment protocol: standard dataset splits, RevIN-based model,
AdamW, StepLR(step=3, gamma=0.5), MSE/MAE reporting, and checkpointing.
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from data_utils import DataBundle, canonical_dataset_name, load_forecasting_data
from models.baselines import build_baseline_model, list_baselines
from models.owgsm import (
    HorizonWeightedMSE,
    OWGSM,
    compute_filter_stats,
    compute_spectral_ratios,
)


PAPER_DATASET_CONFIGS = {
    "ETTh1": {"lr": 5e-4, "weight_decay": 5e-5, "dropout": 0.4, "affine": True},
    "ETTh2": {"lr": 5e-4, "weight_decay": 5e-5, "dropout": 0.4, "affine": True},
    "ETTm1": {"lr": 5e-4, "weight_decay": 5e-5, "dropout": 0.4, "affine": True},
    "ETTm2": {"lr": 5e-4, "weight_decay": 5e-5, "dropout": 0.4, "affine": True},
    "Weather": {"lr": 1e-4, "weight_decay": 5e-5, "dropout": 0.4, "affine": True},
    "Exchange": {"lr": 5e-4, "weight_decay": 0.35, "dropout": 0.55, "affine": False},
}


@dataclass
class ExperimentConfig:
    """Single experiment contract shared by JSON configs and CLI overrides."""

    model: str = "owgsm"
    dataset: str = "ETTh1"
    root_path: str = "."
    save_dir: str = "checkpoints"
    seq_len: int = 720
    pred_len: int = 96
    batch_size: int = 32
    d_model: int = 32
    d_ff: int = 256
    n_heads: int = 4
    e_layers: int = 2
    dropout: float = 0.4
    lr: float = 5e-4
    weight_decay: float = 5e-5
    epochs: int = 10
    patience: int = 5
    seed: int = 2026
    device: str = "auto"
    gpu: int = 0
    use_cpu: bool = False
    require_gpu: bool = False
    amp: bool = True
    features: str = "M"
    target: str = "OT"
    split_policy: str = "standard"
    num_workers: int = 0
    horizon_alpha: float = 0.5
    wavelet_weight: float = 0.01
    feature_weight: float = 0.1
    revin_affine: bool = True
    wavelet_kernel: int = 16
    gsr_tokens: int = 4
    patch_size: int = 8
    mamba_conv_kernel: int = 4
    mamba_expand: int = 2
    input_jitter: float = 0.005
    patch_len: int = 16
    patch_stride: int = 8
    moving_avg: int = 25
    modes: int = 32
    grid_size: int = 8


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(config: ExperimentConfig) -> torch.device:
    """Resolve the requested training device with explicit CUDA diagnostics."""

    requested = config.device.lower().strip()
    if config.use_cpu or requested == "cpu":
        return torch.device("cpu")

    if requested == "auto":
        if torch.cuda.is_available():
            if config.gpu < 0 or config.gpu >= torch.cuda.device_count():
                raise ValueError(
                    f"CUDA device index {config.gpu} is unavailable; "
                    f"this machine has {torch.cuda.device_count()} CUDA device(s)."
                )
            device = torch.device(f"cuda:{config.gpu}")
            torch.cuda.set_device(device)
            return device
        if config.require_gpu:
            raise RuntimeError(
                "GPU training was required, but this PyTorch installation does not see CUDA. "
                "Install a CUDA-enabled PyTorch build, then rerun without --cpu."
            )
        return torch.device("cpu")

    if requested == "cuda":
        requested = f"cuda:{config.gpu}"

    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Device '{requested}' was requested, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build that matches your NVIDIA driver."
            )
        try:
            gpu_id = int(requested.split(":", maxsplit=1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device string: {config.device}") from exc
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"CUDA device index {gpu_id} is unavailable; "
                f"this machine has {torch.cuda.device_count()} CUDA device(s)."
            )
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        return device

    raise ValueError("device must be one of 'auto', 'cpu', 'cuda', or 'cuda:<index>'.")


def autocast_context(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda")
    return torch.cuda.amp.autocast()


def make_grad_scaler(enabled: bool):
    if not enabled:
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def move_to_device(x: torch.Tensor, y: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    non_blocking = device.type == "cuda"
    return (
        x.to(device, non_blocking=non_blocking),
        y.to(device, non_blocking=non_blocking),
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def clear_model_runtime_state(model: nn.Module) -> None:
    """Drop per-batch tensors cached by models for diagnostics or auxiliary losses."""

    if hasattr(model, "last_components"):
        model.last_components = {}


def canonical_model_name(model_name: str) -> str:
    return model_name.lower().replace("-", "").replace("_", "")


def build_model(config: ExperimentConfig, n_channels: int) -> nn.Module:
    """Instantiate OW-GSM or a baseline behind the same forecasting interface."""

    key = canonical_model_name(config.model)
    if key in {"owgsm", "mambastat", "mambastatfusion"}:
        return OWGSM(
            input_dim=n_channels,
            seq_len=config.seq_len,
            pred_len=config.pred_len,
            d_model=config.d_model,
            dropout=config.dropout,
            affine=config.revin_affine,
            wavelet_kernel=config.wavelet_kernel,
            gsr_tokens=config.gsr_tokens,
            patch_size=config.patch_size,
            mamba_conv_kernel=config.mamba_conv_kernel,
            mamba_expand=config.mamba_expand,
            input_jitter=config.input_jitter,
        )

    return build_baseline_model(
        key,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        n_channels=n_channels,
        d_model=config.d_model,
        d_ff=config.d_ff,
        n_heads=config.n_heads,
        e_layers=config.e_layers,
        dropout=config.dropout,
        patch_len=config.patch_len,
        stride=config.patch_stride,
        moving_avg=config.moving_avg,
        modes=config.modes,
        grid_size=config.grid_size,
    )


def align_prediction_and_target(
    pred: torch.Tensor,
    target: torch.Tensor,
    bundle: DataBundle,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Handle M/MS settings where the model predicts all channels but the target is one channel."""

    if pred.size(-1) == target.size(-1):
        return pred, target
    if target.size(-1) == 1 and bundle.target_index is not None:
        return pred[:, :, bundle.target_index : bundle.target_index + 1], target
    raise ValueError(
        f"Prediction channels ({pred.size(-1)}) and target channels ({target.size(-1)}) do not match."
    )


def evaluate(
    model: nn.Module,
    data_loader,
    bundle: DataBundle,
    device: torch.device,
    use_amp: bool = False,
) -> dict[str, float]:
    """Compute paper metrics over all predicted values."""

    model.eval()
    mse_sum = 0.0
    mae_sum = 0.0
    count = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = move_to_device(x, y, device)
            with autocast_context(device, use_amp):
                pred = model(x)
                pred, y = align_prediction_and_target(pred, y, bundle)
            pred = pred.float()
            y = y.float()
            mse = F.mse_loss(pred, y, reduction="sum")
            mae = F.l1_loss(pred, y, reduction="sum")
            mse_sum += mse.item()
            mae_sum += mae.item()
            count += y.numel()
            clear_model_runtime_state(model)

    if count == 0:
        raise RuntimeError("Evaluation loader produced zero targets.")
    return {"mse": mse_sum / count, "mae": mae_sum / count}


def train_one_epoch(
    model: nn.Module,
    data_loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler,
    bundle: DataBundle,
    device: torch.device,
    config: ExperimentConfig,
) -> float:
    """One optimization pass: forecast, align targets, add OW-GSM auxiliary losses, update."""

    model.train()
    total_loss = 0.0
    total_batches = 0

    for x, y in data_loader:
        x, y = move_to_device(x, y, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, config.amp):
            pred = model(x)
            pred, y = align_prediction_and_target(pred, y, bundle)
            loss = criterion(pred, y)
            if hasattr(model, "auxiliary_loss"):
                loss = loss + model.auxiliary_loss(
                    wavelet_weight=config.wavelet_weight,
                    feature_weight=config.feature_weight,
                )
        clear_model_runtime_state(model)

        if scaler is None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        total_loss += loss.item()
        total_batches += 1

    if total_batches == 0:
        raise RuntimeError("Training loader produced zero batches.")
    return total_loss / total_batches


def print_wavelet_stats(model: nn.Module, test_loader, device: torch.device) -> None:
    if not isinstance(model, OWGSM):
        return

    filter_stats = compute_filter_stats(model.wavelet)
    spectral_stats = compute_spectral_ratios(model, test_loader, device)
    print("\nWavelet statistics")
    print("------------------")
    for key, value in {**filter_stats, **spectral_stats}.items():
        print(f"{key}: {value}")


def train_model(config: ExperimentConfig) -> dict[str, float]:
    """End-to-end experiment runner used by both config files and CLI calls."""

    set_seed(config.seed)
    dataset_name = canonical_dataset_name(config.dataset)
    device = get_device(config)
    amp_enabled = config.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = False

    bundle = load_forecasting_data(
        dataset_name=dataset_name,
        seq_len=config.seq_len,
        pred_len=config.pred_len,
        batch_size=config.batch_size,
        root_path=config.root_path,
        feature_mode=config.features,
        target=config.target,
        split_policy=config.split_policy,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(
        f"{bundle.dataset_name}: train={len(bundle.train_loader.dataset)}, "
        f"val={len(bundle.val_loader.dataset)}, test={len(bundle.test_loader.dataset)}"
    )

    model = build_model(config, n_channels=bundle.n_channels).to(device)
    print(f"Model: {config.model}")
    print(f"Parameters: {count_parameters(model):,}")

    criterion = HorizonWeightedMSE(config.pred_len, alpha=config.horizon_alpha).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    scaler = make_grad_scaler(amp_enabled)

    best_state = state_dict_to_cpu(model)
    best_val = float("inf")
    best_epoch = 0
    wait = 0

    for epoch in range(1, config.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            data_loader=bundle.train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            bundle=bundle,
            device=device,
            config=config,
        )
        val_metrics = evaluate(model, bundle.val_loader, bundle, device, use_amp=False)

        improved = val_metrics["mse"] < best_val
        if improved:
            best_val = val_metrics["mse"]
            best_epoch = epoch
            best_state = state_dict_to_cpu(model)
            wait = 0
        else:
            wait += 1

        marker = "*" if improved else " "
        print(
            f"Epoch {epoch:03d}{marker} | train_loss={train_loss:.6f} "
            f"| val_mse={val_metrics['mse']:.6f} | val_mae={val_metrics['mae']:.6f}"
        )

        scheduler.step()
        if config.patience > 0 and wait >= config.patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    model.load_state_dict(best_state)
    test_metrics = evaluate(model, bundle.test_loader, bundle, device, use_amp=False)
    print(f"Test MSE: {test_metrics['mse']:.6f}")
    print(f"Test MAE: {test_metrics['mae']:.6f}")
    print_wavelet_stats(model, bundle.test_loader, device)

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / f"{config.model}_{dataset_name}_sl{config.seq_len}_pl{config.pred_len}.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": asdict(config),
            "dataset": dataset_name,
            "test_metrics": test_metrics,
            "best_epoch": best_epoch,
        },
        save_path,
    )
    print(f"Checkpoint saved to {save_path}")
    return test_metrics


def load_config_file(path: str | None) -> dict:
    """Read a JSON config file; CLI arguments can override the returned values."""

    if path is None:
        return {}
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a JSON object: {path}")
    return config


def build_experiment_config(overrides: dict) -> ExperimentConfig:
    """Merge defaults, paper dataset hyperparameters, config file values, and CLI overrides."""

    valid_fields = {field.name for field in fields(ExperimentConfig)}
    unknown = sorted(key for key in overrides if key not in valid_fields)
    if unknown:
        raise ValueError(f"Unknown config keys: {', '.join(unknown)}")

    values = asdict(ExperimentConfig())
    dataset_name = canonical_dataset_name(str(overrides.get("dataset", values["dataset"])))
    dataset_defaults = PAPER_DATASET_CONFIGS[dataset_name]

    values["dataset"] = dataset_name
    values["lr"] = dataset_defaults["lr"]
    values["weight_decay"] = dataset_defaults["weight_decay"]
    values["dropout"] = dataset_defaults["dropout"]
    values["revin_affine"] = dataset_defaults["affine"]
    values.update(overrides)
    values["dataset"] = canonical_dataset_name(str(values["dataset"]))
    return ExperimentConfig(**values)


def collect_cli_overrides(args: argparse.Namespace) -> dict:
    """Keep only CLI arguments explicitly provided by the user."""

    ignored = {"config", "print_config"}
    return {
        key: value
        for key, value in vars(args).items()
        if key not in ignored and value is not None
    }


def build_arg_parser() -> argparse.ArgumentParser:
    model_names = ["owgsm"] + list_baselines()
    parser = argparse.ArgumentParser(description="Train OW-GSM or a baseline on LTSF benchmarks.")
    parser.add_argument("--config", type=str, default=None, help="Path to a JSON experiment config.")
    parser.add_argument("--model", type=str, default=None, help=f"Options: {', '.join(model_names)}")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--root_path", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--pred_len", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--e_layers", type=int, default=None)
    parser.add_argument("--features", type=str, default=None, choices=["M", "S", "MS"])
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--split_policy", type=str, default=None, choices=["standard", "ratio"])
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="auto, cpu, cuda, or cuda:<index>.")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--cpu", dest="use_cpu", action="store_true", default=None)
    parser.add_argument("--require_gpu", "--require-gpu", dest="require_gpu", action="store_true", default=None)
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=None)
    amp_group.add_argument("--no_amp", "--no-amp", dest="amp", action="store_false", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--horizon_alpha", type=float, default=None)
    parser.add_argument("--wavelet_weight", type=float, default=None)
    parser.add_argument("--feature_weight", "--ortho_weight", dest="feature_weight", type=float, default=None)
    parser.add_argument("--revin_affine", type=lambda value: value.lower() == "true", default=None)
    parser.add_argument("--wavelet_kernel", type=int, default=None)
    parser.add_argument("--gsr_tokens", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--mamba_conv_kernel", type=int, default=None)
    parser.add_argument("--mamba_expand", type=int, default=None)
    parser.add_argument("--input_jitter", type=float, default=None)
    parser.add_argument("--patch_len", type=int, default=None)
    parser.add_argument("--patch_stride", type=int, default=None)
    parser.add_argument("--moving_avg", type=int, default=None)
    parser.add_argument("--modes", type=int, default=None)
    parser.add_argument("--grid_size", type=int, default=None)
    parser.add_argument("--print_config", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    file_overrides = load_config_file(args.config)
    cli_overrides = collect_cli_overrides(args)
    config = build_experiment_config({**file_overrides, **cli_overrides})
    if args.print_config:
        print(json.dumps(asdict(config), indent=2))
    train_model(config)


if __name__ == "__main__":
    main()
