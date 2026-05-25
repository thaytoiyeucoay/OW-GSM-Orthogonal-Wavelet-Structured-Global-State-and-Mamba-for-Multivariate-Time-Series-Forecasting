# OW-GSM: Orthogonal Wavelet-Structured Global State and Mamba for Multivariate Time Series Forecasting

Official repository for **OW-GSM: Orthogonal Wavelet-Structured Global State and Mamba for Multivariate Time Series Forecasting**.

OW-GSM targets long-horizon multivariate time-series forecasting with a learnable orthogonal wavelet split, a structured global state branch, and a lightweight Mamba-style detail branch. The repository now includes the OW-GSM implementation, unified data loading, compact in-repo baselines, and benchmark datasets.

## News

- **2026-05-25:** Added OW-GSM training code, unified benchmark data loader, and baseline model registry.
- **2026-05-25:** Dataset-first repository initialized with ETT, Exchange, and Weather benchmarks.

## Contents

- [Overview](#overview)
- [Repository Layout](#repository-layout)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Implemented Models](#implemented-models)
- [Datasets](#datasets)
- [Benchmark Protocol](#benchmark-protocol)
- [Reproducibility Checklist](#reproducibility-checklist)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [References](#references)
- [License](#license)

## Overview

Long-term forecasting requires models to capture local continuity, long-range temporal dependencies, cross-variable interactions, and scale-specific temporal patterns. OW-GSM follows this motivation with two complementary branches:

- A learnable wavelet splitter produces trend-like and detail-like components.
- The trend branch uses channel-wise global state tokens.
- The detail branch uses patch-level bidirectional Mamba-style encoding.
- A gated fusion layer combines trend and detail features before forecasting.

The code is organized so OW-GSM and all baselines consume the same tensors with shape `(batch, seq_len, channels)` and return forecasts with shape `(batch, pred_len, channels)`.

## Repository Layout

```text
.
|-- dataset/
|   |-- ETTh1.csv
|   |-- ETTh2.csv
|   |-- ETTm1.csv
|   |-- ETTm2.csv
|   |-- exchange.csv
|   `-- weather.csv
|-- configs/
|   |-- owgsm/
|   |-- dlinear/
|   |-- fedformer/
|   |-- itransformer/
|   |-- patchtst/
|   |-- timekan/
|   `-- timexer/
|-- data_utils/
|   |-- __init__.py
|   |-- datasets.py
|   |-- loader.py
|   |-- registry.py
|   `-- splits.py
|-- models/
|   |-- __init__.py
|   |-- baselines.py
|   |-- common.py
|   |-- dlinear.py
|   |-- fedformer.py
|   |-- itransformer.py
|   |-- owgsm.py
|   |-- patchtst.py
|   |-- timekan.py
|   `-- timexer.py
|-- train.py
|-- requirements.txt
`-- README.md
```

Core files:

- `train.py`: training loop, evaluation, checkpointing, and CLI.
- `models/owgsm.py`: OW-GSM architecture, horizon-weighted loss, and wavelet statistics.
- `data_utils/`: dataset registry, split policies, missing-value interpolation, normalization, and sliding-window loaders.
- `configs/<model>/`: JSON experiment configs grouped by model.
- `models/baselines.py`: baseline registry and factory.
- `models/<baseline>.py`: one implementation file per baseline.

## Installation

```bash
conda create -n owgsm python=3.10
conda activate owgsm
pip install -r requirements.txt
```

Install a CUDA-enabled PyTorch build if your system requires a specific CUDA version. The default `requirements.txt` intentionally stays minimal.

## Quick Start

Train OW-GSM on ETTh1:

```bash
python train.py --config configs/owgsm/paper.json
```

Override a config from the CLI:

```bash
python train.py --config configs/owgsm/paper.json --dataset Weather --pred_len 192
```

Train a baseline with the same data pipeline:

```bash
python train.py --config configs/patchtst/base.json --dataset ETTh1 --pred_len 96
```

Inspect the resolved experiment configuration:

```bash
python train.py --config configs/owgsm/paper.json --print_config
```

Common arguments:

| Argument | Meaning |
| --- | --- |
| `--config` | JSON config path; CLI arguments override the file |
| `--model` | `owgsm`, `itransformer`, `patchtst`, `dlinear`, `timekan`, `timexer`, or `fedformer` |
| `--dataset` | `ETTh1`, `ETTh2`, `ETTm1`, `ETTm2`, `Exchange`, or `Weather` |
| `--seq_len` | Lookback length |
| `--pred_len` | Forecast horizon |
| `--features` | `M`, `S`, or `MS` |
| `--split_policy` | `standard` or `ratio` |
| `--save_dir` | Checkpoint directory |

By default, checkpoints are saved under `checkpoints/`.

## Implemented Models

### OW-GSM

The OW-GSM implementation is in `models/owgsm.py`. It follows the paper architecture:

- `RevIN` for reversible instance normalization.
- `LearnableWaveletSplitter` with low-pass/high-pass filters and orthogonality regularization.
- `GlobalStateRegister` for trend-level global state interaction.
- `DetailEncoder` with patch normalization and bidirectional lightweight Mamba-style blocks.
- Adaptive gated fusion with trend as the stable backbone.
- Per-channel MLP decoder.
- Horizon-weighted MSE and auxiliary feature-disentanglement/wavelet losses.
- Post-training wavelet filter and spectral statistics.

Paper-aligned default configuration:

| Parameter | Value |
| --- | --- |
| Lookback length | `720` |
| Hidden dimension | `32` |
| Wavelet kernel size | `16` |
| GSR tokens | `4` |
| Detail patch size | `8` |
| Mamba-style conv kernel | `4` |
| Mamba expansion factor | `2` |
| Optimizer | `AdamW` |
| Scheduler | `StepLR(step_size=3, gamma=0.5)` |
| Epochs | `10` |
| Batch size | `32` |

Dataset-specific OW-GSM defaults:

| Dataset | LR | Weight Decay | Dropout | RevIN Affine |
| --- | ---: | ---: | ---: | --- |
| ETTh1 | `5e-4` | `5e-5` | `0.4` | yes |
| ETTh2 | `5e-4` | `5e-5` | `0.4` | yes |
| ETTm1 | `5e-4` | `5e-5` | `0.4` | yes |
| ETTm2 | `5e-4` | `5e-5` | `0.4` | yes |
| Weather | `1e-4` | `5e-5` | `0.4` | yes |
| Exchange | `5e-4` | `0.35` | `0.55` | no |

### Baselines

The repository includes compact baseline implementations with a shared training interface. These are useful for controlled local experiments and pipeline checks. For paper-grade reproduction, compare against the official repositories and their exact configs.

| Baseline | Local Name | Official Reference Repository |
| --- | --- | --- |
| iTransformer | `itransformer` | https://github.com/thuml/iTransformer |
| PatchTST | `patchtst` | https://github.com/yuqinie98/PatchTST |
| DLinear | `dlinear` | https://github.com/cure-lab/LTSF-Linear |
| TimeKAN | `timekan` | https://github.com/huangst21/TimeKAN |
| TimeXer | `timexer` | https://github.com/thuml/TimeXer |
| FEDformer | `fedformer` | https://github.com/MAZiqing/FEDformer |

## Datasets

The benchmark CSV files are already included under `dataset/`. Rows are chronological, the first column is `date`, and non-date columns are numeric variables. Missing numerical values, if any, are linearly interpolated by `data_utils/loader.py` before train-only normalization.

| Dataset | File | Domain | Frequency | Time Span in This Repo | Time Steps | Variables | Default Target |
| --- | --- | --- | --- | --- | ---: | ---: | --- |
| ETTh1 | `dataset/ETTh1.csv` | Electricity transformer | 1 hour | 2016-07-01 00:00 to 2018-06-26 19:00 | 17,420 | 7 | `OT` |
| ETTh2 | `dataset/ETTh2.csv` | Electricity transformer | 1 hour | 2016-07-01 00:00 to 2018-06-26 19:00 | 17,420 | 7 | `OT` |
| ETTm1 | `dataset/ETTm1.csv` | Electricity transformer | 15 minutes | 2016-07-01 00:00 to 2018-06-26 19:45 | 69,680 | 7 | `OT` |
| ETTm2 | `dataset/ETTm2.csv` | Electricity transformer | 15 minutes | 2016-07-01 00:00 to 2018-06-26 19:45 | 69,680 | 7 | `OT` |
| Exchange | `dataset/exchange.csv` | Exchange rates | 1 day | 1990-01-01 00:00 to 2010-10-10 00:00 | 7,588 | 8 | `OT` |
| Weather | `dataset/weather.csv` | Meteorology | 10 minutes | 2020-01-01 00:10 to 2021-01-01 00:00 | 52,696 | 21 | `OT` |

`Variables` excludes `date`. In multivariate forecasting (`features=M`), all numeric variables are predicted jointly. In multi-to-single settings (`features=MS`), `OT` is the conventional target column.

Data provenance:

- ETT: [ETDataset](https://github.com/zhouhaoyi/ETDataset), introduced with Informer.
- Exchange: [multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data), used by LSTNet.
- Weather: distributed with [Autoformer](https://github.com/thuml/Autoformer), derived from the Max Planck Institute for Biogeochemistry weather station in Jena.

## Benchmark Protocol

For fair comparison with long-horizon forecasting literature:

- Use chronological train/validation/test splits only.
- Do not shuffle across time when constructing splits.
- Fit normalization statistics on the training split only.
- Apply the training scaler to validation and test splits.
- Report MSE and MAE for every prediction horizon.
- Use common prediction horizons of `96`, `192`, `336`, and `720` time steps unless the paper specifies otherwise.
- Keep `seq_len`, `label_len`, `pred_len`, random seed, optimizer, batch size, and normalization settings fixed across comparable runs.
- For ETT variants, use the established 12/4/4-month split when `--split_policy standard`.
- For Exchange and Weather, `--split_policy standard` uses a chronological 70/10/20 train/validation/test split.

Typical experiment grid:

| Dataset Group | Typical `seq_len` | Typical `pred_len` Values | Task |
| --- | ---: | --- | --- |
| ETTh/ETTm | 96 | 96, 192, 336, 720 | Multivariate long-term forecasting |
| Exchange | 96 | 96, 192, 336, 720 | Multivariate long-term forecasting |
| Weather | 96 | 96, 192, 336, 720 | Multivariate long-term forecasting |

## Reproducibility Checklist

When adding new results, include:

- Dataset name and file path.
- Split rule.
- Input length, label length, and prediction horizon.
- Feature mode, for example `M`, `S`, or `MS`.
- Target column.
- Normalization rule and scaler fitting split.
- Random seed.
- Hardware and software environment.
- Mean and standard deviation over repeated runs if applicable.
- Full MSE and MAE table.
- Commit hash for the released code.

## Citation

Please cite OW-GSM once the paper citation is available:

```bibtex
@misc{owgsm2026,
  title  = {OW-GSM: Orthogonal Wavelet-Structured Global State and Mamba for Multivariate Time Series Forecasting},
  author = {OW-GSM Authors},
  year   = {2026},
  note   = {Citation will be updated after publication}
}
```

If you use the datasets shipped in this repository, also cite the corresponding original benchmark papers:

```bibtex
@inproceedings{zhou2021informer,
  title     = {Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting},
  author    = {Zhou, Haoyi and Zhang, Shanghang and Peng, Jieqi and Zhang, Shuai and Li, Jianxin and Xiong, Hui and Zhang, Wancai},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume    = {35},
  number    = {12},
  pages     = {11106--11115},
  year      = {2021}
}

@inproceedings{lai2018lstnet,
  title     = {Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks},
  author    = {Lai, Guokun and Chang, Wei-Cheng and Yang, Yiming and Liu, Hanxiao},
  booktitle = {The 41st International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages     = {95--104},
  year      = {2018}
}

@inproceedings{wu2021autoformer,
  title     = {Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting},
  author    = {Wu, Haixu and Xu, Jiehui and Wang, Jianmin and Long, Mingsheng},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {34},
  pages     = {22419--22430},
  year      = {2021}
}
```

## Acknowledgements

This repository uses standard public benchmarks from the long-term time-series forecasting literature:

- ETT datasets from [ETDataset](https://github.com/zhouhaoyi/ETDataset).
- Exchange data from [multivariate-time-series-data](https://github.com/laiguokun/multivariate-time-series-data).
- Weather benchmark distributed with [Autoformer](https://github.com/thuml/Autoformer), derived from the Max Planck Institute for Biogeochemistry weather station in Jena.

We also acknowledge the official implementations of iTransformer, PatchTST, DLinear, TimeKAN, TimeXer, and FEDformer listed above.

## References

- Informer / ETDataset: https://github.com/zhouhaoyi/ETDataset
- LSTNet multivariate datasets: https://github.com/laiguokun/multivariate-time-series-data
- Autoformer benchmark repository: https://github.com/thuml/Autoformer
- iTransformer: https://github.com/thuml/iTransformer
- PatchTST: https://github.com/yuqinie98/PatchTST
- DLinear / LTSF-Linear: https://github.com/cure-lab/LTSF-Linear
- TimeKAN: https://github.com/huangst21/TimeKAN
- TimeXer: https://github.com/thuml/TimeXer
- FEDformer: https://github.com/MAZiqing/FEDformer
- Max Planck Jena Beutenberg weather station: https://www.bgc-jena.mpg.de/en/servicegroups/fieldexperiements/locations/beutenberg

## License

The repository license will be specified with the code release. Dataset usage should also follow the terms and licenses of the corresponding upstream dataset providers.
