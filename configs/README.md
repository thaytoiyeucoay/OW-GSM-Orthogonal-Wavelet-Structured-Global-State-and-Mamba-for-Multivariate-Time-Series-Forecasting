# Experiment Configs

Each model has its own config folder. JSON keys map directly to `ExperimentConfig` in `train.py`.

Examples:

```bash
python train.py --config configs/owgsm/paper.json
python train.py --config configs/patchtst/base.json --dataset Weather --pred_len 192
```

CLI arguments override values loaded from the JSON file.
