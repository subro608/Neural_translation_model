Train (YAML config):
  python train_translator_paired.py --config configs/translator.yaml

Test (YAML config):
  python test_translator_paired.py --config configs/translator.yaml

Notes:
- CLI flags override YAML values if provided (e.g., add `--device cuda`).
- To enable Weights & Biases logging, set `wandb_off: false` and fill `wandb_project` and `wandb_run_name` in `configs/translator.yaml`.