# LM Research Template

General-purpose JAX/Equinox language model research template. Based on the infrastructure from [End-to-End Test-Time Training](https://arxiv.org/abs/2512.23675), stripped down to a clean, extensible starting point for LM experiments.

## Features

- Standard Transformer with full attention or sliding-window attention
- Distributed training (data + state parallelism) via JAX
- Hydra configuration system with composable experiment configs
- Zarr + Grain data loading pipeline
- Orbax checkpoint management
- Weights & Biases logging
- Multi-node Slurm support via Submitit

## Setup

### Requirements

- **Python** >= 3.12
- **CUDA Toolkit** 12.8.1
- **cuDNN** 9.8.0
- **NCCL** 2.26.2

### Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Dataset

Download Llama-3 tokenized datasets:

```bash
gcloud storage cp -r gs://llama3-dclm-filter-8k/ llama3-dclm-filter-8k
```

Then fill in the `deploy_paths.data` paths in `configs/deploy/interactive.yaml`.

## Running Experiments

### Interactive (single node)

```bash
uv run --exact train \
  +deploy=interactive \
  +experiment=125m-full-attn \
  training.wandb_entity=my-entity \
  training.wandb_project=my-project \
  training.wandb_key=my-key
```

### Dummy data (no dataset needed)

```bash
uv run --exact train \
  +deploy=interactive \
  +experiment=125m-full-attn \
  training.dummy_dataset=true \
  training.log_wandb=false
```

### Multi-node (Slurm)

```bash
uv run --exact train \
  +deploy=submitit \
  hydra.launcher.nodes=4 \
  +experiment=125m-full-attn \
  training.wandb_entity=my-entity \
  training.wandb_project=my-project \
  training.wandb_key=my-key
```

### Resume from checkpoint

```bash
uv run --exact train \
  +deploy=interactive \
  +experiment=125m-full-attn \
  training.resume_exp_name=<previous_exp_name> \
  training.load_part=params
```

## Adding New Architectures

1. Add a new attention/sequence class in `lm/model/attention.py` extending `AttentionBase`
2. Add a `case` in `Block.__init__` in `lm/model/transformer.py`
3. Create a new experiment config under `configs/experiment/`
4. Optionally update sharding rules in `lm/model/sharding.py`

## Project Structure

```
lm/
├── train.py              # Main entry point
├── config.py             # Configuration dataclasses
├── optimizers.py         # AdamW/SGD optimizer builders
├── model/
│   ├── transformer.py    # LanguageModel, CausalLM, TransformerModel, Block
│   ├── attention.py      # Full attention + Sliding window attention
│   ├── loss.py           # Cross-entropy loss
│   ├── loop.py           # Training step + Evaluator
│   ├── data.py           # Batch / BaseModelOutput dataclasses
│   └── sharding.py       # JAX distributed sharding
├── dataloader/
│   └── lm_dataset.py     # Zarr + Grain data pipeline
├── infra/
│   ├── checkpoint.py     # Orbax checkpointing
│   └── wandb_utils.py    # W&B logging
└── utils/
    ├── jax_utils.py      # JAX utilities
    └── filter_utils.py   # Parameter filtering by spec
```
