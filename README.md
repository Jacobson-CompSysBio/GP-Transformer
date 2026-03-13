# GP-Transformer

A Transformer-based model for **Genomic Prediction (GP)** of maize grain yield, developed for the [Genomes to Fields (G2F) Initiative](https://www.genomes2fields.org/) prediction competition. The model predicts hybrid maize yield across diverse field environments by jointly learning from high-dimensional genotypic markers and environmental covariates.

## Competition Goal

The G2F competition challenges participants to predict grain yield for maize hybrids grown across North American field trials spanning multiple years (2014–2024). The key difficulty is **Genotype × Environment (GxE) interaction:** the same hybrid performs differently across environments due to soil, weather, management, and their interactions with genetic background. Models are evaluated on **macro-averaged per-environment Pearson correlation coefficient (PCC)**, which rewards consistent within-environment ranking of hybrids rather than absolute yield prediction.

## Data

| Split | Years | Purpose |
|-------|-------|---------|
| Train | 2014–2022 | Model fitting |
| Validation | 2023 (or LEO holdout) | Early stopping & hyperparameter selection |
| Test | 2024 | Final evaluation |

Each sample consists of:
- **Genotype (G):** 2,024 SNP marker tokens (encoded as 0 / 0.5 / 1) or GRM-standardized continuous features
- **Environment (E):** 705 numeric covariates (weather, soil, management) + optional categorical features (irrigation, treatment, previous crop)
- **Target:** Grain yield (mg/Ha)

Data files are expected under `data/` following the `maize_data_*` directory structure with `X_train.csv`, `y_train.csv`, `X_test.csv`, `y_test.csv`.

## Architecture

### Model Variants

| Model | Class | Description |
|-------|-------|-------------|
| **FullTransformer** | `FullTransformer` | Primary architecture. Tokenizes all inputs (CLS + SNP markers + env features) into a single sequence, applies sinusoidal positional encoding, and processes through Pre-LN transformer blocks. A CLS token pools the final representation for yield regression. |
| **FullTransformerResidual** | `FullTransformerResidual` | Extends FullTransformer with a residual decomposition head: predicts env-mean yield + genotype-specific residual separately. |
| **GxE_Transformer** | `GxE_Transformer` | Three-prong encoder architecture with independent G, E, and LD encoders fused via learned weighted gating, then processed by a GxE interaction module (transformer, MLP, or CNN). |
| **GxE_ResidualTransformer** | `GxE_ResidualTransformer` | Adds residual yield decomposition to the three-prong architecture. |

### Core Components

| Component | Description |
|-----------|-------------|
| **G_Encoder** | Transformer encoder for genotype tokens with CLS pooling. Supports dense or MoE blocks. |
| **E_Encoder** | Multi-layer MLP with residual connections for environmental covariates. |
| **LD_Encoder** | 1D ResNet CNN processing one-hot encoded markers for linkage-disequilibrium patterns. |
| **MoE Layer** | Mixture-of-Experts with Top-K gating, optional shared expert, and load-balancing loss. |
| **FlashAttention** | Bidirectional self-attention via `F.scaled_dot_product_attention`. |
| **Stochastic Depth** | Linearly increasing drop-path rate across transformer blocks. |

### GxE Interaction Backends

The three-prong model supports three GxE fusion strategies (set via `GXE_ENC`):

- **`tf`** — Transformer blocks over the concatenated G+E token sequence (default)
- **`mlp`** — Residual MLP blocks over mean-pooled representations
- **`cnn`** — 1D ResNet blocks over the concatenated sequence

## Training

Training uses **PyTorch DDP** across multiple AMD MI250X GPUs on ORNL Frontier. Key settings are configured as environment variables in `train.slurm`.

### Hyperparameters

| Parameter | Variable | Default | Description |
|-----------|----------|---------|-------------|
| Global batch size | `GBS` | 1024 | Divided evenly across GPUs |
| Learning rate | `LR` | 1e-4 | AdamW with cosine schedule |
| Weight decay | `WEIGHT_DECAY` | 1e-5 | AdamW regularization |
| Embedding dim | `EMB_SIZE` | 256 | Transformer hidden dimension |
| Attention heads | `HEADS` | 4 | Multi-head attention |
| GxE layers | `GXE_LAYERS` | 1 | Transformer depth for GxE interaction |
| Dropout | `DROPOUT` | 0.15 | Applied throughout |
| Early stopping | `EARLY_STOP` | 200 | Patience (epochs) |
| Max epochs | `NUM_EPOCHS` | 3000 | Upper bound for training |

### Loss Functions

Composite losses are specified via `LOSS` (e.g., `"envpcc"`, `"mse+envpcc"`) with per-component weights in `LOSS_WEIGHTS`. When `LOSS` contains multiple nonzero terms, training applies PCGrad to the weighted per-term gradients; validation and checkpoint metrics still use the weighted scalar sum.

| Loss | Description |
|------|-------------|
| `envpcc` | 1 − macro-averaged per-environment Pearson correlation (with Fisher z-transform) |
| `pcc` | 1 − global Pearson correlation |
| `mse` | Mean squared error |
| `envspearman` | 1 − macro-averaged per-environment Spearman correlation (differentiable via soft ranks) |
| `spearman` | 1 − global differentiable Spearman correlation |
| `envmse` | Macro-averaged per-environment MSE |
| `ktau` | Differentiable Kendall-tau ranking loss |
| `xi` | Chatterjee's Xi coefficient loss |

### Contrastive Learning (Optional)

Auxiliary contrastive objectives align learned representations with known genetic/environmental similarity structures.

| Variable | Options | Description |
|----------|---------|-------------|
| `CONTRASTIVE_MODE` | `none`, `g`, `e`, `g+e` | Which contrastive heads to activate |
| `CONTRASTIVE_SIM_TYPE` | `grm`, `ibs` | Genetic similarity metric |
| `CONTRASTIVE_LOSS_TYPE` | `mse`, `cosine`, `kl` | How to match embedding vs. target similarity |

### Validation Strategies

| Strategy | Variable | Description |
|----------|----------|-------------|
| Year-based | default | Train ≤ 2022, validate on 2023 |
| **LEO** | `LEO_VAL=True` | Leave-Environment-Out: hold out 15% of environments (all years) |
| Env-stratified batching | `ENV_STRATIFIED=True` | Ensures each batch contains ≥ `MIN_SAMPLES_PER_ENV` samples per environment for stable envwise loss computation |

## Usage

### Submit a Training Job

```bash
sbatch train.slurm
```

Edit the environment variables at the top of `train.slurm` to configure the model and hyperparameters. Training logs, checkpoints, and evaluation results are saved to `logs/`, `checkpoints/`, and `data/results/` respectively. Metrics are tracked via [Weights & Biases](https://wandb.ai/).

### Key Environment Variables

```bash
# Architecture selection
FULL_TRANSFORMER=True       # Use FullTransformer (vs. three-prong GxE)
G_ENC=True                  # Enable genotype encoder (three-prong only)
E_ENC=True                  # Enable environment encoder (three-prong only)
LD_ENC=True                 # Enable LD encoder (three-prong only)
GXE_ENC=tf                  # GxE backend: tf, mlp, cnn
WG=True                     # Weighted gating for three-prong fusion

# Genotype input
G_INPUT_TYPE=tokens          # "tokens" (discrete SNPs) or "grm" (continuous)

# MoE settings
G_ENCODER_TYPE=moe           # "dense" or "moe"
MOE_NUM_EXPERTS=8
MOE_TOP_K=2
MOE_SHARED_EXPERT=True
```

## Project Structure

```
GP-Transformer/
├── models/
│   ├── config.py          # Config dataclass (all hyperparameters)
│   ├── model.py           # FullTransformer, GxE_Transformer, and residual variants
│   ├── transformer.py     # Attention, MLP, TransformerBlock, MoE blocks, G_Encoder
│   ├── moe.py             # Mixture-of-Experts layer (Top-K gating, shared expert)
│   ├── mlp.py             # MLP blocks, E_Encoder
│   └── cnn.py             # 1D ResNet blocks, LD_Encoder
├── scripts/
│   ├── train.py           # DDP training loop with checkpointing
│   ├── eval.py            # Checkpoint evaluation, metrics, and result export
│   ├── train_residual.py  # Training script for residual model variants
│   └── train_rolling.py   # Rolling-window temporal training
├── utils/
│   ├── dataset.py         # GxE_Dataset, data loading, env-stratified sampling
│   ├── loss.py            # All loss functions (envpcc, contrastive, ranking, etc.)
│   ├── utils.py           # CLI parsing, run naming, samplers, helpers
│   └── get_lr.py          # Cosine learning rate schedule
├── data/                  # Competition datasets (not tracked)
├── train.slurm            # Primary SLURM job script
├── best_train.slurm       # Best-config training script
└── notebooks/             # Analysis notebooks
```

## Requirements

- Python 3.9+
- PyTorch 2.0+ (with ROCm support for AMD GPUs)
- `wandb`, `scipy`, `scikit-learn`, `python-dotenv`, `tqdm`
- Optional: `torchsort` (for differentiable Spearman loss)

## Hardware

Developed and tested on [ORNL Frontier](https://www.olcf.ornl.gov/frontier/) — AMD MI250X GPUs with ROCm 6.3.1, NCCL/RCCL backend for distributed training.
