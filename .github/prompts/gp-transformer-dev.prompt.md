---
description: "Develop and iterate on GP-Transformer validation and loss optimization for maize yield prediction. Use when: working on validation schemes, loss functions, training strategies, or model evaluation for the G2F genomic prediction competition."
agent: "agent"
---

# GP-Transformer Development Assistant

You are helping develop a Transformer-based genomic prediction model for maize grain yield in the [G2F competition](https://www.genomes2fields.org/). The model predicts hybrid maize yield across diverse North American field environments.

## Project Context

Before starting ANY work, read these files to understand current project state:

1. [Project README](../../README.md) — architecture, training, data structure
2. [Shared memory files](../../shared_memory/) — all files in this folder contain evolving project knowledge including validation failure analysis, target-weighted validation design, and rolling training documentation
3. [Loss functions](../../utils/loss.py) — available losses and composition engine
4. [Training loop](../../scripts/train.py) — DDP training, validation flow, checkpoint selection
5. [Dataset](../../utils/dataset.py) — data splits, environment-stratified batching
6. [Target-weighted validation](../../utils/target_weighted_validation.py) — TW fingerprinting, kernel weights, bootstrap selection

## Competition Evaluation

The test metric is **macro-averaged per-environment Pearson correlation coefficient (PCC)** across all 2024 environments. This rewards consistent **within-environment ranking** of hybrids, not absolute yield accuracy.

## Critical Domain Constraints

- **Train**: 2014–2022, **Validation**: 2023, **Test**: 2024
- 90.2% of test hybrids are novel (unseen in training)
- 53% of test hybrids use tester LH287 which has **zero training rows**
- Breeding panels rotate every 2 years — tester lines change between panels
- No holdout from training data can test LH287 combining ability
- LEO validation is incompatible with target-weighted validation (see shared_memory)

## Current Goals

### 1. Validation Scheme — Track Test Performance Monotonically

Develop a validation metric that **ranks hyperparameter configurations in the same order** as the 2024 test metric. Current challenges documented in `shared_memory/VALIDATION_FAILURE_ANALYSIS.md`:

- LEO tests spatial generalization but never sees novel genotypes
- LYO-2023 has only 0.2% novel hybrids
- LYO-2022 has different testers (LH244) and low half-sib connectivity
- Short-budget rolling CV measures learning speed, not converged quality
- Target-weighted validation addresses covariate shift but hasn't been validated for ranking fidelity

**Success criterion**: Spearman ρ > 0.8 between validation ranking and test ranking across multiple hyperparameter configurations. Given the structural difficulty, any statistically significant monotonic relationship is acceptable as a starting point.

When proposing validation changes:
- Explain WHY the approach addresses the structural mismatch documented in VALIDATION_FAILURE_ANALYSIS.md
- Consider whether the approach tests the same capabilities required for 2024 (novel hybrid prediction + novel weather + tester transfer)
- Track validation-vs-test rank correlation across multiple configurations, not just a single comparison

### 2. Optimization Scheme — Ranking + Scaling

Develop a loss/training strategy that achieves both, in priority order:
1. **Within-environment ranking** (primary): correctly ordering hybrid yields — Pearson correlation is the most important metric
2. **Across-environment scaling** (secondary): low RMSE for accurate absolute magnitude predictions

Available loss components: `envpcc`, `envmse`, `envspearman`, `mse`, `pcc`, `spearman`, `ktau`, `xi`, `triplet`

When proposing loss changes:
- **Pearson correlation is king** — never sacrifice correlation for lower RMSE
- Correlation losses alone produce correct rankings but uncalibrated scale
- MSE alone optimizes absolute error but may sacrifice within-environment ranking
- Composite losses need careful weighting — the gradient magnitudes of different terms vary by orders of magnitude
- Environment-stratified batching (`ENV_STRATIFIED=True`) is required for stable per-environment losses
- When combining correlation + MSE terms, ensure the MSE component acts as a regularizer for scale, not as the dominant gradient signal

## Implementation Guidelines

- All training runs on ORNL Frontier (AMD MI250X GPUs) via SLURM — modify `.slurm` files for new configs
- Use PyTorch DDP patterns — changes to training loop must be rank-aware
- Log all new metrics to W&B with descriptive keys under `val_loss/` or `train_loss/` namespaces
- Test changes at small scale first (`debug.slurm` with reduced epochs/data) before full runs

## Shared Memory Protocol

The `shared_memory/` folder is the project's evolving knowledge base. When working on this project:

- **Read ALL shared_memory files** at the start of every session to avoid repeating failed approaches
- **Update existing files** when new results refine or contradict prior analysis (use `str_replace`)
- **Create new files** only for genuinely new topics — prefer extending existing docs
- **Name files** with SCREAMING_SNAKE_CASE describing the topic (e.g., `LOSS_ABLATION_RESULTS.md`)
