"""
Drop-in helpers to plug target-weighted validation into train.py.

Integration points:
  1.  Once at training start: ``tw_validator = create_tw_validator(...)``
  2.  Each epoch, after ``eval_loader()``: call ``tw_evaluate()`` with the
      gathered predictions/targets/env_ids from DDP, plus the val_ds.
  3.  Replace checkpoint selection criterion with ``results['select_score']``.

The heavy data-loading in fit_weights() takes ~10-15s once, and the
per-epoch bootstrap evaluation adds < 1s (it's just numpy on ~16k samples).
"""

import numpy as np
import torch
from typing import Dict, Optional

from utils.validation_integration import TargetWeightedValidator


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------

def create_tw_validator(
    data_dir: str = "data/maize_data_2014-2023_vs_2024_v2/",
    val_year: int = 2023,
    test_year: int = 2024,
    weighting_method: str = "kernel",
    fingerprint_method: str = "mean_std",
    tau: Optional[float] = None,
    n_bootstrap: int = 500,
    pessimistic_quantile: float = 0.10,
    min_samples: int = 5,
    verbose: bool = True,
) -> TargetWeightedValidator:
    """
    Create and initialise the validator.  Call once at training start.
    """
    v = TargetWeightedValidator(
        data_dir=data_dir,
        val_year=val_year,
        test_year=test_year,
        weighting_method=weighting_method,
        fingerprint_method=fingerprint_method,
        tau=tau,
        n_bootstrap=n_bootstrap,
        pessimistic_quantile=pessimistic_quantile,
        min_samples=min_samples,
        verbose=verbose,
    )
    v.fit_weights()
    return v


# --------------------------------------------------------------------------
# Per-epoch evaluation hook
# --------------------------------------------------------------------------

def tw_evaluate(
    validator: TargetWeightedValidator,
    full_preds: torch.Tensor,   # shape (N,) — gathered from DDP
    full_targets: torch.Tensor, # shape (N,)
    full_env_ids: torch.Tensor, # shape (N,) — integer codes from pd.Categorical
    val_ds,                     # the GxE_Dataset(split="val") instance
    y_scalers: Optional[Dict] = None,
) -> Dict:
    """
    Compute target-weighted validation score from the same tensors
    produced by ``_gather_predictions()`` in train.py.

    Parameters
    ----------
    full_preds, full_targets : torch.Tensor
        May be in *scaled* space if scale_targets=True.
    full_env_ids : torch.Tensor (long)
        Integer env codes from ``val_ds.env_codes.codes``.
    val_ds : GxE_Dataset
        The validation dataset — used to map int codes → Env strings.
    y_scalers : dict or None
        If scale_targets is True, pass ``val_ds.label_scalers`` so we can
        inverse-transform before computing correlations.  If None or
        scale_targets was False, predictions and targets are used as-is.

    Returns
    -------
    dict — same as ``TargetWeightedValidator.evaluate()`` output, with
    additional key ``'raw_env_pcc'`` for the unweighted macro-avg.
    """
    # ---- Move to numpy ----
    preds_np = full_preds.detach().cpu().float().numpy().ravel()
    targs_np = full_targets.detach().cpu().float().numpy().ravel()
    env_ids_int = full_env_ids.detach().cpu().long().numpy().ravel()

    # ---- Inverse-transform if scaled ----
    if y_scalers and "total" in y_scalers:
        ls = y_scalers["total"]
        preds_np = ls.inverse_transform(preds_np)
        targs_np = ls.inverse_transform(targs_np)

    # ---- Map integer env codes → string Env names ----
    categories = val_ds.env_codes.categories  # pd.Index of str
    env_names = np.array([str(categories[c]) for c in env_ids_int])

    # ---- Run the target-weighted evaluation ----
    results = validator.evaluate(preds_np, targs_np, env_names)
    return results
