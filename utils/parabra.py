"""Python functions for the PARaBra MegaSEM model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import linalg


ProgressFn = Callable[[str], None]


@dataclass(frozen=True)
class RidgeResult:
    """Store one variance-ridge fit."""

    predictions: np.ndarray
    intercept: float
    h2: float
    iterations: int
    ridge: float
    converged: bool


@dataclass(frozen=True)
class MegaSEMResult:
    """Store the full-rank MegaSEM predictions and fit facts."""

    predictions: np.ndarray
    first_stage: list[dict[str, float | int | bool | str]]
    second_stage: list[dict[str, float | int | bool | str]]
    singular_values: np.ndarray


def mean_impute_markers(markers: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace each missing marker value with its column mean."""

    values = np.asarray(markers, dtype=np.float32).copy()
    missing = np.isnan(values)
    missing_count = int(missing.sum())
    if not missing_count:
        return values, 0

    means = np.nanmean(values, axis=0)
    if np.isnan(means).any():
        bad = np.flatnonzero(np.isnan(means))
        raise ValueError(f"Marker columns have no finite values: {bad[:10].tolist()}")
    rows, columns = np.nonzero(missing)
    values[rows, columns] = means[columns]
    return values, missing_count


def author_gaussian_kernel(markers: np.ndarray, phi: float = 1.0) -> tuple[np.ndarray, float]:
    """Create the bWGR EigenGAU kernel used by the author code."""

    values = np.asarray(markers, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("The marker matrix must contain at least two rows.")
    if not np.isfinite(values).all():
        raise ValueError("The marker matrix contains a nonfinite value.")
    if not np.isfinite(phi) or phi <= 0:
        raise ValueError("phi must be positive and finite.")

    row_norms = np.einsum("ij,ij->i", values, values, dtype=np.float64).astype(np.float32)
    kernel = np.empty((values.shape[0], values.shape[0]), dtype=np.float32)
    np.matmul(values, values.T, out=kernel)
    kernel *= np.float32(-2.0)
    kernel += row_norms[:, None]
    kernel += row_norms[None, :]
    np.maximum(kernel, np.float32(0.0), out=kernel)
    np.sqrt(kernel, out=kernel)
    np.fill_diagonal(kernel, np.float32(0.0))

    count = values.shape[0] * (values.shape[0] - 1)
    mean_distance = float(kernel.sum(dtype=np.float64) / count)
    if not np.isfinite(mean_distance) or mean_distance <= 0:
        raise ValueError("The mean marker distance must be positive and finite.")

    kernel *= np.float32(-phi / mean_distance)
    np.exp(kernel, out=kernel)
    np.fill_diagonal(kernel, np.float32(1.0))
    return kernel, mean_distance


def kernel_feature_gram(kernel: np.ndarray) -> np.ndarray:
    """Create the feature Gram matrix that matches bWGR K2X."""

    values = np.asarray(kernel, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("The kernel must be a square matrix.")
    gram = np.empty_like(values)
    np.matmul(values, values.T, out=gram)
    return gram


def _ridge_updates(
    eigenvalues: np.ndarray,
    signal: np.ndarray,
    y_centered: np.ndarray,
    trace_xsx: float,
    fit_weights: np.ndarray,
    change_weights: np.ndarray,
    df0: float,
    tolerance: float,
    max_iterations: int,
) -> tuple[np.ndarray, float, float, int, bool]:
    """Solve the bWGR variance-ridge fixed point in a spectral basis."""

    n = y_centered.size
    vy = float(y_centered @ y_centered / (n - 1))
    msx = trace_xsx / (n - 1)
    if vy <= 0 or not np.isfinite(vy):
        raise ValueError("The response variance must be positive and finite.")
    if msx <= 0 or not np.isfinite(msx):
        raise ValueError("The feature variance must be positive and finite.")

    ve = 0.5 * vy
    vb = 0.5 * vy / msx
    ve0 = ve * df0
    vb0 = vb * df0
    ridge = ve / vb
    old_rotated = np.zeros_like(signal)
    converged = False

    for iteration in range(1, max_iterations + 1):
        rotated = signal / (eigenvalues + ridge)
        fitted_inner = float((fit_weights * rotated) @ signal)

        ve = (float(y_centered @ y_centered) - fitted_inner + ve0) / (n + df0)
        vb = (fitted_inner + vb0) / (trace_xsx + df0)
        ve = max(ve, np.finfo(np.float64).tiny)
        vb = max(vb, np.finfo(np.float64).tiny)
        new_ridge = ve / vb

        delta = rotated - old_rotated
        change = float((delta * change_weights) @ delta)
        old_rotated = rotated
        ridge = new_ridge
        if not np.isfinite(change) or change < tolerance:
            converged = bool(np.isfinite(change))
            break

    h2 = 1.0 - ve / vy
    return rotated, float(h2), float(ridge), iteration, converged


def fit_variance_ridge_dual(
    gram_observed: np.ndarray,
    gram_cross: np.ndarray,
    response: np.ndarray,
    *,
    df0: float = 20.0,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> RidgeResult:
    """Fit one bWGR variance-ridge model through its sample Gram matrix."""

    y = np.asarray(response, dtype=np.float64)
    observed = np.asarray(gram_observed, dtype=np.float64)
    cross = np.asarray(gram_cross)
    n = y.size
    if n < 2 or observed.shape != (n, n) or cross.shape[1] != n:
        raise ValueError("The dual ridge matrix shapes do not agree.")
    if not np.isfinite(y).all() or not np.isfinite(observed).all():
        raise ValueError("The dual ridge inputs contain a nonfinite value.")

    intercept = float(y.mean())
    y_centered = y - intercept
    row_mean = observed.mean(axis=1, keepdims=True)
    centered = observed - row_mean - row_mean.T + float(observed.mean())
    centered = 0.5 * (centered + centered.T)
    trace_xsx = float(np.trace(centered))

    eigenvalues, eigenvectors = linalg.eigh(
        centered,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    signal = eigenvectors.T @ y_centered
    rotated, h2, ridge, iterations, converged = _ridge_updates(
        eigenvalues,
        signal,
        y_centered,
        trace_xsx,
        eigenvalues,
        eigenvalues,
        df0,
        tolerance,
        max_iterations,
    )

    alpha = eigenvectors @ rotated
    alpha -= alpha.mean()
    alpha_for_product = alpha.astype(cross.dtype, copy=False)
    genetic = np.asarray(cross @ alpha_for_product, dtype=np.float64)
    return RidgeResult(
        predictions=genetic,
        intercept=intercept,
        h2=h2,
        iterations=iterations,
        ridge=ridge,
        converged=converged,
    )


def fit_variance_ridge_primal(
    features_observed: np.ndarray,
    features_predict: np.ndarray,
    response: np.ndarray,
    *,
    df0: float = 20.0,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
) -> RidgeResult:
    """Fit one bWGR variance-ridge model through its feature matrix."""

    y = np.asarray(response, dtype=np.float64)
    observed = np.asarray(features_observed, dtype=np.float64)
    predict = np.asarray(features_predict, dtype=np.float64)
    n = y.size
    if n < 2 or observed.shape[0] != n or predict.shape[1] != observed.shape[1]:
        raise ValueError("The primal ridge matrix shapes do not agree.")
    if not np.isfinite(y).all() or not np.isfinite(observed).all():
        raise ValueError("The primal ridge inputs contain a nonfinite value.")

    intercept = float(y.mean())
    y_centered = y - intercept
    centered = observed - observed.mean(axis=0, keepdims=True)
    xtx = centered.T @ centered
    xtx = 0.5 * (xtx + xtx.T)
    trace_xsx = float(np.trace(xtx))

    eigenvalues, eigenvectors = linalg.eigh(
        xtx,
        overwrite_a=True,
        check_finite=False,
        driver="evd",
    )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    tilde = centered.T @ y_centered
    signal = eigenvectors.T @ tilde
    rotated, h2, ridge, iterations, converged = _ridge_updates(
        eigenvalues,
        signal,
        y_centered,
        trace_xsx,
        np.ones_like(eigenvalues),
        np.ones_like(eigenvalues),
        df0,
        tolerance,
        max_iterations,
    )

    coefficients = eigenvectors @ rotated
    predictions = predict @ coefficients + intercept
    return RidgeResult(
        predictions=np.asarray(predictions, dtype=np.float64),
        intercept=intercept,
        h2=h2,
        iterations=iterations,
        ridge=ridge,
        converged=converged,
    )


def fit_full_rank_megasem(
    response: np.ndarray,
    feature_gram: np.ndarray,
    fit_indices: np.ndarray,
    trait_names: Sequence[str],
    *,
    progress: ProgressFn | None = None,
) -> MegaSEMResult:
    """Fit the full-rank ZSEMF model with an exact spectral ridge solver."""

    y = np.asarray(response, dtype=np.float64)
    gram = np.asarray(feature_gram, dtype=np.float32)
    indices = np.asarray(fit_indices, dtype=np.int64)
    if y.ndim != 2 or len(trait_names) != y.shape[1]:
        raise ValueError("The response matrix and trait names do not agree.")
    if indices.size != y.shape[0]:
        raise ValueError("The fit index count does not match the response rows.")
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("The feature Gram matrix must be square.")
    if indices.min() < 0 or indices.max() >= gram.shape[0]:
        raise ValueError("A fit index is outside the feature Gram matrix.")

    n_all = gram.shape[0]
    trait_count = y.shape[1]
    first_predictions = np.empty((n_all, trait_count), dtype=np.float32)
    first_facts: list[dict[str, float | int | bool | str]] = []

    for column, name in enumerate(trait_names):
        valid = np.isfinite(y[:, column])
        observed_indices = indices[valid]
        cross = gram[:, observed_indices]
        observed_gram = np.asarray(cross[observed_indices, :], dtype=np.float64)
        fit = fit_variance_ridge_dual(observed_gram, cross, y[valid, column])
        first_predictions[:, column] = fit.predictions.astype(np.float32)
        first_facts.append(
            {
                "trait": str(name),
                "observed": int(valid.sum()),
                "h2": fit.h2,
                "iterations": fit.iterations,
                "ridge": fit.ridge,
                "converged": fit.converged,
            }
        )
        if progress and ((column + 1) % 10 == 0 or column + 1 == trait_count):
            progress(f"First ZSEMF stage: {column + 1}/{trait_count} traits")

    first_fit = np.asarray(first_predictions[indices, :], dtype=np.float64)
    _, singular_values, right_vectors_t = linalg.svd(
        first_fit,
        full_matrices=False,
        check_finite=False,
        lapack_driver="gesdd",
    )
    right_vectors = right_vectors_t.T
    latent_all = np.asarray(first_predictions, dtype=np.float64) @ right_vectors
    latent_fit = latent_all[indices, :]

    predictions = np.empty((n_all, trait_count), dtype=np.float32)
    second_facts: list[dict[str, float | int | bool | str]] = []
    for column, name in enumerate(trait_names):
        valid = np.isfinite(y[:, column])
        fit = fit_variance_ridge_primal(latent_fit[valid, :], latent_all, y[valid, column])
        predictions[:, column] = fit.predictions.astype(np.float32)
        second_facts.append(
            {
                "trait": str(name),
                "observed": int(valid.sum()),
                "h2": fit.h2,
                "iterations": fit.iterations,
                "ridge": fit.ridge,
                "converged": fit.converged,
            }
        )
        if progress and ((column + 1) % 10 == 0 or column + 1 == trait_count):
            progress(f"Second ZSEMF stage: {column + 1}/{trait_count} traits")

    return MegaSEMResult(
        predictions=predictions,
        first_stage=first_facts,
        second_stage=second_facts,
        singular_values=singular_values,
    )


def aggregate_state_predictions(
    environment_predictions: np.ndarray,
    environment_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    """Average environment predictions within each two-character state key."""

    values = np.asarray(environment_predictions)
    if values.ndim != 2 or values.shape[1] != len(environment_names):
        raise ValueError("The environment prediction shape does not agree with its names.")
    states = list(dict.fromkeys(str(name)[:2] for name in environment_names))
    output = np.empty((values.shape[0], len(states)), dtype=np.float32)
    names = np.asarray([str(name)[:2] for name in environment_names])
    for column, state in enumerate(states):
        output[:, column] = values[:, names == state].mean(axis=1)
    return output, states


def within_environment_zscore(frame: pd.DataFrame, column: str) -> pd.Series:
    """Standardize one prediction column within each environment."""

    means = frame.groupby("Env", sort=False)[column].transform("mean")
    scales = frame.groupby("Env", sort=False)[column].transform(lambda x: x.std(ddof=0))
    if scales.isna().any() or (scales <= 0).any():
        bad = frame.loc[scales.isna() | (scales <= 0), "Env"].unique().tolist()
        raise ValueError(f"Prediction variance is not positive for environments: {bad}")
    return (frame[column] - means) / scales


def fixed_equal_zscore_ensemble(
    template_predictions: pd.DataFrame,
    parabra_column: str = "ParabraPred",
    transformer_column: str = "TransformerPred",
) -> pd.DataFrame:
    """Blend two models after within-environment z-scores on the input rows."""

    output = template_predictions.copy()
    output["ParabraZ"] = within_environment_zscore(output, parabra_column)
    output["TransformerZ"] = within_environment_zscore(output, transformer_column)
    output["EnsemblePred"] = 0.5 * (output["ParabraZ"] + output["TransformerZ"])
    return output


def environment_scores(
    frame: pd.DataFrame,
    prediction_column: str,
    actual_column: str = "Actual",
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Calculate the official macro environment PCC and related facts."""

    rows: list[dict[str, float | int | str]] = []
    for environment, group in frame.groupby("Env", sort=True):
        actual = group[actual_column].to_numpy(dtype=np.float64)
        predicted = group[prediction_column].to_numpy(dtype=np.float64)
        if actual.size < 2 or np.std(actual) == 0 or np.std(predicted) == 0:
            pcc = np.nan
        else:
            pcc = float(np.corrcoef(actual, predicted)[0, 1])
        rows.append(
            {
                "Env": str(environment),
                "n": int(actual.size),
                "pcc": pcc,
            }
        )

    per_environment = pd.DataFrame(rows)
    valid = per_environment["pcc"].notna()
    if not valid.all():
        bad = per_environment.loc[~valid, "Env"].tolist()
        raise ValueError(f"The PCC is not valid for environments: {bad}")
    weights = per_environment.loc[valid, "n"].to_numpy(dtype=np.float64)
    pcc_values = per_environment.loc[valid, "pcc"].to_numpy(dtype=np.float64)
    summary: dict[str, float | int] = {
        "rows": int(len(frame)),
        "environments": int(valid.sum()),
        "macro_environment_pcc": float(pcc_values.mean()),
        "weighted_environment_pcc": float(np.average(pcc_values, weights=weights)),
        "global_pcc": float(np.corrcoef(frame[actual_column], frame[prediction_column])[0, 1]),
    }
    return per_environment, summary
