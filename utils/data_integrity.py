"""Fail-closed helpers for canonical G2F row identity and X/y alignment.

Canonical sample ids have the form ``<Env>-<Hybrid>``.  Environment names may
themselves contain hyphens, so splitting an id on its first hyphen is not safe.
Replicated plots also share the same id, so X and y must be joined by physical
row position after verifying their row-wise ids; ``merge(on="id")`` is unsafe.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd


def extract_hybrid_name(sample_id: object, env: object) -> str:
    """Remove the exact ``<Env>-`` prefix from one canonical sample id.

    The function deliberately raises on malformed data instead of guessing a
    split point.  This protects environment names such as ``TXH1-Dry_2017``.
    """

    if pd.isna(sample_id) or pd.isna(env):
        raise ValueError(f"sample id and Env must be non-missing: id={sample_id!r}, Env={env!r}")
    value = str(sample_id).strip()
    env_value = str(env).strip()
    if not env_value:
        raise ValueError(f"Env must be non-empty for id={value!r}")
    prefix = f"{env_value}-"
    if not value.startswith(prefix):
        raise ValueError(
            f"Canonical id {value!r} does not start with its exact Env prefix {prefix!r}"
        )
    hybrid = value[len(prefix) :].strip()
    if not hybrid:
        raise ValueError(f"Canonical id {value!r} has an empty hybrid suffix")
    return hybrid


def extract_hybrid_series(
    sample_ids: Iterable[object] | pd.Series,
    envs: Iterable[object] | pd.Series,
    *,
    source: str = "canonical rows",
) -> pd.Series:
    """Vectorized, index-preserving exact-prefix hybrid extraction."""

    ids = sample_ids.copy() if isinstance(sample_ids, pd.Series) else pd.Series(sample_ids)
    env = envs.copy() if isinstance(envs, pd.Series) else pd.Series(envs)
    if len(ids) != len(env):
        raise ValueError(f"{source}: id/Env length mismatch: {len(ids)} != {len(env)}")

    # Pair by physical position, not by potentially different Series indexes.
    result: list[str] = []
    failures: list[str] = []
    for position, (sample_id, env_value) in enumerate(
        zip(ids.to_numpy(dtype=object), env.to_numpy(dtype=object))
    ):
        try:
            result.append(extract_hybrid_name(sample_id, env_value))
        except ValueError as exc:
            if len(failures) < 3:
                failures.append(f"row {position}: {exc}")
            result.append("")
    if failures:
        raise ValueError(f"{source}: malformed canonical ids; " + "; ".join(failures))
    return pd.Series(result, index=ids.index, name="Hybrid", dtype="object")


def split_hybrid_parents(
    hybrid: pd.Series, *, strict: bool = True
) -> tuple[pd.Series, pd.Series]:
    """Split ``Parent1/Parent2`` while preserving the input index.

    Canonical competition hybrids contain exactly one slash and two non-empty
    parents.  Strict mode is the default so malformed entity keys cannot
    silently create bogus parent groups.
    """

    values = hybrid.astype("string").fillna("").str.strip()
    if strict:
        slash_counts = values.str.count("/")
        invalid = (slash_counts != 1) | values.str.startswith("/") | values.str.endswith("/")
        if invalid.any():
            examples = values.loc[invalid].head(3).tolist()
            raise ValueError(
                "Canonical hybrids must contain exactly one slash and two non-empty "
                f"parents; examples: {examples}"
            )

    parts = values.str.split("/", n=1, expand=True)
    parent1 = parts[0].fillna("").astype(str).str.strip()
    if parts.shape[1] > 1:
        parent2 = parts[1].fillna("").astype(str).str.strip()
    else:
        parent2 = pd.Series("", index=hybrid.index, dtype="object")
    return parent1, parent2


def split_hybrid_name(hybrid: object) -> tuple[str, str]:
    """Strict scalar counterpart of :func:`split_hybrid_parents`."""

    if pd.isna(hybrid):
        raise ValueError("Canonical hybrid must be non-missing")
    value = str(hybrid).strip()
    if value.count("/") != 1:
        raise ValueError(
            f"Canonical hybrid must contain exactly one slash: {value!r}"
        )
    parent1, parent2 = (part.strip() for part in value.split("/", 1))
    if not parent1 or not parent2:
        raise ValueError(
            f"Canonical hybrid must contain two non-empty parents: {value!r}"
        )
    return parent1, parent2


def validate_xy_physical_alignment(
    x: pd.DataFrame,
    y: pd.DataFrame,
    *,
    source: str = "canonical X/y",
) -> None:
    """Validate exact physical X/y row identity without copying either frame."""

    if "id" not in x.columns or "id" not in y.columns:
        raise ValueError(f"{source}: both X and y must contain an 'id' column")
    if len(x) != len(y):
        raise ValueError(f"{source}: physical row count mismatch: X={len(x)}, y={len(y)}")

    for frame_name, frame in (("X", x), ("y", y)):
        invalid_id = frame["id"].isna() | frame["id"].astype("string").str.strip().eq("")
        if invalid_id.any():
            position = int(np.flatnonzero(invalid_id.to_numpy(dtype=bool))[0])
            raise ValueError(
                f"{source}: {frame_name} has a missing/empty id at physical row {position}"
            )
        if "Env" in frame.columns:
            invalid_env = frame["Env"].isna() | frame["Env"].astype("string").str.strip().eq("")
            if invalid_env.any():
                position = int(np.flatnonzero(invalid_env.to_numpy(dtype=bool))[0])
                raise ValueError(
                    f"{source}: {frame_name} has a missing/empty Env at physical row {position}"
                )

    x_ids = x["id"].astype("string").fillna("").to_numpy(dtype=str)
    y_ids = y["id"].astype("string").fillna("").to_numpy(dtype=str)
    mismatches = np.flatnonzero(x_ids != y_ids)
    if mismatches.size:
        position = int(mismatches[0])
        raise ValueError(
            f"{source}: id mismatch at physical row {position}: "
            f"X={x_ids[position]!r}, y={y_ids[position]!r}"
        )

    if "Env" in x.columns and "Env" in y.columns:
        x_env = x["Env"].astype("string").fillna("").to_numpy(dtype=str)
        y_env = y["Env"].astype("string").fillna("").to_numpy(dtype=str)
        env_mismatches = np.flatnonzero(x_env != y_env)
        if env_mismatches.size:
            position = int(env_mismatches[0])
            raise ValueError(
                f"{source}: Env mismatch at physical row {position}: "
                f"X={x_env[position]!r}, y={y_env[position]!r}"
            )


def align_xy_by_physical_row(
    x: pd.DataFrame,
    y: pd.DataFrame,
    *,
    y_columns: Sequence[str],
    source: str = "canonical X/y",
) -> pd.DataFrame:
    """Attach y columns after strict row-wise identity validation.

    Duplicate ids are expected for plot replicates.  Consequently, an id merge
    is never performed here.  The returned frame has a fresh RangeIndex and
    keeps every physical X row exactly once.  Large callers that only need the
    check should use :func:`validate_xy_physical_alignment` to avoid a copy.
    """

    missing = [column for column in y_columns if column not in y.columns]
    if missing:
        raise ValueError(f"{source}: y is missing requested columns {missing}")
    collisions = [column for column in y_columns if column in x.columns]
    if collisions:
        raise ValueError(f"{source}: refusing to overwrite X columns {collisions}")
    validate_xy_physical_alignment(x, y, source=source)

    out = x.reset_index(drop=True).copy()
    y_reset = y.reset_index(drop=True)
    for column in y_columns:
        out[column] = y_reset[column].to_numpy(copy=True)
    return out
