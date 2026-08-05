"""Validated access and leakage-safe transforms for normalized G2F bundles.

The normalized bundle intentionally separates physical phenotype rows from the
two repeated entity blocks.  A bundle has this layout::

    manifest.json
    genotypes.csv
    environments_<window>.csv
    rows_<row_policy>_<split>.csv

``manifest.json`` freezes the ordered marker panel, environment feature sets,
and named variants.  This module contains no Torch dependency, so builders,
audits, kernel baselines, and neural loaders can all share the same validation
and transform contract.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .data_integrity import extract_hybrid_series, split_hybrid_parents


_SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_ROW_COLUMNS = (
    "row_index",
    "source_row_index",
    "id",
    "Env",
    "Hybrid",
)


def _require_unique_strings(values: Sequence[object], *, source: str) -> list[str]:
    result = [str(value).strip() for value in values]
    if any(not value for value in result):
        raise ValueError(f"{source} contains an empty name")
    duplicates = pd.Index(result)[pd.Index(result).duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"{source} contains duplicate names: {duplicates[:5]}")
    return result


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sequence_fingerprint(values: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = str(value).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _normalise_drop_columns(value: object, *, source: str) -> list[str]:
    """Accept a list of names or duplicate->representative declarations."""

    if value is None:
        return []
    if isinstance(value, Mapping):
        values = list(value.keys())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
    else:
        raise ValueError(f"{source} must be a list or mapping")
    return _require_unique_strings(values, source=source)


def _read_csv(path: Path, *, source: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"{source} does not exist: {path}")
    return pd.read_csv(path, low_memory=False)


def _validate_entity_key(frame: pd.DataFrame, key: str, *, source: str) -> None:
    if key not in frame.columns:
        raise ValueError(f"{source} is missing key column {key!r}")
    if frame[key].isna().any():
        raise ValueError(f"{source}.{key} contains missing values")
    stripped = frame[key].astype(str).str.strip()
    if stripped.eq("").any():
        raise ValueError(f"{source}.{key} contains empty values")
    if stripped.duplicated().any():
        examples = stripped.loc[stripped.duplicated(keep=False)].unique()[:5].tolist()
        raise ValueError(f"{source}.{key} is not unique: {examples}")
    frame[key] = stripped


def _validate_rows(rows: pd.DataFrame, *, source: str, require_target: bool) -> None:
    missing = [column for column in _ROW_COLUMNS if column not in rows.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")

    row_index = pd.to_numeric(rows["row_index"], errors="raise").to_numpy(dtype=float)
    expected = np.arange(len(rows), dtype=np.int64)
    if (
        not np.isfinite(row_index).all()
        or not np.equal(row_index, np.floor(row_index)).all()
        or not np.array_equal(row_index.astype(np.int64), expected)
    ):
        raise ValueError(f"{source}.row_index must be contiguous 0..{max(len(rows) - 1, 0)}")
    rows["row_index"] = expected

    source_index = pd.to_numeric(rows["source_row_index"], errors="raise").to_numpy(dtype=float)
    if (
        not np.isfinite(source_index).all()
        or not np.equal(source_index, np.floor(source_index)).all()
        or (source_index < 0).any()
    ):
        raise ValueError(f"{source}.source_row_index must contain non-negative integers")
    rows["source_row_index"] = source_index.astype(np.int64)
    if rows["source_row_index"].duplicated().any():
        examples = (
            rows.loc[
                rows["source_row_index"].duplicated(keep=False), "source_row_index"
            ]
            .unique()[:5]
            .tolist()
        )
        raise ValueError(
            f"{source}.source_row_index must identify unique physical source rows: {examples}"
        )

    for column in ("id", "Env", "Hybrid"):
        if rows[column].isna().any():
            raise ValueError(f"{source}.{column} contains missing values")
        rows[column] = rows[column].astype(str).str.strip()
        if rows[column].eq("").any():
            raise ValueError(f"{source}.{column} contains empty values")

    extracted = extract_hybrid_series(rows["id"], rows["Env"], source=source)
    mismatch = extracted.to_numpy(dtype=str) != rows["Hybrid"].to_numpy(dtype=str)
    if mismatch.any():
        position = int(np.flatnonzero(mismatch)[0])
        raise ValueError(
            f"{source}: Hybrid does not match exact id/Env identity at row {position}: "
            f"id={rows.iloc[position]['id']!r}, Env={rows.iloc[position]['Env']!r}, "
            f"Hybrid={rows.iloc[position]['Hybrid']!r}"
        )
    # Validate parent structure here instead of discovering malformed keys in a
    # downstream model worker.
    split_hybrid_parents(rows["Hybrid"], strict=True)

    if require_target and "Yield_Mg_ha" not in rows:
        raise ValueError(f"{source} is missing required training target Yield_Mg_ha")
    if "Yield_Mg_ha" in rows:
        target = pd.to_numeric(rows["Yield_Mg_ha"], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(target).all():
            raise ValueError(f"{source}.Yield_Mg_ha must be finite when present")
        rows["Yield_Mg_ha"] = target


@dataclass(frozen=True)
class CanonicalVariant:
    """One validated variant/split view of a normalized canonical bundle."""

    root: Path
    name: str
    split: str
    row_policy: str
    window: str
    feature_set: str
    rows: pd.DataFrame
    genotypes: pd.DataFrame
    environments: pd.DataFrame
    marker_columns: tuple[str, ...]
    environment_feature_columns: tuple[str, ...]
    constant_columns: tuple[str, ...]
    exact_duplicate_columns: tuple[str, ...]
    hybrid_feature_set: str | None
    hybrid_covariates: pd.DataFrame | None
    hybrid_feature_columns: tuple[str, ...]
    hybrid_constant_columns: tuple[str, ...]
    hybrid_exact_duplicate_columns: tuple[str, ...]
    manifest_fingerprint: str
    dataset_fingerprint: str

    @property
    def environment_columns(self) -> tuple[str, ...]:
        """Ordered usable environment columns after declared no-information drops."""

        dropped = set(self.constant_columns) | set(self.exact_duplicate_columns)
        return tuple(column for column in self.environment_feature_columns if column not in dropped)

    @property
    def hybrid_columns(self) -> tuple[str, ...]:
        """Ordered usable hybrid covariates after declared no-information drops."""

        dropped = set(self.hybrid_constant_columns) | set(
            self.hybrid_exact_duplicate_columns
        )
        return tuple(
            column
            for column in self.hybrid_feature_columns
            if column not in dropped
        )


class CanonicalDataBundle:
    """Load a normalized bundle manifest and materialize fail-closed views."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Canonical manifest does not exist: {manifest_path}")
        manifest_bytes = manifest_path.read_bytes()
        try:
            manifest = json.loads(manifest_bytes)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid canonical manifest JSON: {manifest_path}: {exc}") from exc
        if not isinstance(manifest, Mapping):
            raise ValueError("Canonical manifest must be a JSON object")
        self.manifest: dict[str, Any] = dict(manifest)
        self.manifest_fingerprint = hashlib.sha256(manifest_bytes).hexdigest()

        marker_value = self.manifest.get("marker_columns")
        if not isinstance(marker_value, list) or not marker_value:
            raise ValueError("manifest.marker_columns must be a non-empty list")
        self.marker_columns = tuple(
            _require_unique_strings(marker_value, source="manifest.marker_columns")
        )

        variants = self.manifest.get("variants")
        if not isinstance(variants, Mapping) or not variants:
            raise ValueError("manifest.variants must be a non-empty object")
        self.variants: dict[str, Mapping[str, Any]] = {}
        for raw_name, raw_spec in variants.items():
            name = str(raw_name).strip()
            if not _SAFE_COMPONENT.fullmatch(name):
                raise ValueError(f"Unsafe or empty variant name: {raw_name!r}")
            if not isinstance(raw_spec, Mapping):
                raise ValueError(f"manifest.variants.{name} must be an object")
            spec = dict(raw_spec)
            for key in ("row_policy", "window", "feature_set"):
                value = str(spec.get(key, "")).strip()
                if not _SAFE_COMPONENT.fullmatch(value):
                    raise ValueError(
                        f"manifest.variants.{name}.{key} must be a safe non-empty name"
                    )
                spec[key] = value
            if spec["row_policy"] not in {"complete", "recovered"}:
                raise ValueError(
                    f"manifest.variants.{name}.row_policy must be complete or recovered"
                )
            if "hybrid_feature_set" in spec:
                hybrid_feature_set = spec["hybrid_feature_set"]
                if hybrid_feature_set is None:
                    raise ValueError(
                        f"manifest.variants.{name}.hybrid_feature_set must be a "
                        "safe non-empty name when present"
                    )
                hybrid_feature_set = str(hybrid_feature_set).strip()
                if not _SAFE_COMPONENT.fullmatch(hybrid_feature_set):
                    raise ValueError(
                        f"manifest.variants.{name}.hybrid_feature_set must be a "
                        "safe non-empty name when present"
                    )
                spec["hybrid_feature_set"] = hybrid_feature_set
            self.variants[name] = spec

        feature_sets = self.manifest.get(
            "environment_feature_sets", self.manifest.get("feature_sets")
        )
        if feature_sets is None:
            environment_spec = self.manifest.get("environment", {})
            if isinstance(environment_spec, Mapping):
                feature_sets = environment_spec.get("feature_sets")
        if not isinstance(feature_sets, Mapping) or not feature_sets:
            raise ValueError("manifest.environment_feature_sets must be a non-empty object")
        self.environment_feature_sets: dict[str, Mapping[str, Any]] = {}
        for raw_name, raw_spec in feature_sets.items():
            name = str(raw_name).strip()
            if not _SAFE_COMPONENT.fullmatch(name):
                raise ValueError(f"Unsafe or empty environment feature-set name: {raw_name!r}")
            if isinstance(raw_spec, list):
                spec: dict[str, Any] = {"columns": raw_spec}
            elif isinstance(raw_spec, Mapping):
                spec = dict(raw_spec)
            else:
                raise ValueError(f"Environment feature set {name!r} must be a list or object")
            # Builder manifests keep numeric and categorical schemas separate;
            # the fold-fitted numeric transform consumes only ``numeric``.
            columns = spec.get("columns", spec.get("numeric"))
            if not isinstance(columns, list) or not columns:
                raise ValueError(f"Environment feature set {name!r} needs non-empty columns")
            spec["columns"] = _require_unique_strings(
                columns, source=f"environment feature set {name}.columns"
            )
            spec["drop_constant"] = _normalise_drop_columns(
                spec.get("drop_constant"), source=f"environment feature set {name}.drop_constant"
            )
            spec["drop_exact_duplicates"] = _normalise_drop_columns(
                spec.get("drop_exact_duplicates"),
                source=f"environment feature set {name}.drop_exact_duplicates",
            )
            unknown_drops = (
                set(spec["drop_constant"]) | set(spec["drop_exact_duplicates"])
            ) - set(spec["columns"])
            if unknown_drops:
                raise ValueError(
                    f"Environment feature set {name!r} drops undeclared columns: "
                    f"{sorted(unknown_drops)}"
                )
            self.environment_feature_sets[name] = spec

        hybrid_feature_sets = self.manifest.get("hybrid_feature_sets")
        self.hybrid_feature_sets: dict[str, Mapping[str, Any]] = {}
        if hybrid_feature_sets is not None:
            if not isinstance(hybrid_feature_sets, Mapping):
                raise ValueError("manifest.hybrid_feature_sets must be an object")
            for raw_name, raw_spec in hybrid_feature_sets.items():
                name = str(raw_name).strip()
                if not _SAFE_COMPONENT.fullmatch(name):
                    raise ValueError(
                        f"Unsafe or empty hybrid feature-set name: {raw_name!r}"
                    )
                if isinstance(raw_spec, list):
                    spec = {"columns": raw_spec}
                elif isinstance(raw_spec, Mapping):
                    spec = dict(raw_spec)
                else:
                    raise ValueError(
                        f"Hybrid feature set {name!r} must be a list or object"
                    )
                columns = spec.get("columns")
                if not isinstance(columns, list) or not columns:
                    raise ValueError(
                        f"Hybrid feature set {name!r} needs non-empty columns"
                    )
                spec["columns"] = _require_unique_strings(
                    columns, source=f"hybrid feature set {name}.columns"
                )
                spec["drop_constant"] = _normalise_drop_columns(
                    spec.get("drop_constant"),
                    source=f"hybrid feature set {name}.drop_constant",
                )
                spec["drop_exact_duplicates"] = _normalise_drop_columns(
                    spec.get("drop_exact_duplicates"),
                    source=f"hybrid feature set {name}.drop_exact_duplicates",
                )
                unknown_drops = (
                    set(spec["drop_constant"])
                    | set(spec["drop_exact_duplicates"])
                ) - set(spec["columns"])
                if unknown_drops:
                    raise ValueError(
                        f"Hybrid feature set {name!r} drops undeclared columns: "
                        f"{sorted(unknown_drops)}"
                    )
                self.hybrid_feature_sets[name] = spec

        unknown_sets = {
            str(spec["feature_set"]) for spec in self.variants.values()
        } - set(self.environment_feature_sets)
        if unknown_sets:
            raise ValueError(f"Variants reference unknown environment feature sets: {sorted(unknown_sets)}")
        unknown_hybrid_sets = {
            str(spec["hybrid_feature_set"])
            for spec in self.variants.values()
            if "hybrid_feature_set" in spec
        } - set(self.hybrid_feature_sets)
        if unknown_hybrid_sets:
            raise ValueError(
                "Variants reference unknown hybrid feature sets: "
                f"{sorted(unknown_hybrid_sets)}"
            )

    @property
    def variant_names(self) -> tuple[str, ...]:
        return tuple(self.variants)

    def _verify_declared_file_hash(self, path: Path) -> str:
        actual = _sha256_file(path)
        files = self.manifest.get("files", {})
        if isinstance(files, Mapping) and path.name in files:
            entry = files[path.name]
            expected = entry.get("sha256") if isinstance(entry, Mapping) else entry
            if expected is not None and str(expected).lower() != actual:
                raise ValueError(
                    f"Canonical file checksum mismatch for {path.name}: "
                    f"expected={expected}, actual={actual}"
                )
        return actual

    def load_variant(self, name: str, split: str) -> CanonicalVariant:
        name = str(name).strip()
        if name not in self.variants:
            raise KeyError(f"Unknown canonical variant {name!r}; choices={list(self.variants)}")
        split = str(split).strip().lower()
        if split not in {"train", "test"}:
            raise ValueError("Canonical split must be 'train' or 'test'")
        spec = self.variants[name]
        row_policy = str(spec["row_policy"])
        window = str(spec["window"])
        feature_set = str(spec["feature_set"])
        hybrid_feature_set = (
            str(spec["hybrid_feature_set"])
            if "hybrid_feature_set" in spec
            else None
        )
        rows_path = self.root / f"rows_{row_policy}_{split}.csv"
        genotypes_path = self.root / "genotypes.csv"
        environments_path = self.root / f"environments_{window}.csv"
        hybrid_covariates_path = (
            self.root / "hybrid_covariates.csv"
            if hybrid_feature_set is not None
            else None
        )

        rows = _read_csv(rows_path, source=f"canonical rows for {name}/{split}")
        genotypes = _read_csv(genotypes_path, source="canonical genotypes")
        environments = _read_csv(
            environments_path, source=f"canonical environments for window {window}"
        )
        hybrid_covariates = (
            _read_csv(
                hybrid_covariates_path,
                source=f"canonical hybrid covariates for {hybrid_feature_set}",
            )
            if hybrid_covariates_path is not None
            else None
        )
        _validate_rows(
            rows,
            source=f"rows_{row_policy}_{split}",
            require_target=(split == "train"),
        )
        _validate_entity_key(genotypes, "Hybrid", source="genotypes.csv")
        _validate_entity_key(environments, "Env", source=environments_path.name)
        if hybrid_covariates is not None:
            _validate_entity_key(
                hybrid_covariates,
                "Hybrid",
                source=hybrid_covariates_path.name,
            )

        missing_markers = [column for column in self.marker_columns if column not in genotypes]
        if missing_markers:
            raise ValueError(f"genotypes.csv is missing manifest marker columns: {missing_markers[:5]}")
        for column in self.marker_columns:
            genotypes[column] = pd.to_numeric(genotypes[column], errors="raise")

        feature_spec = self.environment_feature_sets[feature_set]
        environment_columns = tuple(feature_spec["columns"])
        missing_features = [column for column in environment_columns if column not in environments]
        if missing_features:
            raise ValueError(
                f"{environments_path.name} is missing feature-set columns: {missing_features[:5]}"
            )

        hybrid_feature_columns: tuple[str, ...] = ()
        hybrid_constant_columns: tuple[str, ...] = ()
        hybrid_exact_duplicate_columns: tuple[str, ...] = ()
        if hybrid_feature_set is not None:
            if hybrid_covariates is None or hybrid_covariates_path is None:
                raise RuntimeError("Selected hybrid feature set was not loaded")
            hybrid_spec = self.hybrid_feature_sets[hybrid_feature_set]
            hybrid_feature_columns = tuple(hybrid_spec["columns"])
            missing_hybrid_features = [
                column
                for column in hybrid_feature_columns
                if column not in hybrid_covariates
            ]
            if missing_hybrid_features:
                raise ValueError(
                    f"{hybrid_covariates_path.name} is missing feature-set columns: "
                    f"{missing_hybrid_features[:5]}"
                )
            collisions = sorted(
                set(hybrid_feature_columns)
                & (set(self.marker_columns) | set(environment_columns))
            )
            if collisions:
                raise ValueError(
                    "Hybrid covariate columns collide with marker/environment "
                    f"columns: {collisions[:5]}"
                )
            for column in hybrid_feature_columns:
                hybrid_covariates[column] = pd.to_numeric(
                    hybrid_covariates[column], errors="raise"
                )
            hybrid_numeric = hybrid_covariates.loc[
                :, hybrid_feature_columns
            ].to_numpy(dtype=float, copy=False)
            if np.isinf(hybrid_numeric).any():
                raise ValueError(
                    f"{hybrid_covariates_path.name} feature values must be finite or NaN"
                )
            genotype_hybrids = set(genotypes["Hybrid"])
            covariate_hybrids = set(hybrid_covariates["Hybrid"])
            missing_covariate_hybrids = sorted(
                genotype_hybrids - covariate_hybrids
            )
            extra_covariate_hybrids = sorted(
                covariate_hybrids - genotype_hybrids
            )
            if missing_covariate_hybrids or extra_covariate_hybrids:
                raise ValueError(
                    f"{hybrid_covariates_path.name}.Hybrid must exactly match "
                    "genotypes.csv; "
                    f"missing={missing_covariate_hybrids[:5]}, "
                    f"extra={extra_covariate_hybrids[:5]}"
                )
            hybrid_constant_columns = tuple(hybrid_spec["drop_constant"])
            hybrid_exact_duplicate_columns = tuple(
                hybrid_spec["drop_exact_duplicates"]
            )

        missing_hybrids = sorted(set(rows["Hybrid"]) - set(genotypes["Hybrid"]))
        if missing_hybrids:
            raise ValueError(
                f"rows reference {len(missing_hybrids)} absent genotype entities: "
                f"{missing_hybrids[:5]}"
            )
        missing_envs = sorted(set(rows["Env"]) - set(environments["Env"]))
        if missing_envs:
            raise ValueError(
                f"rows reference {len(missing_envs)} absent environment entities: "
                f"{missing_envs[:5]}"
            )

        selected_paths = [rows_path, genotypes_path, environments_path]
        if hybrid_covariates_path is not None:
            selected_paths.append(hybrid_covariates_path)
        file_hashes = {
            path.name: self._verify_declared_file_hash(path)
            for path in selected_paths
        }
        dataset_fingerprint = _sha256_json(
            {
                "manifest_sha256": self.manifest_fingerprint,
                "variant": name,
                "split": split,
                "spec": dict(spec),
                "files": file_hashes,
            }
        )
        return CanonicalVariant(
            root=self.root,
            name=name,
            split=split,
            row_policy=row_policy,
            window=window,
            feature_set=feature_set,
            rows=rows,
            genotypes=genotypes,
            environments=environments,
            marker_columns=self.marker_columns,
            environment_feature_columns=environment_columns,
            constant_columns=tuple(feature_spec["drop_constant"]),
            exact_duplicate_columns=tuple(feature_spec["drop_exact_duplicates"]),
            hybrid_feature_set=hybrid_feature_set,
            hybrid_covariates=hybrid_covariates,
            hybrid_feature_columns=hybrid_feature_columns,
            hybrid_constant_columns=hybrid_constant_columns,
            hybrid_exact_duplicate_columns=hybrid_exact_duplicate_columns,
            manifest_fingerprint=self.manifest_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
        )


def load_canonical_variant(
    root: str | Path, variant: str, split: str
) -> CanonicalVariant:
    """Convenience wrapper for one validated bundle view."""

    return CanonicalDataBundle(root).load_variant(variant, split)


@dataclass
class GenotypeImputationResult:
    """Order-independent genotype values plus an original-missingness mask."""

    values: pd.DataFrame
    mask: pd.DataFrame
    provenance: dict[str, Any]


def fit_order_independent_genotypes(
    data: CanonicalVariant | pd.DataFrame,
    train_reference_hybrids: Iterable[object],
    marker_columns: Sequence[object] | None = None,
) -> GenotypeImputationResult:
    """Impute a frozen marker panel from original training-reference calls only.

    For each missing target call, the first choice is the median of *originally
    observed* calls among training-reference hybrids sharing either parent.
    The per-marker median over all training-reference hybrids is the fallback.
    Imputed values are never reused as donors, which makes the result invariant
    to CSV row order and target traversal order.
    """

    if isinstance(data, CanonicalVariant):
        genotypes = data.genotypes
        if marker_columns is None:
            marker_columns = data.marker_columns
    else:
        genotypes = data
    if marker_columns is None:
        raise ValueError("marker_columns is required when data is a DataFrame")
    markers = _require_unique_strings(marker_columns, source="fixed marker_columns")

    frame = genotypes.copy()
    _validate_entity_key(frame, "Hybrid", source="genotypes")
    missing = [column for column in markers if column not in frame]
    if missing:
        raise ValueError(f"genotypes is missing fixed marker columns: {missing[:5]}")
    numeric = frame.loc[:, markers].apply(pd.to_numeric, errors="raise")
    original = numeric.to_numpy(dtype=np.float64, copy=True)
    if np.isinf(original).any():
        raise ValueError("Genotype marker values must be finite or NaN")

    raw_reference = [str(value).strip() for value in train_reference_hybrids]
    references = _require_unique_strings(
        raw_reference, source="train_reference_hybrids"
    )
    if not references:
        raise ValueError("train_reference_hybrids must not be empty")
    position_by_hybrid = {hybrid: i for i, hybrid in enumerate(frame["Hybrid"])}
    absent = [hybrid for hybrid in references if hybrid not in position_by_hybrid]
    if absent:
        raise ValueError(f"Training-reference hybrids are absent from genotypes: {absent[:5]}")
    reference_positions = np.asarray(
        [position_by_hybrid[hybrid] for hybrid in references], dtype=np.int64
    )

    # Parse every target and reference before fitting so malformed parent keys
    # fail before any partial output is produced.
    parent1, parent2 = split_hybrid_parents(frame["Hybrid"], strict=True)
    parent_pairs = list(zip(parent1.tolist(), parent2.tolist()))
    parent_to_reference_positions: dict[str, set[int]] = {}
    for position in reference_positions.tolist():
        for parent in set(parent_pairs[position]):
            parent_to_reference_positions.setdefault(parent, set()).add(position)

    reference_values = original[reference_positions, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        marker_fallback = np.nanmedian(reference_values, axis=0)
    no_fallback = np.flatnonzero(~np.isfinite(marker_fallback))
    if no_fallback.size:
        columns = [markers[int(index)] for index in no_fallback[:5]]
        raise ValueError(
            "Training-reference hybrids have no observed fallback for marker(s): "
            f"{columns}"
        )

    imputed = original.copy()
    original_missing = np.isnan(original)
    sibling_count = np.zeros(len(markers), dtype=np.int64)
    fallback_count = np.zeros(len(markers), dtype=np.int64)
    for position, parents in enumerate(parent_pairs):
        missing_indices = np.flatnonzero(original_missing[position])
        if not missing_indices.size:
            continue
        donor_positions = sorted(
            set().union(
                *(parent_to_reference_positions.get(parent, set()) for parent in set(parents))
            )
        )
        if donor_positions:
            donor_values = original[np.asarray(donor_positions), :][:, missing_indices]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                sibling_median = np.nanmedian(donor_values, axis=0)
        else:
            sibling_median = np.full(len(missing_indices), np.nan, dtype=np.float64)
        use_sibling = np.isfinite(sibling_median)
        if use_sibling.any():
            indices = missing_indices[use_sibling]
            imputed[position, indices] = sibling_median[use_sibling]
            sibling_count[indices] += 1
        use_fallback = ~use_sibling
        if use_fallback.any():
            indices = missing_indices[use_fallback]
            imputed[position, indices] = marker_fallback[indices]
            fallback_count[indices] += 1

    if not np.isfinite(imputed).all():
        raise RuntimeError("Internal error: genotype imputation left non-finite values")
    values = pd.DataFrame(imputed, columns=markers)
    values.insert(0, "Hybrid", frame["Hybrid"].to_numpy(copy=True))
    mask = pd.DataFrame(original_missing, columns=markers, dtype=bool)
    mask.insert(0, "Hybrid", frame["Hybrid"].to_numpy(copy=True))
    provenance: dict[str, Any] = {
        "algorithm": "original_training_sibling_median_then_training_marker_median_v1",
        "frozen_marker_selection": True,
        "marker_columns": markers,
        "marker_columns_sha256": _sequence_fingerprint(markers),
        "train_reference_hybrid_count": len(references),
        "train_reference_hybrids_sha256": _sequence_fingerprint(references),
        "target_hybrid_count": len(frame),
        "original_missing_count": int(original_missing.sum()),
        "sibling_imputed_count": int(sibling_count.sum()),
        "fallback_imputed_count": int(fallback_count.sum()),
        "per_marker": {
            marker: {
                "original_missing": int(original_missing[:, index].sum()),
                "sibling_imputed": int(sibling_count[index]),
                "fallback_imputed": int(fallback_count[index]),
                "training_reference_median": float(marker_fallback[index]),
            }
            for index, marker in enumerate(markers)
        },
    }
    return GenotypeImputationResult(values=values, mask=mask, provenance=provenance)


@dataclass
class HybridCovariateTransformResult:
    """Hybrid covariates transformed with one unique-Hybrid training fit."""

    values: pd.DataFrame
    mask: pd.DataFrame
    scaler: StandardScaler
    provenance: dict[str, Any]


def fit_hybrid_covariate_transform(
    data: CanonicalVariant | pd.DataFrame,
    train_reference_hybrids: Iterable[object],
    feature_columns: Sequence[object] | None = None,
    *,
    constant_columns: Sequence[object] = (),
    exact_duplicate_columns: Sequence[object] = (),
) -> HybridCovariateTransformResult:
    """Fit median fill and scaling on unique training-reference hybrids.

    Public parent-derived features are stored at the canonical ``Hybrid`` entity
    level so tester-role and transfer masks can accompany each value.  The
    transform uses only hybrids present in the fitting role to estimate medians
    and scaling, then applies that frozen transform to the complete entity table.
    Raw missingness remains available in the returned mask.
    """

    if isinstance(data, CanonicalVariant):
        hybrid_covariates = data.hybrid_covariates
        if hybrid_covariates is None or data.hybrid_feature_set is None:
            raise ValueError(
                "CanonicalVariant does not select a hybrid feature set"
            )
        if feature_columns is not None:
            raise ValueError(
                "feature_columns must be omitted for CanonicalVariant input"
            )
        feature_columns = data.hybrid_feature_columns
        constant_columns = data.hybrid_constant_columns
        exact_duplicate_columns = data.hybrid_exact_duplicate_columns
    else:
        hybrid_covariates = data
    if feature_columns is None:
        raise ValueError("feature_columns is required when data is a DataFrame")

    features = _require_unique_strings(
        feature_columns, source="hybrid feature_columns"
    )
    constant = _require_unique_strings(
        list(constant_columns), source="hybrid constant_columns"
    )
    exact = _require_unique_strings(
        list(exact_duplicate_columns),
        source="hybrid exact_duplicate_columns",
    )
    unknown_drops = (set(constant) | set(exact)) - set(features)
    if unknown_drops:
        raise ValueError(
            "Declared hybrid drops are not selected features: "
            f"{sorted(unknown_drops)}"
        )
    overlap = set(constant) & set(exact)
    if overlap:
        raise ValueError(
            "Hybrid columns declared as both constant and exact duplicate: "
            f"{sorted(overlap)}"
        )
    drops = set(constant) | set(exact)
    selected = [column for column in features if column not in drops]
    if not selected:
        raise ValueError("No hybrid covariates remain after declared drops")

    frame = hybrid_covariates.copy()
    _validate_entity_key(frame, "Hybrid", source="hybrid_covariates")
    missing = [column for column in features if column not in frame]
    if missing:
        raise ValueError(
            f"hybrid_covariates is missing selected feature columns: {missing[:5]}"
        )
    numeric = frame.loc[:, selected].apply(pd.to_numeric, errors="raise")
    original = numeric.to_numpy(dtype=np.float64, copy=True)
    if np.isinf(original).any():
        raise ValueError("Hybrid covariate values must be finite or NaN")

    raw_reference = [str(value).strip() for value in train_reference_hybrids]
    references = list(dict.fromkeys(raw_reference))
    if not references or any(not value for value in references):
        raise ValueError(
            "train_reference_hybrids must contain non-empty hybrid names"
        )
    position_by_hybrid = {
        hybrid: index for index, hybrid in enumerate(frame["Hybrid"])
    }
    absent = [
        hybrid for hybrid in references if hybrid not in position_by_hybrid
    ]
    if absent:
        raise ValueError(
            "Training-reference hybrids are absent from hybrid_covariates: "
            f"{absent[:5]}"
        )
    reference_positions = np.asarray(
        [position_by_hybrid[hybrid] for hybrid in references], dtype=np.int64
    )
    reference_values = original[reference_positions, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(reference_values, axis=0)
    no_median = np.flatnonzero(~np.isfinite(medians))
    if no_median.size:
        columns = [selected[int(index)] for index in no_median[:5]]
        raise ValueError(
            "Training-reference hybrids have no observed median for feature(s): "
            f"{columns}"
        )

    original_missing = np.isnan(original)
    filled = np.where(original_missing, medians[None, :], original)
    scaler = StandardScaler()
    scaler.fit(filled[reference_positions, :])
    transformed = scaler.transform(filled)
    if not np.isfinite(transformed).all():
        raise RuntimeError(
            "Internal error: hybrid covariate transform produced non-finite values"
        )

    values = pd.DataFrame(transformed, columns=selected)
    values.insert(0, "Hybrid", frame["Hybrid"].to_numpy(copy=True))
    mask = pd.DataFrame(original_missing, columns=selected, dtype=bool)
    mask.insert(0, "Hybrid", frame["Hybrid"].to_numpy(copy=True))
    reference_observed = np.count_nonzero(
        ~original_missing[reference_positions, :], axis=0
    )
    target_observed = np.count_nonzero(~original_missing, axis=0)
    provenance: dict[str, Any] = {
        "algorithm": "train_unique_hybrid_median_standard_scaler_v1",
        "feature_columns_before_drop": features,
        "feature_columns": selected,
        "feature_columns_sha256": _sequence_fingerprint(selected),
        "declared_constant_drops": constant,
        "declared_exact_duplicate_drops": exact,
        "train_reference_hybrid_count": len(references),
        "train_reference_hybrids_sha256": _sequence_fingerprint(references),
        "target_hybrid_count": len(frame),
        "median_fill_count": int(original_missing.sum()),
        "reference_observed_count": {
            column: int(reference_observed[index])
            for index, column in enumerate(selected)
        },
        "target_observed_count": {
            column: int(target_observed[index])
            for index, column in enumerate(selected)
        },
        "medians": {
            column: float(medians[index])
            for index, column in enumerate(selected)
        },
        "scaler_mean": {
            column: float(scaler.mean_[index])
            for index, column in enumerate(selected)
        },
        "scaler_scale": {
            column: float(scaler.scale_[index])
            for index, column in enumerate(selected)
        },
        "scaler_var": {
            column: float(scaler.var_[index])
            for index, column in enumerate(selected)
        },
        "scaler_n_samples_seen": int(scaler.n_samples_seen_),
    }
    return HybridCovariateTransformResult(
        values=values,
        mask=mask,
        scaler=scaler,
        provenance=provenance,
    )


@dataclass
class EnvironmentTransformResult:
    """Environment entity matrix transformed with one unique-Env train fit."""

    values: pd.DataFrame
    mask: pd.DataFrame
    scaler: StandardScaler
    provenance: dict[str, Any]


def fit_environment_transform(
    data: CanonicalVariant | pd.DataFrame,
    train_reference_envs: Iterable[object],
    feature_columns: Sequence[object] | None = None,
    *,
    constant_columns: Sequence[object] = (),
    exact_duplicate_columns: Sequence[object] = (),
    unobserved_reference_policy: str = "error",
) -> EnvironmentTransformResult:
    """Fit median fill and scaling on unique training environments.

    Passing a :class:`CanonicalVariant` selects the ordered feature set and its
    declared constant/exact-duplicate drops directly from the manifest.  The
    fitted transform is then applied once to the complete environment entity
    table, avoiding plot-replicate weighting in both imputation and scaling.
    """

    if isinstance(data, CanonicalVariant):
        environments = data.environments
        if feature_columns is not None:
            raise ValueError("feature_columns must be omitted for CanonicalVariant input")
        feature_columns = data.environment_feature_columns
        constant_columns = data.constant_columns
        exact_duplicate_columns = data.exact_duplicate_columns
    else:
        environments = data
    if feature_columns is None:
        raise ValueError("feature_columns is required when data is a DataFrame")

    policy = str(unobserved_reference_policy).strip().lower()
    if policy not in {"error", "neutralize"}:
        raise ValueError(
            "unobserved_reference_policy must be one of ['error', 'neutralize']"
        )

    features = _require_unique_strings(feature_columns, source="environment feature_columns")
    constant = _require_unique_strings(
        list(constant_columns), source="constant_columns"
    )
    exact = _require_unique_strings(
        list(exact_duplicate_columns), source="exact_duplicate_columns"
    )
    unknown_drops = (set(constant) | set(exact)) - set(features)
    if unknown_drops:
        raise ValueError(f"Declared environment drops are not selected features: {sorted(unknown_drops)}")
    overlap = set(constant) & set(exact)
    if overlap:
        raise ValueError(f"Environment columns declared as both constant and exact duplicate: {sorted(overlap)}")
    drops = set(constant) | set(exact)
    selected = [column for column in features if column not in drops]
    if not selected:
        raise ValueError("No environment features remain after declared drops")

    frame = environments.copy()
    _validate_entity_key(frame, "Env", source="environments")
    missing = [column for column in features if column not in frame]
    if missing:
        raise ValueError(f"environments is missing selected feature columns: {missing[:5]}")
    numeric = frame.loc[:, selected].apply(pd.to_numeric, errors="raise")
    original = numeric.to_numpy(dtype=np.float64, copy=True)
    if np.isinf(original).any():
        raise ValueError("Environment feature values must be finite or NaN")

    raw_reference = [str(value).strip() for value in train_reference_envs]
    # Duplicate plot-level Env inputs are intentionally collapsed before fit.
    references = list(dict.fromkeys(raw_reference))
    if not references or any(not value for value in references):
        raise ValueError("train_reference_envs must contain non-empty environment names")
    position_by_env = {env: i for i, env in enumerate(frame["Env"])}
    absent = [env for env in references if env not in position_by_env]
    if absent:
        raise ValueError(f"Training-reference environments are absent: {absent[:5]}")
    reference_positions = np.asarray(
        [position_by_env[env] for env in references], dtype=np.int64
    )
    reference_values = original[reference_positions, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        medians = np.nanmedian(reference_values, axis=0)
    no_median = np.flatnonzero(~np.isfinite(medians))
    if no_median.size and policy == "error":
        columns = [selected[int(index)] for index in no_median[:5]]
        raise ValueError(
            "Training-reference environments have no observed median for feature(s): "
            f"{columns}"
        )

    original_missing = np.isnan(original)
    neutralized_columns = [selected[int(index)] for index in no_median]
    if no_median.size:
        # A feature absent from the fitting era is not learnable.  Keep the
        # manifest-defined input shape stable, but force that feature to the
        # standardized neutral value for every environment, including the
        # pseudo-future.  This prevents later X values from entering through
        # random/untrained weights and uses no target labels.
        medians = medians.copy()
        medians[no_median] = 0.0
    filled = np.where(original_missing, medians[None, :], original)
    transform_mask = original_missing.copy()
    if no_median.size:
        filled[:, no_median] = 0.0
        transform_mask[:, no_median] = True
    scaler = StandardScaler()
    scaler.fit(filled[reference_positions, :])
    transformed = scaler.transform(filled)
    if not np.isfinite(transformed).all():
        raise RuntimeError("Internal error: environment transform produced non-finite values")

    values = pd.DataFrame(transformed, columns=selected)
    values.insert(0, "Env", frame["Env"].to_numpy(copy=True))
    mask = pd.DataFrame(transform_mask, columns=selected, dtype=bool)
    mask.insert(0, "Env", frame["Env"].to_numpy(copy=True))
    provenance: dict[str, Any] = {
        "algorithm": (
            "train_unique_env_median_standard_scaler_neutralize_unobserved_v1"
            if neutralized_columns
            else "train_unique_env_median_standard_scaler_v1"
        ),
        "feature_columns_before_drop": features,
        "feature_columns": selected,
        "feature_columns_sha256": _sequence_fingerprint(selected),
        "declared_constant_drops": constant,
        "declared_exact_duplicate_drops": exact,
        "unobserved_reference_policy": policy,
        "neutralized_unobserved_columns": neutralized_columns,
        "train_reference_env_count": len(references),
        "train_reference_envs_sha256": _sequence_fingerprint(references),
        "target_env_count": len(frame),
        "raw_missing_count": int(original_missing.sum()),
        "median_fill_count": int(transform_mask.sum()),
        "medians": {column: float(medians[i]) for i, column in enumerate(selected)},
        "scaler_mean": {
            column: float(scaler.mean_[i]) for i, column in enumerate(selected)
        },
        "scaler_scale": {
            column: float(scaler.scale_[i]) for i, column in enumerate(selected)
        },
        "scaler_var": {
            column: float(scaler.var_[i]) for i, column in enumerate(selected)
        },
        "scaler_n_samples_seen": int(scaler.n_samples_seen_),
    }
    return EnvironmentTransformResult(
        values=values,
        mask=mask,
        scaler=scaler,
        provenance=provenance,
    )


__all__ = [
    "CanonicalDataBundle",
    "CanonicalVariant",
    "EnvironmentTransformResult",
    "GenotypeImputationResult",
    "HybridCovariateTransformResult",
    "fit_environment_transform",
    "fit_hybrid_covariate_transform",
    "fit_order_independent_genotypes",
    "load_canonical_variant",
]
