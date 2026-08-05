#!/usr/bin/env python3
"""Build the normalized, provenance-frozen GP-Transformer canonical v4 bundle.

The builder deliberately stops before any learned, split-dependent transform.
It records raw marker missingness, environment block missingness, and exact
physical rows.  Genotype/environment imputation and scaling are fit later from
the training rows of each validation fold by :mod:`utils.canonical_data`.

The output is entity-normalized: genotype and environment values are stored
once, while row tables retain every physical yield observation.  Dataset
ablations therefore share facts instead of duplicating multi-gigabyte flat CSVs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_TRAIN = Path("data/Training_data")
RAW_TEST = Path("data/Testing_data")
DEFAULT_OUTPUT = Path("data/maize_data_2014-2023_vs_2024_v4")
TARGET = "Yield_Mg_ha"

WEATHER_COLUMNS = [
    "RH2M",
    "T2M_MAX",
    "ALLSKY_SFC_SW_DWN",
    "T2MWET",
    "QV2M",
    "T2M_MIN",
    "T2MDEW",
    "PS",
    "T2M",
    "WS2M",
    "PRECTOTCORR",
]
WEATHER_STATS = ("min", "max", "mean", "acum")
CATEGORICAL_COLUMNS = ["Irrigated", "Treatment", "Previous_Crop"]

# These are the only information-free removals authorized by the audit.  The
# later member is removed in raw source-header order; high-correlation pruning
# remains an explicit future ablation.
CONSTANT_ENV_COLUMNS = ["PRECTOTCORR_min"]
EXACT_DUPLICATE_ENV_COLUMNS = {
    "CumHI30_pGerEme": "HI30_pGerEme",
    "yield_pMatHar": "yield_pEnGMat",
    "PotRunoff_pGerEme": "PotInf_pGerEme",
    "PotRunoff_pEmeEnJ": "PotInf_pEmeEnJ",
    "PotRunoff_pEnJFlo": "PotInf_pEnJFlo",
    "PotRunoff_pFloFla": "PotInf_pFloFla",
    "PotRunoff_pFlaFlw": "PotInf_pFlaFlw",
    "PotRunoff_pFlwStG": "PotInf_pFlwStG",
    "PotRunoff_pStGEnG": "PotInf_pStGEnG",
    "PotRunoff_pEnGMat": "PotInf_pEnGMat",
    "PotRunoff_pMatHar": "PotInf_pMatHar",
}

STATION_LAT = "Weather_Station_Latitude (in decimal numbers NOT DMS)"
STATION_LON = "Weather_Station_Longitude (in decimal numbers NOT DMS)"
TXH4_2019_COORDINATE_OVERRIDE = (33.73715555555555, -101.73278888888888)

SOIL_FEATURES = {
    "1:1 Soil pH": "soil_ph",
    "WDRF Buffer pH": "soil_buffer_ph",
    "Organic Matter LOI %": "soil_organic_matter_pct",
    "CEC/Sum of Cations me/100g": "soil_cec",
    "% Sand": "soil_sand_pct",
    "% Silt": "soil_silt_pct",
    "% Clay": "soil_clay_pct",
}

EXPECTED_COUNTS = {
    "finite_train_rows": 164_921,
    "genotyped_train_rows": 163_026,
    "complete_train_rows": 143_050,
    "observed_test_rows": 9_486,
    "submission_rows": 10_057,
    "retained_markers": 2_224,
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        _json_safe(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _sha256(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance() -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": None if status is None else bool(status),
        "status_sha256": None
        if status is None
        else hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _site_key(env: str) -> str:
    match = re.fullmatch(r"(.+)_([0-9]{4})", str(env).strip())
    if match is None:
        raise ValueError(f"Environment does not end in _YYYY: {env!r}")
    return match.group(1)


def _env_year(env: str) -> int:
    match = re.search(r"([0-9]{4})$", str(env).strip())
    if match is None:
        raise ValueError(f"Environment does not end in a four-digit year: {env!r}")
    return int(match.group(1))


def _validate_hybrids(values: pd.Series, source: str) -> pd.Series:
    result = values.astype(str).str.strip()
    valid = result.str.count("/").eq(1) & ~result.str.startswith("/") & ~result.str.endswith("/")
    if not bool(valid.all()):
        sample = result.loc[~valid].head(10).tolist()
        raise ValueError(f"{source} has malformed Parent1/Parent2 hybrid names: {sample}")
    return result


def _read_genotypes(path: Path) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    frame = pd.read_csv(path, sep="\t", skiprows=1, na_values=["NA"], low_memory=False)
    if frame.empty or len(frame.columns) < 2:
        raise ValueError(f"Malformed numerical genotype table: {path}")
    frame = frame.rename(columns={frame.columns[0]: "Hybrid"})
    frame["Hybrid"] = _validate_hybrids(frame["Hybrid"], str(path))
    if frame["Hybrid"].duplicated().any():
        raise ValueError(f"{path} contains duplicate Hybrid rows")
    marker_columns = list(frame.columns[1:])
    if any(re.fullmatch(r"S[0-9]+_[0-9]+", str(c)) is None for c in marker_columns):
        raise ValueError("Genotype marker columns must all use S<chromosome>_<position>")
    numeric = frame[marker_columns].apply(pd.to_numeric, errors="coerce")
    observed = numeric.to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(observed)
    if finite.any() and not bool(np.isin(observed[finite], [0.0, 0.5, 1.0]).all()):
        bad = np.unique(observed[finite][~np.isin(observed[finite], [0.0, 0.5, 1.0])])[:10]
        raise ValueError(f"Unexpected raw genotype values: {bad.tolist()}")

    # Preserve the exact audited panel.  This rule intentionally matches the
    # released all-hybrid genotype table and is frozen before target fitting.
    missing_fraction = numeric.isna().mean(axis=0)
    retained = missing_fraction[missing_fraction <= 0.10].index.tolist()
    if len(retained) != EXPECTED_COUNTS["retained_markers"]:
        raise ValueError(
            f"Expected {EXPECTED_COUNTS['retained_markers']} markers at <=10% missing; "
            f"found {len(retained)}"
        )
    output = pd.concat([frame[["Hybrid"]], numeric[retained].astype(np.float32)], axis=1)
    provenance = {
        "raw_hybrids": len(frame),
        "raw_markers": len(marker_columns),
        "retained_markers": len(retained),
        "rule": "missing_fraction_le_0.10_on_released_all_hybrid_panel",
        "observed_encoding": {"0": "minor_homozygote", "0.5": "heterozygote", "1": "major_homozygote"},
        "raw_missing_calls_retained": int(output[retained].isna().sum().sum()),
        "imputation": "not_applied_by_builder; fit on split training references",
    }
    return output, retained, provenance


def _parse_dates(values: pd.Series) -> pd.Series:
    # The release mixes two- and four-digit slash dates.  pandas handles both,
    # and errors are retained as NaT for explicit window fallbacks.
    return pd.to_datetime(values, errors="coerce", format="mixed")


def _weather_windows(
    train_traits: pd.DataFrame,
    train_weather_season: pd.DataFrame,
    test_meta: pd.DataFrame,
    test_weather_season: pd.DataFrame,
) -> pd.DataFrame:
    planted = train_traits.assign(_date=_parse_dates(train_traits["Date_Planted"]))
    harvested = train_traits.assign(_date=_parse_dates(train_traits["Date_Harvested"]))
    train_dates = pd.DataFrame(
        {
            "planting_date": planted.groupby("Env")["_date"].min(),
            "harvest_date": harvested.groupby("Env")["_date"].max(),
        }
    )
    train_season = train_weather_season.copy()
    train_season["_date"] = pd.to_datetime(
        train_season["Date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    )
    season_bounds = train_season.groupby("Env")["_date"].agg(["min", "max"])
    train_dates = train_dates.join(season_bounds, how="left")
    train_dates["window_start_preplant14"] = train_dates["planting_date"] - pd.Timedelta(days=14)
    train_dates["window_start_current"] = train_dates["planting_date"]
    train_dates["window_end"] = (train_dates["harvest_date"] + pd.Timedelta(days=14)).fillna(
        train_dates["max"]
    )
    has_weather = train_dates["min"].notna() & train_dates["max"].notna()
    start_matches = train_dates.loc[has_weather, "min"].eq(
        train_dates.loc[has_weather, "window_start_preplant14"]
    )
    end_matches = train_dates.loc[has_weather, "max"].eq(
        train_dates.loc[has_weather, "window_end"]
    )
    if not bool(start_matches.all()) or not bool(end_matches.all()):
        raise ValueError("Training seasons-only boundaries do not match explicit planting/harvest windows")

    test_dates = test_meta.set_index("Env")[["Date_Planted"]].copy()
    test_dates["planting_date"] = _parse_dates(test_dates.pop("Date_Planted"))
    test_season = test_weather_season.copy()
    test_season["_date"] = pd.to_datetime(
        test_season["Date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    )
    test_bounds = test_season.groupby("Env")["_date"].agg(["min", "max"])
    test_dates = test_dates.join(test_bounds, how="left")
    test_dates["window_start_preplant14"] = test_dates["planting_date"] - pd.Timedelta(days=14)
    test_dates["window_start_current"] = test_dates["planting_date"]
    test_dates["window_end"] = test_dates["max"]
    if not bool(test_dates["min"].eq(test_dates["window_start_preplant14"]).all()):
        raise ValueError("Testing seasons-only boundaries do not begin 14 days before planting")

    columns = [
        "planting_date",
        "window_start_preplant14",
        "window_start_current",
        "window_end",
    ]
    result = pd.concat([train_dates[columns], test_dates[columns]], axis=0)
    if result.index.duplicated().any():
        raise ValueError("Training/testing environment names overlap unexpectedly")
    return result


def _aggregate_weather(
    weather: pd.DataFrame,
    windows: pd.DataFrame,
    start_column: str,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    _require_columns(weather, ["Env", "Date", *WEATHER_COLUMNS], Path(label))
    work = weather.copy()
    work["_date"] = pd.to_datetime(
        work["Date"].astype("Int64").astype(str), format="%Y%m%d", errors="coerce"
    )
    for column in WEATHER_COLUMNS:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.join(windows[[start_column, "window_end"]], on="Env", how="left", validate="many_to_one")
    selected = work[
        work["_date"].ge(work[start_column]) & work["_date"].le(work["window_end"])
    ].copy()
    if selected.empty:
        raise ValueError(f"{label}: explicit weather window selected no rows")

    grouped = selected.groupby("Env", sort=False)
    pieces: dict[str, pd.Series] = {}
    for stat in WEATHER_STATS:
        for column in WEATHER_COLUMNS:
            name = f"{column}_{stat}"
            if stat == "min":
                pieces[name] = grouped[column].min()
            elif stat == "max":
                pieces[name] = grouped[column].max()
            elif stat == "mean":
                pieces[name] = grouped[column].mean()
            else:
                pieces[name] = grouped[column].sum(min_count=1)
    result = pd.DataFrame(pieces)
    result["start_month"] = grouped["_date"].min().dt.month.astype(float)
    result["num_days"] = grouped.size().astype(float)
    for column in WEATHER_COLUMNS:
        result[f"meta__weather_missing_days__{column}"] = grouped[column].apply(
            lambda values: int(values.isna().sum())
        )
    result["weather_block_present"] = 1.0
    result["weather_any_missing_days"] = (
        result[[f"meta__weather_missing_days__{c}" for c in WEATHER_COLUMNS]].sum(axis=1) > 0
    ).astype(float)

    # Directly guard the defect found in v2: each accumulation must equal the
    # corresponding named source column's skip-missing sum.
    for column in WEATHER_COLUMNS:
        expected = grouped[column].sum(min_count=1).sort_index()
        observed = result[f"{column}_acum"].sort_index()
        if not np.allclose(expected, observed, rtol=0, atol=1e-10, equal_nan=True):
            raise AssertionError(f"{label}: cross-column accumulation detected for {column}")

    provenance = {
        "window": label,
        "start_column": start_column,
        "environments": int(result.index.nunique()),
        "rows": len(selected),
        "date_min": str(selected["_date"].min().date()),
        "date_max": str(selected["_date"].max().date()),
        "missing_source_cells": int(selected[WEATHER_COLUMNS].isna().sum().sum()),
        "aggregation": "named-column min/max/mean/sum(min_count=1)",
    }
    return result, provenance


def _map_irrigation(value: Any) -> str:
    if pd.isna(value):
        return "UNK"
    text = str(value).strip().lower()
    if text in {"yes", "y", "irrigated"}:
        return "yes"
    if text in {"no", "n", "non-irrigated", "non irrigated", "dryland", "dry land"}:
        return "no"
    return "UNK"


def _map_treatment(value: Any) -> str:
    if pd.isna(value):
        return "UNK"
    text = re.sub(r"\s+", " ", str(value).strip()).lower()
    if text in {"standard", "irrigated", "early planting"}:
        return "Standard"
    if text in {"drought", "dry land", "dryland", "dryland optimal"}:
        return "Dry"
    if text.startswith("late"):
        return "Late"
    if text == "disease trial":
        return "Disease trial"
    return "UNK"


def _map_crop(value: Any) -> str:
    if pd.isna(value):
        return "UNK"
    text = re.sub(r"\s+", " ", str(value).strip()).lower()
    positive = {"soybean", "soybeans", "peanut", "peanuts", "beans", "clover"}
    moderate = {"corn", "maize", "wheat"}
    negative = {"cotton", "fallow", "cereal rye", "sugar beet", "sugarbeet"}
    if text in positive:
        return "positive_impact"
    if text in moderate:
        return "moderate_positive_impact"
    if text in negative:
        return "negative_impact"
    return "UNK"


def _build_metadata(train_path: Path, test_path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = pd.read_csv(train_path, low_memory=False)
    test = pd.read_csv(test_path, low_memory=False)
    for frame, source in ((train, train_path), (test, test_path)):
        _require_columns(frame, ["Env", STATION_LAT, STATION_LON, *CATEGORICAL_COLUMNS], source)
        if frame["Env"].duplicated().any():
            raise ValueError(f"{source} contains duplicate Env rows")
        frame["split"] = "train" if source == train_path else "test"
    all_meta = pd.concat([train, test], ignore_index=True, sort=False)
    all_meta["site_key"] = all_meta["Env"].map(_site_key)
    all_meta["Year"] = all_meta["Env"].map(_env_year)
    all_meta["station_latitude_raw"] = pd.to_numeric(all_meta[STATION_LAT], errors="coerce")
    all_meta["station_longitude_raw"] = pd.to_numeric(all_meta[STATION_LON], errors="coerce")
    lat_cols = [c for c in all_meta.columns if c.startswith("Latitude_of_Field_Corner_")]
    lon_cols = [c for c in all_meta.columns if c.startswith("Longitude_of_Field_Corner_")]
    all_meta["field_latitude"] = all_meta[lat_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    all_meta["field_longitude"] = all_meta[lon_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    train_only = all_meta[all_meta["split"].eq("train")]
    site_station = train_only.groupby("site_key")[["station_latitude_raw", "station_longitude_raw"]].mean()
    all_meta["station_present"] = all_meta[["station_latitude_raw", "station_longitude_raw"]].notna().all(axis=1).astype(float)
    all_meta["field_present"] = all_meta[["field_latitude", "field_longitude"]].notna().all(axis=1).astype(float)
    all_meta["latitude"] = all_meta["station_latitude_raw"]
    all_meta["longitude"] = all_meta["station_longitude_raw"]
    missing_station = all_meta[["latitude", "longitude"]].isna().any(axis=1)
    all_meta.loc[missing_station, "latitude"] = all_meta.loc[missing_station, "site_key"].map(
        site_station["station_latitude_raw"]
    )
    all_meta.loc[missing_station, "longitude"] = all_meta.loc[missing_station, "site_key"].map(
        site_station["station_longitude_raw"]
    )
    txh4 = all_meta["Env"].eq("TXH4_2019") & all_meta[["latitude", "longitude"]].isna().any(axis=1)
    all_meta.loc[txh4, ["latitude", "longitude"]] = TXH4_2019_COORDINATE_OVERRIDE
    still_missing = all_meta[["latitude", "longitude"]].isna().any(axis=1)
    if still_missing.any():
        missing = all_meta.loc[still_missing, "Env"].tolist()
        raise ValueError(f"No historical station-coordinate reference for environments: {missing}")
    all_meta["station_coordinate_imputed"] = (1.0 - all_meta["station_present"]).astype(float)

    # Freeze the three documented manual irrigation repairs rather than hiding
    # them in a notebook cell.
    manual_irrigation_yes = {"NEH1_2015", "TXH1_2021", "TXH3_2021"}
    all_meta["Irrigated"] = all_meta["Irrigated"].map(_map_irrigation)
    all_meta.loc[all_meta["Env"].isin(manual_irrigation_yes), "Irrigated"] = "yes"
    all_meta["Treatment"] = all_meta["Treatment"].map(_map_treatment)
    all_meta["Previous_Crop"] = all_meta["Previous_Crop"].map(_map_crop)

    keep = [
        "Env", "Year", "site_key", "latitude", "longitude", *CATEGORICAL_COLUMNS,
        "station_latitude_raw", "station_longitude_raw", "field_latitude", "field_longitude",
        "station_present", "field_present", "station_coordinate_imputed",
    ]
    result = all_meta[keep].set_index("Env")
    provenance = {
        "station_coordinate_rule": "own station, else historical training same-site mean",
        "coordinate_override": {"TXH4_2019": list(TXH4_2019_COORDINATE_OVERRIDE)},
        "manual_irrigation_yes": sorted(manual_irrigation_yes),
        "category_mapping": "exact normalized mappings; unrecognized and missing values become UNK",
    }
    return result, provenance


def _build_soil(train_path: Path, test_path: Path, environments: pd.Index) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames = []
    for path in (train_path, test_path):
        frame = pd.read_csv(path, low_memory=False)
        _require_columns(frame, ["Env", *SOIL_FEATURES], path)
        frame["Year"] = frame["Env"].map(_env_year)
        frame["site_key"] = frame["Env"].map(_site_key)
        for raw in SOIL_FEATURES:
            frame[raw] = pd.to_numeric(frame[raw], errors="coerce")
        frames.append(frame[["Env", "Year", "site_key", *SOIL_FEATURES]])
    raw = pd.concat(frames, ignore_index=True)
    exact = raw.groupby("Env")[[*SOIL_FEATURES]].median()
    records = raw.drop_duplicates("Env").set_index("Env")[["Year", "site_key"]].join(exact)
    output = pd.DataFrame(index=environments)
    for raw_name, output_name in SOIL_FEATURES.items():
        output[output_name] = np.nan
    output["soil_present_current"] = 0.0
    output["soil_carried_prior"] = 0.0
    output["soil_age_years"] = np.nan

    record_by_site = {site: group.sort_values("Year") for site, group in records.groupby("site_key")}
    for env in output.index:
        year = _env_year(env)
        site = _site_key(env)
        if env in records.index:
            source = records.loc[env]
            output.loc[env, "soil_present_current"] = 1.0
            output.loc[env, "soil_age_years"] = 0.0
        else:
            candidates = record_by_site.get(site)
            if candidates is None:
                continue
            candidates = candidates[candidates["Year"] < year]
            if candidates.empty:
                continue
            source = candidates.iloc[-1]
            output.loc[env, "soil_carried_prior"] = 1.0
            output.loc[env, "soil_age_years"] = float(year - int(source["Year"]))
        for raw_name, output_name in SOIL_FEATURES.items():
            output.loc[env, output_name] = source[raw_name]
    output["soil_block_present"] = output[list(SOIL_FEATURES.values())].notna().any(axis=1).astype(float)
    provenance = {
        "stable_features": SOIL_FEATURES,
        "within_environment_aggregation": "median",
        "missing_environment_rule": "most recent strictly prior same-site record",
        "dynamic_nutrients_excluded": True,
    }
    return output, provenance


def _row_table(frame: pd.DataFrame, policy: str, source: str) -> pd.DataFrame:
    work = frame.copy()
    work["Hybrid"] = _validate_hybrids(work["Hybrid"], source)
    work["Env"] = work["Env"].astype(str).str.strip()
    work[TARGET] = pd.to_numeric(work[TARGET], errors="coerce")
    if "source_row_index" not in work:
        work["source_row_index"] = np.arange(len(work), dtype=np.int64)
    work["id"] = work["Env"] + "-" + work["Hybrid"]
    counts = work.groupby(["Env", "Hybrid"], sort=False)[TARGET].transform("size")
    variances = work.groupby(["Env", "Hybrid"], sort=False)[TARGET].transform(lambda x: x.var(ddof=1))
    work["cell_replicate_count"] = counts.astype(np.int32)
    work["cell_yield_variance"] = variances.fillna(0.0).astype(float)
    work.insert(0, "row_index", np.arange(len(work), dtype=np.int64))
    work["row_policy"] = policy
    columns = [
        "row_index", "source_row_index", "id", "Env", "Hybrid", TARGET,
        "cell_replicate_count", "cell_yield_variance", "row_policy",
    ]
    result = work[columns]
    expected_ids = result["Env"] + "-" + result["Hybrid"]
    if not result["id"].equals(expected_ids):
        raise AssertionError(f"{source}: exact <Env>-<Hybrid> identity construction failed")
    return result


def _write_csv(frame: pd.DataFrame, path: Path) -> dict[str, Any]:
    frame.to_csv(path, index=False, lineterminator="\n")
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": _sha256(path), "rows": len(frame)}


def build(root: Path, output: Path, strict_counts: bool = True) -> dict[str, Any]:
    root = root.resolve()
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")

    paths = {
        "train_traits": root / RAW_TRAIN / "1_Training_Trait_Data_2014_2023.csv",
        "train_meta": root / RAW_TRAIN / "2_Training_Meta_Data_2014_2023.csv",
        "train_soil": root / RAW_TRAIN / "3_Training_Soil_Data_2015_2023.csv",
        "train_weather_full": root / RAW_TRAIN / "4_Training_Weather_Data_2014_2023_full_year.csv",
        "train_weather_season": root / RAW_TRAIN / "4_Training_Weather_Data_2014_2023_seasons_only.csv",
        "genotypes": root / RAW_TRAIN / "5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt",
        "train_ec": root / RAW_TRAIN / "6_Training_EC_Data_2014_2023.csv",
        "submission": root / RAW_TEST / "1_Submission_Template_2024.csv",
        "test_meta": root / RAW_TEST / "2_Testing_Meta_Data_2024.csv",
        "test_soil": root / RAW_TEST / "3_Testing_Soil_Data_2024.csv",
        "test_weather_full": root / RAW_TEST / "4_Testing_Weather_Data_2024_full_year.csv",
        "test_weather_season": root / RAW_TEST / "4_Testing_Weather_Data_2024_seasons_only.csv",
        "test_ec": root / RAW_TEST / "6_Testing_EC_Data_2024.csv",
        "test_observed": root / RAW_TEST / "7_Testing_Observed_Values.csv",
    }
    missing_paths = [str(path) for path in paths.values() if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing raw inputs: {missing_paths}")
    raw_manifest = {
        name: {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": _sha256(path)}
        for name, path in paths.items()
    }

    train_traits = pd.read_csv(paths["train_traits"], low_memory=False)
    _require_columns(
        train_traits,
        ["Env", "Hybrid", TARGET, "Date_Planted", "Date_Harvested"],
        paths["train_traits"],
    )
    train_traits.insert(0, "source_row_index", np.arange(len(train_traits), dtype=np.int64))
    train_traits[TARGET] = pd.to_numeric(train_traits[TARGET], errors="coerce")
    finite_train = train_traits[train_traits[TARGET].notna()].copy()

    test_observed = pd.read_csv(paths["test_observed"], low_memory=False)
    submission = pd.read_csv(paths["submission"], low_memory=False)
    for frame, source in ((test_observed, paths["test_observed"]), (submission, paths["submission"])):
        _require_columns(frame, ["Env", "Hybrid", TARGET], source)
        frame["Hybrid"] = _validate_hybrids(frame["Hybrid"], str(source))
    test_observed.insert(0, "source_row_index", np.arange(len(test_observed), dtype=np.int64))
    test_observed[TARGET] = pd.to_numeric(test_observed[TARGET], errors="coerce")
    if test_observed[TARGET].isna().any():
        raise ValueError("Observed test file contains missing target values")

    genotypes, marker_columns, genotype_provenance = _read_genotypes(paths["genotypes"])
    genotype_names = set(genotypes["Hybrid"])
    finite_train["Hybrid"] = finite_train["Hybrid"].astype(str).str.strip()
    genotyped_train = finite_train[finite_train["Hybrid"].isin(genotype_names)].copy()
    genotyped_test = test_observed[test_observed["Hybrid"].isin(genotype_names)].copy()
    if len(genotyped_test) != len(test_observed):
        missing = sorted(set(test_observed["Hybrid"]) - genotype_names)
        raise ValueError(f"Observed test hybrids missing genotypes: {missing[:10]}")

    train_meta_raw = pd.read_csv(paths["train_meta"], low_memory=False)
    test_meta_raw = pd.read_csv(paths["test_meta"], low_memory=False)
    train_weather_season = pd.read_csv(paths["train_weather_season"], low_memory=False)
    test_weather_season = pd.read_csv(paths["test_weather_season"], low_memory=False)
    windows = _weather_windows(train_traits, train_weather_season, test_meta_raw, test_weather_season)
    train_weather_full = pd.read_csv(paths["train_weather_full"], low_memory=False)
    test_weather_full = pd.read_csv(paths["test_weather_full"], low_memory=False)
    weather_full = pd.concat([train_weather_full, test_weather_full], ignore_index=True, sort=False)
    current_weather, current_weather_prov = _aggregate_weather(
        weather_full, windows, "window_start_current", "planting_to_harvest_plus_14"
    )
    preplant_weather, preplant_weather_prov = _aggregate_weather(
        weather_full, windows, "window_start_preplant14", "preplant_14_to_harvest_plus_14"
    )

    metadata, metadata_provenance = _build_metadata(paths["train_meta"], paths["test_meta"])
    environment_index = metadata.index
    soil, soil_provenance = _build_soil(paths["train_soil"], paths["test_soil"], environment_index)

    train_ec = pd.read_csv(paths["train_ec"], low_memory=False)
    test_ec = pd.read_csv(paths["test_ec"], low_memory=False)
    _require_columns(train_ec, ["Env"], paths["train_ec"])
    _require_columns(test_ec, ["Env"], paths["test_ec"])
    ec_columns = [c for c in train_ec.columns if c != "Env"]
    if [c for c in test_ec.columns if c != "Env"] != ec_columns:
        raise ValueError("Training and testing EC schemas differ")
    ec = pd.concat([train_ec, test_ec], ignore_index=True)
    if ec["Env"].duplicated().any():
        raise ValueError("Combined EC data contains duplicate Env rows")
    ec = ec.set_index("Env")[ec_columns].apply(pd.to_numeric, errors="coerce")

    def environment_table(weather: pd.DataFrame) -> pd.DataFrame:
        result = metadata.join(weather, how="left").join(ec, how="left").join(soil, how="left")
        result["weather_block_present"] = result["weather_block_present"].fillna(0.0)
        result["weather_any_missing_days"] = result["weather_any_missing_days"].fillna(0.0)
        result["ec_block_present"] = result[ec_columns].notna().all(axis=1).astype(float)
        return result.reset_index()

    env_current = environment_table(current_weather)
    env_preplant = environment_table(preplant_weather)

    # Prove every declared exact removal on complete historical environments.
    historical_complete = env_current[
        env_current["Year"].le(2023)
        & env_current["weather_block_present"].eq(1.0)
        & env_current["ec_block_present"].eq(1.0)
    ]
    for column in CONSTANT_ENV_COLUMNS:
        if historical_complete[column].nunique(dropna=False) != 1:
            raise ValueError(f"Declared constant environment column is not constant: {column}")
    for duplicate, original in EXACT_DUPLICATE_ENV_COLUMNS.items():
        if not np.allclose(
            historical_complete[duplicate], historical_complete[original], rtol=0, atol=0, equal_nan=True
        ):
            raise ValueError(f"Declared exact duplicate differs: {duplicate} != {original}")

    train_weather_envs = set(current_weather.index)
    train_ec_envs = set(train_ec["Env"].astype(str))
    complete_train = genotyped_train[
        genotyped_train["Env"].isin(train_weather_envs & train_ec_envs)
    ].copy()
    # All observed test weather environments live in the combined weather index;
    # unlike EC, there is no intended whole-block test fill.
    complete_test = genotyped_test[genotyped_test["Env"].isin(set(current_weather.index))].copy()
    recovered_train = genotyped_train[genotyped_train["Env"].isin(set(metadata.index))].copy()
    recovered_test = genotyped_test[genotyped_test["Env"].isin(set(metadata.index))].copy()

    counts = {
        "finite_train_rows": len(finite_train),
        "genotyped_train_rows": len(genotyped_train),
        "complete_train_rows": len(complete_train),
        "recovered_train_rows": len(recovered_train),
        "observed_test_rows": len(test_observed),
        "complete_test_rows": len(complete_test),
        "submission_rows": len(submission),
        "retained_markers": len(marker_columns),
    }
    if strict_counts:
        failures = {
            key: (counts.get(key), expected)
            for key, expected in EXPECTED_COUNTS.items()
            if counts.get(key) != expected
        }
        if failures:
            raise ValueError(f"Raw-release count guard failed: {failures}")
        if counts["complete_test_rows"] != EXPECTED_COUNTS["observed_test_rows"]:
            raise ValueError("Complete observed-test row count changed unexpectedly")

    rows_complete_train = _row_table(complete_train, "complete", "complete training rows")
    rows_complete_test_with_targets = _row_table(
        complete_test, "complete", "complete observed-test rows"
    )
    rows_recovered_train = _row_table(recovered_train, "recovered", "recovered training rows")
    rows_recovered_test_with_targets = _row_table(
        recovered_test, "recovered", "recovered observed-test rows"
    )
    target_identity_columns = ["row_index", "id", "Env", "Hybrid", TARGET]
    targets_complete_test = rows_complete_test_with_targets[target_identity_columns].copy()
    targets_recovered_test = rows_recovered_test_with_targets[target_identity_columns].copy()
    # Unscored training/feature code may open rows_*_test.csv, but must be
    # structurally unable to read held-out labels.  Only the separate scorer or
    # explicit neural evaluator opens targets_*_test.csv.
    rows_complete_test = rows_complete_test_with_targets.drop(columns=[TARGET])
    rows_recovered_test = rows_recovered_test_with_targets.drop(columns=[TARGET])

    submission_view = submission.copy()
    submission_view.insert(0, "row_index", np.arange(len(submission_view), dtype=np.int64))
    submission_view["id"] = submission_view["Env"].astype(str) + "-" + submission_view["Hybrid"].astype(str)
    observed_keys = set(zip(test_observed["Env"].astype(str), test_observed["Hybrid"].astype(str)))
    submission_view["observed"] = [
        (str(env), str(hybrid)) in observed_keys
        for env, hybrid in zip(submission_view["Env"], submission_view["Hybrid"])
    ]
    submission_view["exclusion_reason"] = np.where(
        submission_view["observed"], "", "not_released_in_observed_values"
    )
    submission_view = submission_view[
        ["row_index", "id", "Env", "Hybrid", TARGET, "observed", "exclusion_reason"]
    ]

    weather_feature_columns = [
        *[f"{column}_{stat}" for stat in WEATHER_STATS for column in WEATHER_COLUMNS],
        "start_month",
        "num_days",
    ]
    removed = set(CONSTANT_ENV_COLUMNS) | set(EXACT_DUPLICATE_ENV_COLUMNS)
    base_numeric = [
        "latitude", "longitude",
        *[c for c in weather_feature_columns if c not in removed],
        *[c for c in ec_columns if c not in removed],
    ]
    recovered_numeric = [*base_numeric, "weather_block_present", "ec_block_present"]
    field_numeric = [
        *base_numeric,
        "field_latitude", "field_longitude", "station_present", "field_present",
        "station_coordinate_imputed",
    ]
    soil_numeric = [
        *base_numeric,
        *SOIL_FEATURES.values(),
        "soil_present_current", "soil_carried_prior", "soil_age_years", "soil_block_present",
    ]
    for columns, name in (
        (base_numeric, "base"), (recovered_numeric, "recovered"),
        (field_numeric, "field"), (soil_numeric, "soil"),
    ):
        absent = [c for c in columns if c not in env_current.columns]
        if absent:
            raise ValueError(f"Environment feature set {name} references absent columns: {absent}")

    generated: dict[str, Any] = {}
    generated["genotypes"] = _write_csv(genotypes, output / "genotypes.csv")
    genotype_mask = np.packbits(genotypes[marker_columns].isna().to_numpy(dtype=np.uint8), axis=1)
    mask_path = output / "genotype_missing_mask.npz"
    np.savez_compressed(
        mask_path,
        hybrid=genotypes["Hybrid"].astype(str).to_numpy(),
        packed_missing_mask=genotype_mask,
        n_markers=np.array([len(marker_columns)], dtype=np.int64),
    )
    generated["genotype_missing_mask"] = {
        "path": mask_path.name, "bytes": mask_path.stat().st_size, "sha256": _sha256(mask_path),
        "rows": len(genotypes), "packing": "numpy.packbits(axis=1)",
    }
    for key, frame, filename in (
        ("environments_current", env_current, "environments_current.csv"),
        ("environments_preplant14", env_preplant, "environments_preplant14.csv"),
        ("rows_complete_train", rows_complete_train, "rows_complete_train.csv"),
        ("rows_complete_test", rows_complete_test, "rows_complete_test.csv"),
        ("rows_recovered_train", rows_recovered_train, "rows_recovered_train.csv"),
        ("rows_recovered_test", rows_recovered_test, "rows_recovered_test.csv"),
        ("targets_complete_test", targets_complete_test, "targets_complete_test.csv"),
        ("targets_recovered_test", targets_recovered_test, "targets_recovered_test.csv"),
        ("rows_submission", submission_view, "rows_submission.csv"),
    ):
        generated[key] = _write_csv(frame, output / filename)

    variants = {
        "complete_current": {"row_policy": "complete", "window": "current", "feature_set": "base"},
        "complete_preplant14": {"row_policy": "complete", "window": "preplant14", "feature_set": "base"},
        "recovered_current": {"row_policy": "recovered", "window": "current", "feature_set": "recovered"},
        "recovered_preplant14": {"row_policy": "recovered", "window": "preplant14", "feature_set": "recovered"},
        "complete_current_field": {"row_policy": "complete", "window": "current", "feature_set": "field"},
        "complete_current_soil": {"row_policy": "complete", "window": "current", "feature_set": "soil"},
    }
    manifest_core = {
        "format": "gxe-normalized-v1",
        "dataset_id": "canonical-v4-20260715",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "builder": str(Path(__file__).resolve().relative_to(REPO_ROOT)),
        "git": _git_provenance(),
        "raw_inputs": raw_manifest,
        "generated_files": generated,
        "files": {
            entry["path"]: {"bytes": entry["bytes"], "sha256": entry["sha256"]}
            for entry in generated.values()
        },
        "counts": counts,
        "marker_columns": marker_columns,
        "marker_provenance": genotype_provenance,
        "environment_feature_sets": {
            "base": {"columns": base_numeric, "categorical": CATEGORICAL_COLUMNS},
            "recovered": {"columns": recovered_numeric, "categorical": CATEGORICAL_COLUMNS},
            "field": {"columns": field_numeric, "categorical": CATEGORICAL_COLUMNS},
            "soil": {"columns": soil_numeric, "categorical": CATEGORICAL_COLUMNS},
        },
        "environment": {
            "categorical_columns": CATEGORICAL_COLUMNS,
            "feature_sets": {
                "base": {"numeric": base_numeric, "categorical": CATEGORICAL_COLUMNS},
                "recovered": {"numeric": recovered_numeric, "categorical": CATEGORICAL_COLUMNS},
                "field": {"numeric": field_numeric, "categorical": CATEGORICAL_COLUMNS},
                "soil": {"numeric": soil_numeric, "categorical": CATEGORICAL_COLUMNS},
            },
            "constant_columns_removed": CONSTANT_ENV_COLUMNS,
            "exact_duplicate_columns_removed": EXACT_DUPLICATE_ENV_COLUMNS,
            "weather": {"current": current_weather_prov, "preplant14": preplant_weather_prov},
            "metadata": metadata_provenance,
            "soil": soil_provenance,
            "missing_value_policy": "not_applied_by_builder; fit medians on unique split-training environments",
        },
        "variants": variants,
        "row_contract": {
            "row_index": "contiguous physical position within the selected row-policy table",
            "source_row_index": "zero-based physical row in the released source CSV",
            "identity": "id is exactly Env + '-' + Hybrid; duplicate ids are valid plot replicates",
            "test": (
                "rows_*_test.csv contains no target; only explicit evaluators open "
                "targets_*_test.csv, which contains released observed values"
            ),
        },
    }
    fingerprint_payload = {
        "format": manifest_core["format"],
        "dataset_id": manifest_core["dataset_id"],
        "raw_inputs": {k: v["sha256"] for k, v in raw_manifest.items()},
        "generated_files": {k: v["sha256"] for k, v in generated.items()},
        "marker_columns": marker_columns,
        "feature_sets": manifest_core["environment"]["feature_sets"],
        "variants": variants,
    }
    manifest_core["dataset_fingerprint"] = hashlib.sha256(_canonical_json(fingerprint_payload)).hexdigest()
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(_json_safe(manifest_core), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "dataset_fingerprint": manifest_core["dataset_fingerprint"], "counts": counts, "variants": list(variants)}, indent=2))
    return manifest_core


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-count-drift", action="store_true",
        help="Allow non-release fixtures/counts; schema and integrity checks still apply.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.output if args.output.is_absolute() else args.root / args.output
    build(args.root, output, strict_counts=not args.allow_count_drift)


if __name__ == "__main__":
    main()
