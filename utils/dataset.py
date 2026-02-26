# import necessary libraries
import time, sys
import re
from pathlib import Path
from collections import defaultdict
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from transformers import (
    AutoModel, 
    AutoModelForSequenceClassification, 
    AutoModelForMaskedLM, 
    AutoTokenizer
    )

from transformers.models.bert.configuration_bert import BertConfig

from .utils import *

# categorical environment columns in X_* files
ENV_CATEGORICAL_COLS = ("Irrigated", "Treatment", "Previous_Crop")
ENV_CAT_UNK = "UNK"


def normalize_env_categorical_mode(value: str) -> str:
    """
    Normalize env categorical preprocessing mode.
    - drop: legacy baseline behavior (remove categorical env fields)
    - onehot: one-hot encode categorical env fields
    """
    v = str(value).strip().lower()
    if v in {"", "false", "0", "off", "none", "no", "drop", "legacy"}:
        return "drop"
    if v in {"true", "1", "on", "yes", "onehot", "one_hot", "ohe"}:
        return "onehot"
    raise ValueError(
        f"Unsupported env_categorical_mode='{value}'. Allowed: ['drop', 'onehot']"
    )


def _normalize_env_cat(s: pd.Series) -> pd.Series:
    """Normalize categorical env values and map missing/empty to UNK."""
    out = s.fillna(ENV_CAT_UNK).astype(str).str.strip()
    return out.replace({"": ENV_CAT_UNK, "nan": ENV_CAT_UNK, "NaN": ENV_CAT_UNK, "None": ENV_CAT_UNK})


def encode_env_categorical_features(e_block: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode known categorical environment columns while preserving numeric env features.
    Returns a float32 dataframe ready for scaling.
    """
    e_proc = e_block.copy()
    present_cats = [c for c in ENV_CATEGORICAL_COLS if c in e_proc.columns]
    for col in present_cats:
        e_proc[col] = _normalize_env_cat(e_proc[col])

    if present_cats:
        e_proc = pd.get_dummies(
            e_proc,
            columns=present_cats,
            prefix=present_cats,
            prefix_sep="=",
            dtype=np.float32,
        )

    e_proc = e_proc.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    return e_proc


def preprocess_env_features(e_block: pd.DataFrame, env_categorical_mode: str) -> pd.DataFrame:
    """
    Apply env categorical preprocessing mode and return numeric float32 env matrix.
    """
    mode = normalize_env_categorical_mode(env_categorical_mode)
    if mode == "drop":
        e_proc = e_block.drop(
            columns=[c for c in ENV_CATEGORICAL_COLS if c in e_block.columns],
            errors="ignore",
        )
        return e_proc.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(np.float32)
    return encode_env_categorical_features(e_block)

# ----------------------------------------------------------------
# helper to get year from locations file
def _env_year_from_str(env_str: str) -> int:
    # expects strings like "DEH1_2014" -> 2014
    # robust to trailing spaces etc.
    m = re.search(r'(\d{4})$', str(env_str).strip())
    if m:
        return int(m.group(1))
    raise ValueError(f"Could not parse year from Env='{env_str}'")

# ----------------------------------------------------------------
# LEO (Leave-Environment-Out) validation split helper
def compute_leo_val_envs(
    x_raw: pd.DataFrame,
    test_year: int = 2024,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> set:
    """
    Compute which environments should be held out for LEO validation.
    
    Returns a set of environment names that will be used for validation.
    These environments are entirely held out from training to better
    simulate generalization to unseen environments (like the test set).
    """
    rng = np.random.default_rng(seed)
    
    # Get environments from years before test_year
    x_raw = x_raw.copy()
    x_raw['Year'] = x_raw['Env'].astype(str).apply(_env_year_from_str)
    train_val_mask = x_raw['Year'] < test_year
    
    # Get unique environments from train/val pool
    all_envs = x_raw.loc[train_val_mask, 'Env'].unique()
    if len(all_envs) == 0:
        raise ValueError("No pre-test environments available for LEO split selection.")
    n_val_envs = min(len(all_envs), max(1, int(len(all_envs) * val_fraction)))
    
    # Randomly select environments for validation
    val_envs = set(rng.choice(all_envs, size=n_val_envs, replace=False))
    
    return val_envs


def _extract_genotype_series(x_raw: pd.DataFrame) -> pd.Series:
    """
    Return genotype/hybrid identifier for each row.
    Prefers explicit Hybrid column when present, else parses id as "{Env}-{Hybrid}".
    """
    if "Hybrid" in x_raw.columns:
        return x_raw["Hybrid"].astype(str)
    return x_raw["id"].astype(str).apply(lambda x: x.split("-", 1)[1] if "-" in x else x)


def compute_lgo_val_genotypes(
    x_raw: pd.DataFrame,
    test_year: int = 2024,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> set:
    """
    Compute which genotypes should be held out for LGO validation.
    """
    rng = np.random.default_rng(seed)

    x_raw = x_raw.copy()
    x_raw["Year"] = x_raw["Env"].astype(str).apply(_env_year_from_str)
    x_raw["GenotypeID"] = _extract_genotype_series(x_raw)
    train_val_mask = x_raw["Year"] < test_year

    all_genotypes = x_raw.loc[train_val_mask, "GenotypeID"].dropna().astype(str).unique()
    if len(all_genotypes) == 0:
        raise ValueError("No pre-test genotypes available for LGO split selection.")
    n_val_genotypes = min(len(all_genotypes), max(1, int(len(all_genotypes) * val_fraction)))
    val_genotypes = set(rng.choice(all_genotypes, size=n_val_genotypes, replace=False))

    return val_genotypes


PARENT_UNK = "UNK"
PARENT_PAIR_SEP = "|||"


def _split_hybrid_parents(hybrid: str) -> tuple[str, str]:
    raw = str(hybrid).strip()
    if not raw:
        return PARENT_UNK, PARENT_UNK
    parts = [p.strip() for p in raw.split("/", 1)]
    if len(parts) == 1:
        p1 = parts[0] if parts[0] else PARENT_UNK
        return p1, PARENT_UNK
    p1 = parts[0] if parts[0] else PARENT_UNK
    p2 = parts[1] if parts[1] else PARENT_UNK
    return p1, p2


def _extract_parent_series(hybrid_series: pd.Series) -> tuple[pd.Series, pd.Series]:
    p1_list = []
    p2_list = []
    for h in hybrid_series.astype(str).tolist():
        p1, p2 = _split_hybrid_parents(h)
        p1_list.append(p1)
        p2_list.append(p2)
    return pd.Series(p1_list, index=hybrid_series.index), pd.Series(p2_list, index=hybrid_series.index)


def _pair_key(p1: str, p2: str) -> str:
    return f"{p1}{PARENT_PAIR_SEP}{p2}"


def _fit_index_map(values: pd.Series) -> dict:
    uniq = sorted({str(v).strip() if str(v).strip() else PARENT_UNK for v in values.astype(str).tolist()})
    mapping = {PARENT_UNK: 0}
    for v in uniq:
        if v == PARENT_UNK:
            continue
        mapping[v] = len(mapping)
    return mapping


def _map_with_unk(values: pd.Series, mapping: dict) -> np.ndarray:
    return np.asarray([int(mapping.get(str(v), 0)) for v in values.astype(str).tolist()], dtype=np.int64)


def _default_key_inbreds_path(data_path: str) -> Path:
    return (Path(data_path).resolve().parent / "Training_data" / "key_inbreds_G2F_2014-2025.txt")


def _load_key_inbreds_metadata(data_path: str, parent_key_path: Optional[str] = None) -> dict:
    key_path = Path(parent_key_path) if parent_key_path else _default_key_inbreds_path(data_path)
    if not key_path.exists():
        return {}
    try:
        df = pd.read_csv(key_path, sep="\t")
    except Exception:
        return {}
    if "Cultivar" not in df.columns:
        return {}
    out = {}
    for _, row in df.iterrows():
        cultivar = str(row.get("Cultivar", "")).strip()
        if not cultivar:
            continue
        out[cultivar] = {
            "Dataset": str(row.get("Dataset", PARENT_UNK)).strip() or PARENT_UNK,
            "SourceName": str(row.get("SourceName", PARENT_UNK)).strip() or PARENT_UNK,
            "Bioproject": str(row.get("Bioproject", PARENT_UNK)).strip() or PARENT_UNK,
        }
    return out


def _shrunk_mean_map(
    key_series: pd.Series,
    value_series: pd.Series,
    alpha: float,
    global_mean: float,
) -> tuple[dict, dict]:
    frame = pd.DataFrame({"k": key_series.astype(str), "v": value_series.astype(float)})
    grp = frame.groupby("k")["v"].agg(["mean", "count"])
    count = grp["count"].astype(float)
    weight = count / (count + float(alpha))
    shrunk = weight * grp["mean"] + (1.0 - weight) * float(global_mean)
    return shrunk.to_dict(), grp["count"].astype(int).to_dict()


def _compute_train_prior_parent_history(
    years: np.ndarray,
    envs: np.ndarray,
    hybrids: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
) -> dict:
    n = len(years)
    out = {
        "p1_seen": np.zeros(n, dtype=np.float32),
        "p2_seen": np.zeros(n, dtype=np.float32),
        "pair_seen": np.zeros(n, dtype=np.float32),
        "p1_hybrid_count": np.zeros(n, dtype=np.float32),
        "p1_env_count": np.zeros(n, dtype=np.float32),
        "p1_year_count": np.zeros(n, dtype=np.float32),
        "p2_hybrid_count": np.zeros(n, dtype=np.float32),
        "p2_env_count": np.zeros(n, dtype=np.float32),
        "p2_year_count": np.zeros(n, dtype=np.float32),
    }

    p1_hyb = defaultdict(set)
    p1_env = defaultdict(set)
    p1_year = defaultdict(set)
    p2_hyb = defaultdict(set)
    p2_env = defaultdict(set)
    p2_year = defaultdict(set)
    seen_pairs = set()

    years_sorted = sorted({int(y) for y in years.tolist()})
    for y in years_sorted:
        idx = np.where(years == y)[0]
        for i in idx:
            k1 = p1[i]
            k2 = p2[i]
            pk = _pair_key(k1, k2)
            out["p1_seen"][i] = 1.0 if k1 in p1_hyb else 0.0
            out["p2_seen"][i] = 1.0 if k2 in p2_hyb else 0.0
            out["pair_seen"][i] = 1.0 if pk in seen_pairs else 0.0

            out["p1_hybrid_count"][i] = float(len(p1_hyb[k1]))
            out["p1_env_count"][i] = float(len(p1_env[k1]))
            out["p1_year_count"][i] = float(len(p1_year[k1]))
            out["p2_hybrid_count"][i] = float(len(p2_hyb[k2]))
            out["p2_env_count"][i] = float(len(p2_env[k2]))
            out["p2_year_count"][i] = float(len(p2_year[k2]))

        for i in idx:
            k1 = p1[i]
            k2 = p2[i]
            p1_hyb[k1].add(hybrids[i])
            p1_env[k1].add(envs[i])
            p1_year[k1].add(int(years[i]))
            p2_hyb[k2].add(hybrids[i])
            p2_env[k2].add(envs[i])
            p2_year[k2].add(int(years[i]))
            seen_pairs.add(_pair_key(k1, k2))

    return out


def _compute_train_oof_parent_effects(
    hybrids: pd.Series,
    p1: pd.Series,
    p2: pd.Series,
    resid: pd.Series,
    n_folds: int,
    shrink_alpha: float,
) -> dict:
    n = len(hybrids)
    if n == 0:
        return {
            "p1_gca": np.zeros(0, dtype=np.float32),
            "p2_gca": np.zeros(0, dtype=np.float32),
            "pair_sca": np.zeros(0, dtype=np.float32),
            "pair_sca_count": np.zeros(0, dtype=np.float32),
            "pair_sca_weight": np.zeros(0, dtype=np.float32),
        }
    n_folds = max(2, int(n_folds))
    fold_id = (pd.util.hash_pandas_object(hybrids.astype(str), index=False).astype(np.uint64) % np.uint64(n_folds)).astype(int).to_numpy()
    global_mean = float(resid.mean())

    p1_oof = np.full(n, global_mean, dtype=np.float32)
    p2_oof = np.full(n, global_mean, dtype=np.float32)
    pair_oof = np.full(n, global_mean, dtype=np.float32)
    pair_count_oof = np.zeros(n, dtype=np.float32)
    pair_weight_oof = np.zeros(n, dtype=np.float32)

    pair_key_all = (p1.astype(str) + PARENT_PAIR_SEP + p2.astype(str))
    for f in range(n_folds):
        train_mask = fold_id != f
        valid_mask = ~train_mask
        if not np.any(valid_mask):
            continue

        p1_map, _ = _shrunk_mean_map(p1[train_mask], resid[train_mask], shrink_alpha, global_mean)
        p2_map, _ = _shrunk_mean_map(p2[train_mask], resid[train_mask], shrink_alpha, global_mean)
        pair_map, pair_count_map = _shrunk_mean_map(pair_key_all[train_mask], resid[train_mask], shrink_alpha, global_mean)

        idx = np.where(valid_mask)[0]
        p1_oof[idx] = p1.iloc[idx].astype(str).map(p1_map).fillna(global_mean).to_numpy(dtype=np.float32)
        p2_oof[idx] = p2.iloc[idx].astype(str).map(p2_map).fillna(global_mean).to_numpy(dtype=np.float32)
        pair_vals = pair_key_all.iloc[idx].astype(str)
        pair_oof[idx] = pair_vals.map(pair_map).fillna(global_mean).to_numpy(dtype=np.float32)
        counts = pair_vals.map(pair_count_map).fillna(0).to_numpy(dtype=np.float32)
        pair_count_oof[idx] = counts
        pair_weight_oof[idx] = counts / (counts + float(shrink_alpha))

    return {
        "p1_gca": p1_oof,
        "p2_gca": p2_oof,
        "pair_sca": pair_oof,
        "pair_sca_count": pair_count_oof,
        "pair_sca_weight": pair_weight_oof,
    }


# can use this for both rolling and non-rolling training
class GxE_Dataset(Dataset):

    def __init__(self,
                 split='train', # default LYO: train <= 2022, val == 2023
                 data_path='data/maize_data_2014-2023_vs_2024_v2/', # need to go up one level and then down to data directory
                 residual: bool = False,
                 scaler: StandardScaler | None = None,
                 train_year_max: int | None = None,
                 val_year: int | None = None,
                 y_scalers: Optional[Dict[str, LabelScaler]] = None,
                 scale_targets: bool = True,
                 g_input_type: str = "tokens",
                 env_categorical_mode: str = "drop",
                 marker_stats: Optional[Dict[str, object]] = None,
                 val_prediction: str = "lyo",
                 val_holdout_ids: Optional[set] = None,
                 leo_val: bool = False,
                 leo_val_envs: Optional[set] = None,
                 leo_val_fraction: float = 0.15,
                 lgo_val_fraction: float = 0.15,
                 leo_seed: int = 42,
                 parent_features: bool = False,
                 parent_use_embeddings: bool = True,
                 parent_use_interaction: bool = True,
                 parent_use_seen_flags: bool = True,
                 parent_use_history_features: bool = True,
                 parent_use_gca_features: bool = True,
                 parent_use_sca_features: bool = True,
                 parent_use_source_meta: bool = True,
                 parent_oof_folds: int = 5,
                 parent_shrink_alpha: float = 10.0,
                 parent_stats: Optional[Dict[str, object]] = None,
                 parent_key_path: Optional[str] = None,
                 ):
        
        """
        Parameters:
            split (str): 'train' (2014-2022), 'val' (2023), 'test' (2024), or 'sub'
            data_path (str): path to data directory
            residual (bool): if True, return residual targets; else return total targets
            scaler (StandardScaler|None): if None and split=='train', fit here; otherwise reuse passed scaler
            train_year_max (int|None): if not None and split=='train', filter to years <= this value
            val_year (int|None): if not None and split=='val', filter to this year
            y_scalers (Optional[Dict[str, LabelScaler]]): if not None, use these scalers for y
            scale_targets (bool): if True, scale targets using y_scalers
            g_input_type (str): "tokens" for discrete marker tokens, "grm" for GRM-standardized marker features
            env_categorical_mode (str): "drop" (legacy baseline) or "onehot" categorical env handling
            marker_stats (Optional[Dict[str, object]]): train-fitted marker stats required for val/test in g_input_type='grm'
            val_prediction (str): 'lyo' (default), 'leo', or 'lgo' validation split mode
            val_holdout_ids (Optional[set]): pre-computed holdout ids from train split (envs for LEO, genotypes for LGO)
            leo_val (bool): deprecated alias for val_prediction='leo'
            leo_val_envs (Optional[set]): deprecated alias for val_holdout_ids in LEO mode
            leo_val_fraction (float): fraction of environments to hold out for LEO val
            lgo_val_fraction (float): fraction of genotypes to hold out for LGO val
            leo_seed (int): random seed for LEO/LGO holdout selection
            parent_features (bool): enable parent-aware feature pipeline
            parent_use_embeddings (bool): expose parent ids for learned embeddings
            parent_use_interaction (bool): expose explicit p1 x p2 embedding interaction path
            parent_use_seen_flags (bool): add seen/novel parent flags
            parent_use_history_features (bool): add parent history counts (hybrid/env/year)
            parent_use_gca_features (bool): add shrinkage GCA-like parent effects
            parent_use_sca_features (bool): add shrinkage SCA-like pair effects
            parent_use_source_meta (bool): add parent source metadata ids and same_source flag
            parent_oof_folds (int): folds for OOF parent-effect estimation in train split
            parent_shrink_alpha (float): shrinkage strength for GCA/SCA effects
            parent_stats (Optional[Dict[str, object]]): train-fitted parent stats for val/test
            parent_key_path (Optional[str]): override path to key_inbreds metadata file
        """
        super().__init__()
        self.split = split
        self.data_path = data_path
        self.residual_flag = residual
        self.scale_targets = scale_targets
        self.g_input_type = str(g_input_type).strip().lower()
        self.env_categorical_mode = normalize_env_categorical_mode(env_categorical_mode)
        if leo_val and str(val_prediction).strip().lower() == "lyo":
            val_prediction = "leo"
        self.val_prediction = normalize_val_prediction_mode(val_prediction)
        self.leo_val = self.val_prediction == "leo"
        self.lgo_val = self.val_prediction == "lgo"
        self.parent_features = bool(parent_features)
        self.parent_use_embeddings = bool(parent_use_embeddings) and self.parent_features
        self.parent_use_interaction = bool(parent_use_interaction) and self.parent_use_embeddings
        self.parent_use_seen_flags = bool(parent_use_seen_flags) and self.parent_features
        self.parent_use_history_features = bool(parent_use_history_features) and self.parent_features
        self.parent_use_gca_features = bool(parent_use_gca_features) and self.parent_features
        self.parent_use_sca_features = bool(parent_use_sca_features) and self.parent_features
        self.parent_use_source_meta = bool(parent_use_source_meta) and self.parent_features
        self.parent_oof_folds = int(parent_oof_folds)
        self.parent_shrink_alpha = float(parent_shrink_alpha)
        self.parent_key_path = parent_key_path
        self.parent_stats = None
        self.parent_numeric_cols = []
        self.parent1_id_arr = None
        self.parent2_id_arr = None
        self.parent1_dataset_id_arr = None
        self.parent2_dataset_id_arr = None
        self.parent1_source_id_arr = None
        self.parent2_source_id_arr = None
        self.parent1_bioproject_id_arr = None
        self.parent2_bioproject_id_arr = None
        self.n_parent1_ids = 1
        self.n_parent2_ids = 1
        self.n_parent_dataset_ids = 1
        self.n_parent_source_ids = 1
        self.n_parent_bioproject_ids = 1
        if val_holdout_ids is None and leo_val_envs is not None:
            val_holdout_ids = leo_val_envs
        if self.g_input_type not in {"tokens", "grm"}:
            raise ValueError(f"g_input_type must be one of ['tokens', 'grm'] (got {g_input_type})")

        ############################################
        ### N_ENV FEATURES IN ORIGINAL DATA HERE ###
        ############################################ 
        N_ENV = 705

        ### LOAD DATA ###
        if split == 'train':
            x_path = data_path + 'X_train.csv'
            y_path = data_path + 'y_train.csv'
        elif split == 'val':
            x_path = data_path + 'X_train.csv'
            y_path = data_path + 'y_train.csv'
        elif split == 'test':
            x_path = data_path + 'X_test.csv'
            y_path = data_path + 'y_test.csv'
        else:
            raise ValueError(f"Invalid split='{split}'")
        
        ### READ X/Y ###
        # keep cols 'id', 'Env', stored in X_* files
        x_raw = pd.read_csv(x_path)
        y_raw = pd.read_csv(y_path)

        # drop unnamed index cols, if present
        x_raw = x_raw.loc[:, ~x_raw.columns.str.contains(r'^Unnamed')]
        y_raw = y_raw.loc[:, ~y_raw.columns.str.contains(r'^Unnamed')]

        # sanity checks
        if 'Env' not in x_raw.columns:
            raise ValueError("X_* file must contain an 'Env' column.")
        if 'id' not in x_raw.columns:
            # if 'id' used as an index upstream, recover it
            if x_raw.index.name and x_raw.index.name.lower() == 'id':
                x_raw = x_raw.reset_index().rename(columns={x_raw.columns[0]: 'id'})
            else:
                raise ValueError("X_* file must contain an 'id' column.")

        # derive Year from Env
        x_raw['Year'] = x_raw['Env'].astype(str).apply(_env_year_from_str)

        ### SPLIT MASK (ROLLING SETUP DEFAULTS) ###
        # Explicit split modes:
        # - LYO: year holdout (legacy default)
        # - LEO: environment holdout
        # - LGO: genotype holdout
        if split in ("train", "val") and self.val_prediction in {"leo", "lgo"}:
            pre_test_mask = x_raw["Year"] < 2024
            if self.val_prediction == "leo":
                if val_holdout_ids is None:
                    val_holdout_ids = compute_leo_val_envs(
                        x_raw, test_year=2024, val_fraction=leo_val_fraction, seed=leo_seed
                    )
                in_val = x_raw["Env"].isin(val_holdout_ids)
            else:
                x_raw["GenotypeID"] = _extract_genotype_series(x_raw)
                if val_holdout_ids is None:
                    val_holdout_ids = compute_lgo_val_genotypes(
                        x_raw, test_year=2024, val_fraction=lgo_val_fraction, seed=leo_seed
                    )
                in_val = x_raw["GenotypeID"].isin(val_holdout_ids)

            self.val_holdout_ids = set(val_holdout_ids)
            if split == "train":
                keep_mask = pre_test_mask & ~in_val
            else:
                keep_mask = pre_test_mask & in_val
        else:
            self.val_holdout_ids = None
            if split == "train":
                cutoff = 2022 if train_year_max is None else train_year_max
                keep_mask = x_raw['Year'] <= cutoff
            elif split == "val":
                which = 2023 if val_year is None else val_year
                keep_mask = x_raw['Year'] == which
            else: # 'test', 'sub'
                keep_mask = x_raw['Year'] >= 2024
        self.leo_val_envs = self.val_holdout_ids if self.val_prediction == "leo" else None
        self.lgo_val_genotypes = self.val_holdout_ids if self.val_prediction == "lgo" else None
        
        ### FILTER/ALIGN X/Y BY MASK BUILT ON X ###
        x_filt = x_raw.loc[keep_mask.values].reset_index(drop=True)
        y_filt = y_raw.loc[keep_mask.values].reset_index(drop=True)

        # sanity check
        if len(x_filt) != len(y_filt):
            raise ValueError(f"Length mismatch after filtering: X={len(x_filt)}, Y={len(y_filt)}."
                             "Ensue X_* and y_* have identical row order.")

        # separate metadata
        self.meta = x_filt[['id', 'Env', 'Year']].copy()

        ### CATEGORICAL ENV MAPPING FOR ENVWISE LOSSES ###
        # use *only* filtered envs so codes match indices
        self.env_codes = pd.Categorical(self.meta['Env'].astype(str))
        self.env_id_tensor = torch.tensor(self.env_codes.codes, dtype=torch.long)

        ### HYBRID ID MAPPING FOR CONTRASTIVE LOSSES ###
        # Extract hybrid name from id column (format: "{Env}-{Hybrid}")
        # Falls back to full id if no "-" separator is present
        hybrid_names = _extract_genotype_series(x_filt)
        self.hybrid_codes = pd.Categorical(hybrid_names)
        self.hybrid_id_tensor = torch.tensor(self.hybrid_codes.codes, dtype=torch.long)
        parent1_series, parent2_series = _extract_parent_series(hybrid_names)
        pair_series = parent1_series.astype(str) + PARENT_PAIR_SEP + parent2_series.astype(str)

        parent_numeric = pd.DataFrame(index=x_filt.index)
        if self.parent_features:
            if split == "train":
                parent_meta = _load_key_inbreds_metadata(data_path, self.parent_key_path)
                p1_to_idx = _fit_index_map(parent1_series)
                p2_to_idx = _fit_index_map(parent2_series)

                p1_dataset = parent1_series.astype(str).map(lambda p: parent_meta.get(p, {}).get("Dataset", PARENT_UNK))
                p2_dataset = parent2_series.astype(str).map(lambda p: parent_meta.get(p, {}).get("Dataset", PARENT_UNK))
                p1_source = parent1_series.astype(str).map(lambda p: parent_meta.get(p, {}).get("SourceName", PARENT_UNK))
                p2_source = parent2_series.astype(str).map(lambda p: parent_meta.get(p, {}).get("SourceName", PARENT_UNK))
                p1_bioproject = parent1_series.astype(str).map(lambda p: parent_meta.get(p, {}).get("Bioproject", PARENT_UNK))
                p2_bioproject = parent2_series.astype(str).map(lambda p: parent_meta.get(p, {}).get("Bioproject", PARENT_UNK))

                dataset_to_idx = _fit_index_map(pd.concat([p1_dataset, p2_dataset], ignore_index=True))
                source_to_idx = _fit_index_map(pd.concat([p1_source, p2_source], ignore_index=True))
                bioproject_to_idx = _fit_index_map(pd.concat([p1_bioproject, p2_bioproject], ignore_index=True))

                y_total_raw = y_filt["Yield_Mg_ha"].astype(float).reset_index(drop=True)
                env_series = self.meta["Env"].astype(str).reset_index(drop=True)
                resid = y_total_raw - y_total_raw.groupby(env_series).transform("mean")
                global_resid = float(resid.mean())

                p1_gca_map, _ = _shrunk_mean_map(parent1_series, resid, self.parent_shrink_alpha, global_resid)
                p2_gca_map, _ = _shrunk_mean_map(parent2_series, resid, self.parent_shrink_alpha, global_resid)
                pair_gca_map, pair_count_map = _shrunk_mean_map(pair_series, resid, self.parent_shrink_alpha, global_resid)

                # full-history maps for val/test projection
                p1_hist_hybrid_map = pd.DataFrame({"p": parent1_series, "hyb": hybrid_names}).groupby("p")["hyb"].nunique().to_dict()
                p1_hist_env_map = pd.DataFrame({"p": parent1_series, "env": self.meta["Env"].astype(str)}).groupby("p")["env"].nunique().to_dict()
                p1_hist_year_map = pd.DataFrame({"p": parent1_series, "year": self.meta["Year"].astype(int)}).groupby("p")["year"].nunique().to_dict()
                p2_hist_hybrid_map = pd.DataFrame({"p": parent2_series, "hyb": hybrid_names}).groupby("p")["hyb"].nunique().to_dict()
                p2_hist_env_map = pd.DataFrame({"p": parent2_series, "env": self.meta["Env"].astype(str)}).groupby("p")["env"].nunique().to_dict()
                p2_hist_year_map = pd.DataFrame({"p": parent2_series, "year": self.meta["Year"].astype(int)}).groupby("p")["year"].nunique().to_dict()

                prior_hist = _compute_train_prior_parent_history(
                    years=self.meta["Year"].astype(int).to_numpy(),
                    envs=self.meta["Env"].astype(str).to_numpy(),
                    hybrids=hybrid_names.astype(str).to_numpy(),
                    p1=parent1_series.astype(str).to_numpy(),
                    p2=parent2_series.astype(str).to_numpy(),
                )
                oof_effects = _compute_train_oof_parent_effects(
                    hybrids=hybrid_names.astype(str),
                    p1=parent1_series.astype(str),
                    p2=parent2_series.astype(str),
                    resid=resid.astype(float),
                    n_folds=self.parent_oof_folds,
                    shrink_alpha=self.parent_shrink_alpha,
                )

                self.parent_stats = {
                    "p1_to_idx": p1_to_idx,
                    "p2_to_idx": p2_to_idx,
                    "dataset_to_idx": dataset_to_idx,
                    "source_to_idx": source_to_idx,
                    "bioproject_to_idx": bioproject_to_idx,
                    "train_p1_set": list(set(parent1_series.astype(str).tolist())),
                    "train_p2_set": list(set(parent2_series.astype(str).tolist())),
                    "train_pair_set": list(set(pair_series.astype(str).tolist())),
                    "p1_hist_hybrid_map": {str(k): int(v) for k, v in p1_hist_hybrid_map.items()},
                    "p1_hist_env_map": {str(k): int(v) for k, v in p1_hist_env_map.items()},
                    "p1_hist_year_map": {str(k): int(v) for k, v in p1_hist_year_map.items()},
                    "p2_hist_hybrid_map": {str(k): int(v) for k, v in p2_hist_hybrid_map.items()},
                    "p2_hist_env_map": {str(k): int(v) for k, v in p2_hist_env_map.items()},
                    "p2_hist_year_map": {str(k): int(v) for k, v in p2_hist_year_map.items()},
                    "p1_gca_map": {str(k): float(v) for k, v in p1_gca_map.items()},
                    "p2_gca_map": {str(k): float(v) for k, v in p2_gca_map.items()},
                    "pair_sca_map": {str(k): float(v) for k, v in pair_gca_map.items()},
                    "pair_count_map": {str(k): int(v) for k, v in pair_count_map.items()},
                    "global_resid": float(global_resid),
                    "parent_shrink_alpha": float(self.parent_shrink_alpha),
                    "key_parent_meta": parent_meta,
                    "options": {
                        "use_embeddings": self.parent_use_embeddings,
                        "use_interaction": self.parent_use_interaction,
                        "use_seen_flags": self.parent_use_seen_flags,
                        "use_history_features": self.parent_use_history_features,
                        "use_gca_features": self.parent_use_gca_features,
                        "use_sca_features": self.parent_use_sca_features,
                        "use_source_meta": self.parent_use_source_meta,
                    },
                }
            else:
                if parent_stats is None:
                    raise ValueError("For val/test with parent_features=True, pass parent_stats from train split.")
                self.parent_stats = parent_stats
                opts = dict(self.parent_stats.get("options", {}))
                if opts:
                    self.parent_use_embeddings = bool(opts.get("use_embeddings", self.parent_use_embeddings))
                    self.parent_use_interaction = bool(opts.get("use_interaction", self.parent_use_interaction))
                    self.parent_use_seen_flags = bool(opts.get("use_seen_flags", self.parent_use_seen_flags))
                    self.parent_use_history_features = bool(opts.get("use_history_features", self.parent_use_history_features))
                    self.parent_use_gca_features = bool(opts.get("use_gca_features", self.parent_use_gca_features))
                    self.parent_use_sca_features = bool(opts.get("use_sca_features", self.parent_use_sca_features))
                    self.parent_use_source_meta = bool(opts.get("use_source_meta", self.parent_use_source_meta))
                p1_to_idx = dict(self.parent_stats.get("p1_to_idx", {PARENT_UNK: 0}))
                p2_to_idx = dict(self.parent_stats.get("p2_to_idx", {PARENT_UNK: 0}))
                dataset_to_idx = dict(self.parent_stats.get("dataset_to_idx", {PARENT_UNK: 0}))
                source_to_idx = dict(self.parent_stats.get("source_to_idx", {PARENT_UNK: 0}))
                bioproject_to_idx = dict(self.parent_stats.get("bioproject_to_idx", {PARENT_UNK: 0}))
                parent_meta = dict(self.parent_stats.get("key_parent_meta", {}))
                p1_dataset = parent1_series.astype(str).map(lambda p: parent_meta.get(p, {}).get("Dataset", PARENT_UNK))
                p2_dataset = parent2_series.astype(str).map(lambda p: parent_meta.get(p, {}).get("Dataset", PARENT_UNK))
                p1_source = parent1_series.astype(str).map(lambda p: parent_meta.get(p, {}).get("SourceName", PARENT_UNK))
                p2_source = parent2_series.astype(str).map(lambda p: parent_meta.get(p, {}).get("SourceName", PARENT_UNK))
                p1_bioproject = parent1_series.astype(str).map(lambda p: parent_meta.get(p, {}).get("Bioproject", PARENT_UNK))
                p2_bioproject = parent2_series.astype(str).map(lambda p: parent_meta.get(p, {}).get("Bioproject", PARENT_UNK))

                prior_hist = None
                oof_effects = None

            # ids for embedding features
            self.parent1_id_arr = _map_with_unk(parent1_series, p1_to_idx)
            self.parent2_id_arr = _map_with_unk(parent2_series, p2_to_idx)
            self.n_parent1_ids = max(1, len(p1_to_idx))
            self.n_parent2_ids = max(1, len(p2_to_idx))

            # optional parent metadata ids
            self.parent1_dataset_id_arr = _map_with_unk(p1_dataset, dataset_to_idx)
            self.parent2_dataset_id_arr = _map_with_unk(p2_dataset, dataset_to_idx)
            self.parent1_source_id_arr = _map_with_unk(p1_source, source_to_idx)
            self.parent2_source_id_arr = _map_with_unk(p2_source, source_to_idx)
            self.parent1_bioproject_id_arr = _map_with_unk(p1_bioproject, bioproject_to_idx)
            self.parent2_bioproject_id_arr = _map_with_unk(p2_bioproject, bioproject_to_idx)
            self.n_parent_dataset_ids = max(1, len(dataset_to_idx))
            self.n_parent_source_ids = max(1, len(source_to_idx))
            self.n_parent_bioproject_ids = max(1, len(bioproject_to_idx))

            # Seen flags (novelty) and history counts
            if self.parent_use_seen_flags:
                if split == "train":
                    parent_numeric["parent_p1_seen_train"] = prior_hist["p1_seen"]
                    parent_numeric["parent_p2_seen_train"] = prior_hist["p2_seen"]
                    parent_numeric["parent_pair_seen_train"] = prior_hist["pair_seen"]
                else:
                    p1_seen_set = set(self.parent_stats.get("train_p1_set", []))
                    p2_seen_set = set(self.parent_stats.get("train_p2_set", []))
                    pair_seen_set = set(self.parent_stats.get("train_pair_set", []))
                    parent_numeric["parent_p1_seen_train"] = parent1_series.astype(str).map(lambda k: 1.0 if k in p1_seen_set else 0.0)
                    parent_numeric["parent_p2_seen_train"] = parent2_series.astype(str).map(lambda k: 1.0 if k in p2_seen_set else 0.0)
                    parent_numeric["parent_pair_seen_train"] = pair_series.astype(str).map(lambda k: 1.0 if k in pair_seen_set else 0.0)

            if self.parent_use_history_features:
                if split == "train":
                    parent_numeric["parent_p1_hist_hybrid_count"] = prior_hist["p1_hybrid_count"]
                    parent_numeric["parent_p1_hist_env_count"] = prior_hist["p1_env_count"]
                    parent_numeric["parent_p1_hist_year_count"] = prior_hist["p1_year_count"]
                    parent_numeric["parent_p2_hist_hybrid_count"] = prior_hist["p2_hybrid_count"]
                    parent_numeric["parent_p2_hist_env_count"] = prior_hist["p2_env_count"]
                    parent_numeric["parent_p2_hist_year_count"] = prior_hist["p2_year_count"]
                else:
                    parent_numeric["parent_p1_hist_hybrid_count"] = parent1_series.astype(str).map(self.parent_stats.get("p1_hist_hybrid_map", {})).fillna(0.0)
                    parent_numeric["parent_p1_hist_env_count"] = parent1_series.astype(str).map(self.parent_stats.get("p1_hist_env_map", {})).fillna(0.0)
                    parent_numeric["parent_p1_hist_year_count"] = parent1_series.astype(str).map(self.parent_stats.get("p1_hist_year_map", {})).fillna(0.0)
                    parent_numeric["parent_p2_hist_hybrid_count"] = parent2_series.astype(str).map(self.parent_stats.get("p2_hist_hybrid_map", {})).fillna(0.0)
                    parent_numeric["parent_p2_hist_env_count"] = parent2_series.astype(str).map(self.parent_stats.get("p2_hist_env_map", {})).fillna(0.0)
                    parent_numeric["parent_p2_hist_year_count"] = parent2_series.astype(str).map(self.parent_stats.get("p2_hist_year_map", {})).fillna(0.0)

            if self.parent_use_gca_features:
                if split == "train":
                    parent_numeric["parent_p1_gca"] = oof_effects["p1_gca"]
                    parent_numeric["parent_p2_gca"] = oof_effects["p2_gca"]
                else:
                    global_resid = float(self.parent_stats.get("global_resid", 0.0))
                    parent_numeric["parent_p1_gca"] = parent1_series.astype(str).map(self.parent_stats.get("p1_gca_map", {})).fillna(global_resid)
                    parent_numeric["parent_p2_gca"] = parent2_series.astype(str).map(self.parent_stats.get("p2_gca_map", {})).fillna(global_resid)

            if self.parent_use_sca_features:
                if split == "train":
                    parent_numeric["parent_pair_sca"] = oof_effects["pair_sca"]
                    parent_numeric["parent_pair_sca_count"] = oof_effects["pair_sca_count"]
                    parent_numeric["parent_pair_sca_weight"] = oof_effects["pair_sca_weight"]
                else:
                    global_resid = float(self.parent_stats.get("global_resid", 0.0))
                    alpha = float(self.parent_stats.get("parent_shrink_alpha", self.parent_shrink_alpha))
                    parent_numeric["parent_pair_sca"] = pair_series.astype(str).map(self.parent_stats.get("pair_sca_map", {})).fillna(global_resid)
                    counts = pair_series.astype(str).map(self.parent_stats.get("pair_count_map", {})).fillna(0.0)
                    parent_numeric["parent_pair_sca_count"] = counts
                    parent_numeric["parent_pair_sca_weight"] = counts.astype(float) / (counts.astype(float) + alpha)

            if self.parent_use_source_meta:
                parent_numeric["parent_same_source"] = (
                    (p1_source.astype(str) == p2_source.astype(str))
                    & (p1_source.astype(str) != PARENT_UNK)
                ).astype(np.float32)

            parent_numeric = parent_numeric.fillna(0.0).astype(np.float32)
            self.parent_numeric_cols = list(parent_numeric.columns)
        else:
            self.parent_stats = None

        ### BUILD FT. MATRICES ###
        # drop metadata and accidental columns
        feature_df = x_filt.drop(columns=['id', 'Env', 'Year', 'Hybrid', 'GenotypeID', 'Yield_Mg_ha'], errors='ignore')

        # sanity
        if feature_df.shape[1] < N_ENV:
            raise ValueError(f"Feature matrix has fewer columns ({feature_df.shape[1]}) than N_ENV ({N_ENV})."
                             "Check your X_* file layout.")
        
        # genotype features (all but last N_ENV)
        g_block = feature_df.iloc[:, :-N_ENV].astype(np.float32)
        e_block = feature_df.iloc[:, -N_ENV:]

        e_block = preprocess_env_features(e_block, self.env_categorical_mode)
        if self.parent_features and len(parent_numeric.columns):
            e_block = pd.concat([e_block.reset_index(drop=True), parent_numeric.reset_index(drop=True)], axis=1)
        # Keep val/test schema aligned with the train-fitted scaler feature order.
        if split != "train" and scaler is not None and hasattr(scaler, "feature_names_in_"):
            ref_cols = [str(c) for c in scaler.feature_names_in_.tolist()]
            e_block = e_block.reindex(columns=ref_cols, fill_value=0.0)
        elif split != "train" and scaler is not None and hasattr(scaler, "n_features_in_"):
            expected = int(scaler.n_features_in_)
            if e_block.shape[1] != expected:
                raise ValueError(
                    f"Encoded env feature dimension mismatch: got {e_block.shape[1]}, expected {expected}. "
                    "Use checkpoints/scalers that include feature_names_in for robust categorical alignment."
                )


        # store for __getitem__
        # marker inputs can be either:
        # - tokens: discrete dosages {0,1,2}
        # - grm: standardized dosage features (x - 2p) / sqrt(2p(1-p))
        g_dosage = (g_block * 2.0).astype(np.float32)  # convert from [0, 0.5, 1] -> [0, 1, 2]
        self.g_raw_dosage = g_dosage
        self.marker_stats = None
        if self.g_input_type == "tokens":
            self.g_data = g_dosage.round().astype('int64')
        else:
            if split == "train":
                p = np.clip(g_dosage.mean(axis=0).to_numpy(dtype=np.float32) / 2.0, 0.0, 1.0)
                scale = np.sqrt(2.0 * p * (1.0 - p)).astype(np.float32)
                valid = (scale > 1e-8).astype(np.float32)
                scale_safe = scale.copy()
                scale_safe[valid < 0.5] = 1.0
                self.marker_stats = {
                    "p": p,
                    "scale": scale_safe,
                    "valid": valid,
                    "columns": list(g_dosage.columns),
                }
            else:
                if marker_stats is None:
                    raise ValueError("For val/test/sub with g_input_type='grm', you must pass marker_stats from train.")
                p = np.asarray(marker_stats["p"], dtype=np.float32)
                scale_safe = np.asarray(marker_stats["scale"], dtype=np.float32)
                valid = np.asarray(marker_stats["valid"], dtype=np.float32)
                cols = list(marker_stats.get("columns", list(g_dosage.columns)))
                if cols != list(g_dosage.columns):
                    raise ValueError("marker_stats columns do not match current genotype columns.")
                self.marker_stats = {
                    "p": p,
                    "scale": scale_safe,
                    "valid": valid,
                    "columns": cols,
                }
            g_scaled = (g_dosage.to_numpy(dtype=np.float32) - (2.0 * p)[None, :]) / scale_safe[None, :]
            g_scaled = g_scaled * valid[None, :]
            self.g_data = pd.DataFrame(g_scaled, columns=g_dosage.columns, index=g_dosage.index)
        self.e_cols = list(e_block.columns)

        # env scaling (fit on train, reuse elsewhere)
        self.scaler = scaler if scaler is not None else StandardScaler()
        if split == 'train':
            e_scaled = self.scaler.fit_transform(e_block)
        else:
            if scaler is None:
                raise ValueError("For val/test/sub you must pass a fitted scaler.")
            e_scaled = self.scaler.transform(e_block)
        self.e_data = pd.DataFrame(e_scaled, columns=self.e_cols, index=self.g_data.index)

        ### TARGETS / METADATA FOR OUTPUTS ###
        self.y_data = y_filt.reset_index(drop=True)

        # submission convenience
        if split == "sub":
            self.y_data = self.y_data[['Env', 'Hybrid', 'Yield_Mg_ha']]
        
        ### TARGETS: TOTAL, ENV MEAN, RESIDUAL ###
        # sanity
        if 'Yield_Mg_ha' not in self.y_data.columns:
            raise ValueError("y_* file must contain 'Yield_Mg_ha'.")
        
        total = self.y_data['Yield_Mg_ha'].astype(float)
        env_key = self.meta['Env'].astype(str).values # length = rows
        ymean = total.groupby(env_key).transform('mean')
        resid = total - ymean

        ### TARGET SCALING ###
        self.label_scalers: Dict[str, LabelScaler] = {}
        if self.scale_targets:
            if self.split == "train":
                def fit_ls(series: pd.Series) -> LabelScaler:
                    return LabelScaler(mean=float(series.mean()), std=float(series.std(ddof=0)))
                self.label_scalers['total'] = fit_ls(total)
                self.label_scalers['ymean'] = fit_ls(ymean)
                self.label_scalers['resid'] = fit_ls(resid)
            else:
                if y_scalers is None:
                    raise ValueError("For val/test/sub you must pass y_scalers.")
                self.label_scalers = y_scalers
            
            total = pd.Series(self.label_scalers['total'].transform(total.values))
            ymean = pd.Series(self.label_scalers['ymean'].transform(ymean.values))
            resid = pd.Series(self.label_scalers['resid'].transform(resid.values))

        self.total_series = total.reset_index(drop=True)
        self.env_mean = ymean.reset_index(drop=True)
        self.residual = resid.reset_index(drop=True)

        # block size for tokenizers, etc.
        self.block_size = self.g_data.shape[1]
        self.n_env_fts = len(self.e_cols)

        # final sanity check 
        assert len(self.env_id_tensor) == len(self.g_data) == len(self.e_data) == len(self.y_data), \
            "Misalignment after filtering and feature/target construction"
        
    def __len__(self):
        # return length (number of rows) in dataset
        return len(self.g_data)
    
    def __getitem__(self, index: int):

        """
        Returns:
            x: {'g_data': Long[g_tokens], 'e_data': Float[env_features]}
            y:
                - for train/val: dict with 'y' (or residual pieces) and 'env_id'
                - for test/sub: dict with 'Env', 'Hybrid', 'Yield_Mg_ha'
        """
        if self.g_input_type == "tokens":
            g_tensor = torch.tensor(self.g_data.iloc[index, :].values, dtype=torch.long)
        else:
            g_tensor = torch.tensor(self.g_data.iloc[index, :].values, dtype=torch.float32)
        env_data = torch.tensor(self.e_data.iloc[index, :].values, dtype=torch.float32)
        x = {'g_data': g_tensor, 'e_data': env_data}
        if self.parent_features and self.parent_use_embeddings:
            x["parent1_id"] = torch.tensor(int(self.parent1_id_arr[index]), dtype=torch.long)
            x["parent2_id"] = torch.tensor(int(self.parent2_id_arr[index]), dtype=torch.long)
            if self.parent_use_source_meta:
                x["parent1_dataset_id"] = torch.tensor(int(self.parent1_dataset_id_arr[index]), dtype=torch.long)
                x["parent2_dataset_id"] = torch.tensor(int(self.parent2_dataset_id_arr[index]), dtype=torch.long)
                x["parent1_source_id"] = torch.tensor(int(self.parent1_source_id_arr[index]), dtype=torch.long)
                x["parent2_source_id"] = torch.tensor(int(self.parent2_source_id_arr[index]), dtype=torch.long)
                x["parent1_bioproject_id"] = torch.tensor(int(self.parent1_bioproject_id_arr[index]), dtype=torch.long)
                x["parent2_bioproject_id"] = torch.tensor(int(self.parent2_bioproject_id_arr[index]), dtype=torch.long)
        if self.g_input_type == "grm":
            x["g_data_raw"] = torch.tensor(self.g_raw_dosage.iloc[index, :].values, dtype=torch.float32)
        
        if self.split in ('sub', 'test'):
            row = self.y_data.iloc[index]
            # return only the columns that exist (Env and Yield_Mg_ha are typical)
            keep_cols = [c for c in ('Env', 'Hybrid', 'Yield_Mg_ha') if c in self.y_data.columns]
            y = {c: row[c] for c in keep_cols}
            return x, y

        # scaled targets if scale_targets=True or raw otherwise 
        y_total = torch.tensor([self.total_series.iloc[index]], dtype=torch.float32)
        env_id = self.env_id_tensor[index]

        hybrid_id = self.hybrid_id_tensor[index]

        if not self.residual_flag:
            return x, {"y": y_total, "env_id": env_id, "hybrid_id": hybrid_id}

        # residual out
        y_env_mean = torch.tensor([self.env_mean.iloc[index]], dtype=torch.float32)
        y_residual = torch.tensor([self.residual.iloc[index]], dtype=torch.float32)
        targets = {
            'total': y_total,
            'ymean': y_env_mean,
            'resid': y_residual,
            'env_id': env_id,
            'hybrid_id': hybrid_id,
        }
        return x, targets
