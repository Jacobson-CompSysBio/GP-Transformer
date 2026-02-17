# import necessary libraries
import time, sys
import re
from pathlib import Path
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
    n_val_envs = max(1, int(len(all_envs) * val_fraction))
    
    # Randomly select environments for validation
    val_envs = set(rng.choice(all_envs, size=n_val_envs, replace=False))
    
    return val_envs


# can use this for both rolling and non-rolling training
class GxE_Dataset(Dataset):

    def __init__(self,
                 split='train', # train <= 2022, val == 2023
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
                 leo_val: bool = False,
                 leo_val_envs: Optional[set] = None,
                 leo_val_fraction: float = 0.15,
                 leo_seed: int = 42,
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
            leo_val (bool): if True, use Leave-Environment-Out validation
            leo_val_envs (Optional[set]): pre-computed set of val environments (from train split)
            leo_val_fraction (float): fraction of environments to hold out for LEO val
            leo_seed (int): random seed for LEO environment selection
        """
        super().__init__()
        self.split = split
        self.data_path = data_path
        self.residual_flag = residual
        self.leo_val = leo_val
        self.scale_targets = scale_targets
        self.g_input_type = str(g_input_type).strip().lower()
        self.env_categorical_mode = normalize_env_categorical_mode(env_categorical_mode)
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
        # LEO validation: hold out entire environments (not years)
        if leo_val and split in ("train", "val"):
            # Compute or use provided LEO val environments
            if leo_val_envs is None:
                # First call (train split) - compute the held-out environments
                leo_val_envs = compute_leo_val_envs(
                    x_raw, test_year=2024, val_fraction=leo_val_fraction, seed=leo_seed
                )
            self.leo_val_envs = leo_val_envs
            
            # Filter to years before test (2014-2023)
            pre_test_mask = x_raw['Year'] < 2024
            env_in_val = x_raw['Env'].isin(leo_val_envs)
            
            if split == "train":
                # Train: all pre-2024 data EXCEPT held-out environments
                keep_mask = pre_test_mask & ~env_in_val
            else:  # val
                # Val: only the held-out environments (from any year < 2024)
                keep_mask = pre_test_mask & env_in_val
        else:
            self.leo_val_envs = None
            if split == "train":
                cutoff = 2022 if train_year_max is None else train_year_max
                keep_mask = x_raw['Year'] <= cutoff
            elif split == "val":
                which = 2023 if val_year is None else val_year
                keep_mask = x_raw['Year'] == which
            else: # 'test', 'sub'
                keep_mask = x_raw['Year'] >= 2024
        
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
        hybrid_names = self.meta['id'].astype(str).apply(
            lambda x: x.split('-', 1)[1] if '-' in x else x
        )
        self.hybrid_codes = pd.Categorical(hybrid_names)
        self.hybrid_id_tensor = torch.tensor(self.hybrid_codes.codes, dtype=torch.long)

        ### BUILD FT. MATRICES ###
        # drop metadata and accidental columns
        feature_df = x_filt.drop(columns=['id', 'Env', 'Year', 'Hybrid', 'Yield_Mg_ha'], errors='ignore')

        # sanity
        if feature_df.shape[1] < N_ENV:
            raise ValueError(f"Feature matrix has fewer columns ({feature_df.shape[1]}) than N_ENV ({N_ENV})."
                             "Check your X_* file layout.")
        
        # genotype features (all but last N_ENV)
        g_block = feature_df.iloc[:, :-N_ENV].astype(np.float32)
        e_block = feature_df.iloc[:, -N_ENV:]

        e_block = preprocess_env_features(e_block, self.env_categorical_mode)
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
