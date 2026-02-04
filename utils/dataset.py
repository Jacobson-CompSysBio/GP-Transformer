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

        ### BUILD FT. MATRICES ###
        # drop metadata and accidental columns
        feature_df = x_filt.drop(columns=['id', 'Env', 'Year', 'Hybrid', 'Yield_Mg_ha'], errors='ignore')

        # sanity
        if feature_df.shape[1] < N_ENV:
            raise ValueError(f"Feature matrix has fewer columns ({feature_df.shape[1]}) than N_ENV ({N_ENV})."
                             "Check your X_* file layout.")
        
        # genotype features (all but last N_ENV)
        g_block = feature_df.iloc[:, :-N_ENV]
        e_block = feature_df.iloc[:, -N_ENV:]

        ######################
        ### DROP COLS HERE ###
        ######################        
        e_block = e_block.drop(columns=["Irrigated", "Treatment", "Previous_Crop"])
        #e_block = e_block.iloc[:, :2] # select only first two cols, lat/lon
        

        # store for __getitem__
        self.g_data = (g_block * 2).astype('int64') # 0, 0.5, 1 --> 0, 1, 2
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
        tokens = torch.tensor(self.g_data.iloc[index, :].values, dtype=torch.long)
        env_data = torch.tensor(self.e_data.iloc[index, :].values, dtype=torch.float32)
        x = {'g_data': tokens, 'e_data': env_data}
        
        if self.split in ('sub', 'test'):
            row = self.y_data.iloc[index]
            # return only the columns that exist (Env and Yield_Mg_ha are typical)
            keep_cols = [c for c in ('Env', 'Hybrid', 'Yield_Mg_ha') if c in self.y_data.columns]
            y = {c: row[c] for c in keep_cols}
            return x, y

        # scaled targets if scale_targets=True or raw otherwise 
        y_total = torch.tensor([self.total_series.iloc[index]], dtype=torch.float32)
        env_id = self.env_id_tensor[index]

        if not self.residual_flag:
            return x, {"y": y_total, "env_id": env_id}

        # residual out
        y_env_mean = torch.tensor([self.env_mean.iloc[index]], dtype=torch.float32)
        y_residual = torch.tensor([self.residual.iloc[index]], dtype=torch.float32)
        targets = {
            'total': y_total,
            'ymean': y_env_mean,
            'resid': y_residual,
            'env_id': env_id
        }
        return x, targets