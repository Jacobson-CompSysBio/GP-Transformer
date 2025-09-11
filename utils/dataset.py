# import necessary libraries
import time, sys
import re
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from transformers import (
    AutoModel, 
    AutoModelForSequenceClassification, 
    AutoModelForMaskedLM, 
    AutoTokenizer
    )

from transformers.models.bert.configuration_bert import BertConfig

# ----------------------------------------------------------------
# helper to get year from locations file
def _env_year_from_str(env_str: str) -> int:
    # expects strings like "DEH1_2014" -> 2014
    # robust to trailing spaces etc.
    m = re.search(r'(\d{4})$', str(env_str).strip())
    if m:
        return int(m.group(1))
    raise ValueError(f"Could not parse year from Env='{env_str}'")

# rolling GxE dataset
class GxE_Dataset(Dataset):

    def __init__(self,
                 split='train', # train <= 2022, val == 2023
                 data_path='data/maize_data_2014-2023_vs_2024/', # need to go up one level and then down to data directory
                 index_map_path='data/maize_data_2014-2023_vs_2024/location_2014_2023.csv',
                 scaler: StandardScaler | None = None,
                 train_year_max: int | None = None,
                 val_year: int | None = None
                 ):
        
        """
        Parameters:
            split (str): 'train' (2014-2022), 'val' (2023), 'test' (2024), or 'sub'
            data_path (str): path to data directory
            index_map_path (str): path to the INDEX -> Env mapping file
            scaler (StandardScaler|None): if None and split=='train', fit here; otherwise reuse passed scaler
        """

        self.split = split
        self.data_path = data_path

        # load data depending on split
        if split == 'train':
            x_path = data_path + 'X_train.csv'
            y_path = data_path + 'y_train.csv'
        elif split == 'val':
            x_path = data_path + 'X_train.csv'
            y_path = data_path + 'y_train.csv'
        elif split == 'sub':
            x_path = data_path + 'X_test.csv'
            y_path = data_path + 'X_test.csv'
        else:
            x_path = data_path + 'X_test.csv'
            y_path = data_path + 'y_test.csv'
        
        # load mapping function, compute year per row
        # INDEX col MUST align with row order after reset_index(drop=True)
        idx_map = pd.read_csv(index_map_path)
        if 'INDEX' not in idx_map.columns or 'Env' not in idx_map.columns:
            raise ValueError(f"index_map_path must contain INDEX, Env columns")
        idx_map['Year'] = idx_map['Env'].apply(_env_year_from_str)

        # get data for rolling max
        if split == "train":
            if train_year_max is not None:
                keep_mask = idx_map['Year'] <= train_year_max
            else:
                keep_mask = idx_map['Year'] <= 2022
        elif split == "val":
            if val_year is not None:
                keep_mask = idx_map['Year'] == val_year
            else:
                keep_mask = idx_map['Year'] == 2023
        elif split in ('test', 'sub'):
            keep_mask = idx_map['Year'] >= 2024
        else:
            raise ValueError(f"Invalid split='{split}'")

        # load data
        self.x_data = pd.read_csv(x_path, index_col=0).reset_index(drop=True) # reset index col
        if split == "sub":
            self.x_data = self.x_data.drop(columns=['Env', 'Hybrid', 'Yield_Mg_ha'], errors='ignore')
        self.y_data = pd.read_csv(y_path, index_col=0).reset_index(drop=True)

        # for explicit test, remove non-feature cols from X
        if split == "test":
            self.x_data = self.x_data.drop(columns=['Env', 'Hybrid', 'Yield_Mg_ha'], errors='ignore')

        if split == "sub":
            # keep metadata for submission
            self.y_data = self.y_data[['Env', 'Hybrid', 'Yield_Mg_ha']]
        
        # year filtering with index map
        # align lengths (sanity check)
        if len(idx_map) != len(self.x_data):
            raise ValueError(f"Length mismatch: idx_map={len(idx_map)}, x_data={len(self.x_data)}")

        if split == "train":
            keep_mask = idx_map['Year'] <= 2022
        elif split == "val":
            keep_mask = idx_map['Year'] == 2023
        elif split in ('test', 'sub'):
            keep_mask = idx_map['Year'] >= 2024
        else:
            raise ValueError(f"Invalid split='{split}'")
        
        self.x_data = self.x_data.loc[keep_mask.values].reset_index(drop=True)
        self.y_data = self.y_data.loc[keep_mask.values].reset_index(drop=True)
        self.idx_map = idx_map.loc[keep_mask.values].reset_index(drop=True)

        # feature partitioning and scaling
        # first 2240 are genotypes, last 374 are lat/lon and ECs
        self.scaler = scaler if scaler is not None else StandardScaler()

        # genotype features
        self.g_data = (self.x_data.iloc[:, :-374] * 2).astype('int64')

        # env features
        self.e_cols = list(self.x_data.columns[-374:])
        e_block = self.x_data[self.e_cols].copy()

        # fit scaler ONLY on train to avoid data leakage
        if scaler is None:
            if split != 'train':
                raise ValueError("For val/test/sub split you must pass a scaler")
            self.e_data = pd.DataFrame(self.scaler.fit_transform(e_block), columns=self.e_cols)
        else:
            self.e_data = pd.DataFrame(self.scaler.transform(e_block), columns=self.e_cols)

    def __len__(self):
        # return length (number of rows) in dataset
        return len(self.x_data)
    
    def __getitem__(self, index: int):

        """
        Parameters
            index (int): index to return data from
        Returns:
            obs (dict): dict of 'g_data', 'e_data' (x values), and 'target' (y value)
        """

        # get genotype data
        tokens = torch.tensor(self.g_data.iloc[index, :].values, dtype=torch.long)

        # get env data
        env_data = torch.tensor(self.e_data.iloc[index, :].values, dtype=torch.float32)

        x = {'g_data': tokens,  
             'e_data': env_data}
        
        if self.split in ('sub', 'test'):
            y = {'Env': self.y_data.iloc[index, 0],
                 'Hybrid': self.y_data.iloc[index, 1],
                 'Yield_Mg_ha': self.y_data.iloc[index, 2]}
            return x, y

        # regression target
        y = torch.tensor(self.y_data.iloc[index], dtype=torch.float32) 
        return x, y

# only genotype data
class G_Dataset(Dataset):

    def __init__(self,
                 split='train',
                 data_path='../data/maize_data_2014-2023_vs_2024/'
                 ):
        """
        Parameters:
            split (str): train, val, or test
            data_path (str): path to data
        """
        self.split = split
        self.data_path = data_path

        # load data depending on split
        if split == 'train':
            x_path = data_path + 'X_train.csv'
            y_path = data_path +'y_train.csv'
        elif split == 'val':
            x_path = data_path + 'X_val.csv'
            y_path = data_path + 'y_val.csv'
        elif split == 'sub':
            x_path = data_path + 'X_test.csv'
        else:
            assert data_path != '../data/maize_data_2014-2023_vs_2024/', '2024 y_test.csv not available'
            x_path = data_path + 'X_test.csv'
            y_path = data_path + 'y_test.csv'

        # load data
        self.x_data = pd.read_csv(x_path, index_col=0).reset_index(drop=True) # reset index col
        if split != 'sub':
            self.y_data = pd.read_csv(y_path, index_col=0).reset_index(drop=True)

        # first 2240 features are genotype data
        self.g_data = self.x_data.iloc[:, :-374] * 2 # make these ints

    def __len__(self):
        # return length (number of rows) in dataset
        return len(self.x_data)
    
    def __getitem__(self, index: int):
        """
        Parameters
            index (int): index to return data from
        Returns:
            obs (dict): dict of 'g_data' (x value), and 'target' (y value)
        """

        # get genotype data
        tokens = torch.tensor(self.g_data.iloc[index, :].values, dtype=torch.long) # (add 2 to make everything positive)
        x = {'g_data': tokens}
        
        if self.split == 'sub':
            return x

        y = torch.tensor(self.y_data.iloc[index].values, dtype=torch.float32)
        
        return x, y 

# only env data
class E_Dataset(Dataset):

    def __init__(self,
                 split='train',
                 data_path='../data/maize_data_2014-2023_vs_2024/'
                 ):
        """
        Parameters:
            split (str): train, val, or test
            data_path (str): path to data
        """
        self.split = split
        self.data_path = data_path

        # load data depending on split
        if split == 'train':
            x_path = data_path + 'X_train.csv'
            y_path = data_path +'y_train.csv'
        elif split == 'val':
            x_path = data_path + 'X_val.csv'
            y_path = data_path + 'y_val.csv'
        elif split == 'sub':
            x_path = data_path + 'X_test.csv'
        else:
            assert data_path != '../data/maize_data_2014-2023_vs_2024/', '2024 y_test.csv not available'
            x_path = data_path + 'X_test.csv'
            y_path = data_path + 'y_test.csv'

        # standard scaler init
        self.scaler = StandardScaler()

        # load data
        self.x_data = pd.read_csv(x_path, index_col=0).reset_index(drop=True) # reset index col
        if split != 'sub':
            self.y_data = pd.read_csv(y_path, index_col=0).reset_index(drop=True)

        # last 2240 features are lat/long and EC data
        self.e_data = self.x_data.iloc[:, -374:]
        cols = self.e_data.columns
        self.e_data = pd.DataFrame(self.scaler.fit_transform(self.e_data), columns=cols) 

    def __len__(self):
        # return length (number of rows) in dataset
        return len(self.x_data)
    
    def __getitem__(self, index: int):
        """
        Parameters
            index (int): index to return data from
        Returns:
            obs (dict): dict of 'e_data' (x values), and 'target' (y value)
        """

        # get env data
        env_data = torch.tensor(self.e_data.iloc[index, :].values, dtype=torch.float32)

        x = {'e_data': env_data}

        if self.split == 'sub':
            return x

        y = torch.tensor(self.y_data.iloc[index].values, dtype=torch.float32)
        
        return x, y