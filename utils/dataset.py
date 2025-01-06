# import necessary libraries
import time
import torch
import torch.nn.functional as F
import pandas as pd

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoModel, 
    AutoModelForSequenceClassification, 
    AutoModelForMaskedLM, 
    AutoTokenizer
    )

from transformers.models.bert.configuration_bert import BertConfig

# ----------------------------------------------------------------

class GxE_Dataset(Dataset):

    def __init__(self, 
                 split='train'
                 ):
        """
        Parameters:
            split (str): train, val, or test
        """

        # load data depending on split
        if split == 'train':
            x_path = '../data/position_ec_raw_genotype/X_train.csv'
            y_path = '../data/position_ec_raw_genotype/y_train.csv'
        elif split == 'val':
            x_path = '../data/position_ec_raw_genotype/X_val.csv'
            y_path = '../data/position_ec_raw_genotype/y_val.csv'
        else:
            x_path = '../data/position_ec_raw_genotype/X_test.csv'
            y_path = '../data/position_ec_raw_genotype/y_test.csv'

        # load data
        self.x_data = pd.read_csv(x_path, index_col=0).reset_index(drop=True) # reset index col
        self.y_data = pd.read_csv(y_path, index_col=0).reset_index(drop=True)

        # first 2240 features are genotype data
        self.g_data = self.x_data.iloc[:, :2240]

        # last 2240 features are lat/long and EC data
        self.e_data = self.x_data.iloc[:, 2240:] 

    def __len__(self):
        # return length (number of rows) in dataset
        return len(self.y_data)
    
    def __getitem__(self, index: int):

        """
        Parameters
            index (int): index to return data from
        Returns:
            obs (dict): dict of 'g_data', 'e_data' (x values), and 'target' (y value)
        """

        # get genotype data
        tokens = torch.tensor(self.g_data.iloc[index, :].values, dtype=torch.long) + 2 # (add 2 to make everything positive)

        # get env data
        env_data = torch.tensor(self.e_data.iloc[index, :].values, dtype=torch.float32)

        x = {'g_data': tokens,  
             'e_data': env_data}
        y = torch.tensor(self.y_data.iloc[index].values, dtype=torch.float32)
        
        return x, y

# only genotype data
class G_Dataset(Dataset):

    def __init__(self,
                 tokenizer, 
                 split='train'
                 ):
        """
        Parameters:
            split (str): train, val, or test
        """

        # load data depending on split
        if split == 'train':
            x_path = '../data/position_ec_raw_genotype/X_train.csv'
            y_path = '../data/position_ec_raw_genotype/y_train.csv'
        elif split == 'val':
            x_path = '../data/position_ec_raw_genotype/X_val.csv'
            y_path = '../data/position_ec_raw_genotype/y_val.csv'
        else:
            x_path = '../data/position_ec_raw_genotype/X_test.csv'
            y_path = '../data/position_ec_raw_genotype/y_test.csv'

        # load data
        self.x_data = pd.read_csv(x_path, index_col=0).reset_index(drop=True) # reset index col
        self.y_data = pd.read_csv(y_path, index_col=0).reset_index(drop=True)

        # first 2240 features are genotype data
        self.g_data = self.x_data.iloc[:, :2240]

        # get tokenizer ready
        self.tokenizer = tokenizer

    def __len__(self):
        # return length (number of rows) in dataset
        return len(self.y_data)
    
    def __getitem__(self, index: int):
        """
        Parameters
            index (int): index to return data from
        Returns:
            obs (dict): dict of 'g_data' (x value), and 'target' (y value)
        """

        # get genotype data
        tokens = torch.tensor(self.g_data.iloc[index, :].values, dtype=torch.long) + 2 # (add 2 to make everything positive)
        x = {'g_data': tokens}
        
        y = torch.tensor(self.y_data.iloc[index].values, dtype=torch.float32)
        
        return x, y 

# only env data
class E_Dataset(Dataset):

    def __init__(self,
                 split='train'
                 ):
        """
        Parameters:
            split (str): train, val, or test
            tokenizer (hf tokenizer): tokenizer to use for markers
        """

        # load data depending on split
        if split == 'train':
            x_path = '../data/position_ec_raw_genotype/X_train.csv'
            y_path = '../data/position_ec_raw_genotype/y_train.csv'
        elif split == 'val':
            x_path = '../data/position_ec_raw_genotype/X_val.csv'
            y_path = '../data/position_ec_raw_genotype/y_val.csv'
        else:
            x_path = '../data/position_ec_raw_genotype/X_test.csv'
            y_path = '../data/position_ec_raw_genotype/y_test.csv'

        # load data
        self.x_data = pd.read_csv(x_path, index_col=0).reset_index(drop=True) # reset index col
        self.y_data = pd.read_csv(y_path, index_col=0).reset_index(drop=True)

        # last 2240 features are lat/long and EC data
        self.e_data = self.x_data.iloc[:, 2240:] 

    def __len__(self):
        # return length (number of rows) in dataset
        return len(self.y_data)
    
    def __getitem__(self, index: int):
        """
        Parameters
            index (int): index to return data from
        Returns:
            obs (dict): dict of 'ec_data' (x values), and 'target' (y value)
        """

        # get env data
        env_data = torch.tensor(self.e_data.iloc[index, :].values, dtype=torch.float32)

        x = {'e_data': env_data}

        y = torch.tensor(self.y_data.iloc[index].values, dtype=torch.float32)
        
        return x, y