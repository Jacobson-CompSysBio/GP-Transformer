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

class GxEDataset(Dataset):

    def __init__(self,
                 tokenizer, 
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
        self.x_data = pd.read_csv(x_path, index_col=0).reset_index() # reset index col
        self.y_data = pd.read_csv(y_path, index_col=0).reset_index()

        # first 2240 features are genotype data
        self.g_data = self.x_data.iloc[:, :2240] # don't need first column

        # last 2240 features are lat/long and EC data
        self.e_data = self.x_data.iloc[:, 2240:] 

        # get tokenizer ready
        self.tokenizer = tokenizer

    def __len__(self):
        # return length (number of rows) in dataset
        return len(self.y_data)
    
    def __getitem__(self, index):
        """
        Parameters
            index (int): index to return data from
        Returns:
            obs (dict): dict of 'tokens', 'attention_mask', 'ec_data' (all x values), and 'target' (y value)
        """

        # get genotype data
        inputs = self.tokenizer(self.g_data[index], return_tensors="pt")
        tokens = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # get env data
        env_data = torch.tensor(self.e_data[index].values, dtype=torch.float32)

        obs = {'tokens': tokens, 
               'attention_mask': attention_mask, 
               'ec_data': env_data,
               'target': torch.tensor(self.y_data[index].values, dtype=torch.float32)}
        
        return obs 

# only genotype data
class GDataset(Dataset):

    def __init__(self,
                 tokenizer, 
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
        self.x_data = pd.read_csv(x_path, index_col=0).reset_index() # reset index col
        self.y_data = pd.read_csv(y_path, index_col=0).reset_index()

        # first 2240 features are genotype data
        self.g_data = self.x_data.iloc[:, :2240]

        # get tokenizer ready
        self.tokenizer = tokenizer

    def __len__(self):
        # return length (number of rows) in dataset
        return len(self.y_data)
    
    def __getitem__(self, index):
        """
        Parameters
            index (int): index to return data from
        Returns:
            obs (dict): dict of 'tokens', 'attention_mask' (both x values), and 'target' (y value)
        """

        # get genotype data
        inputs = self.tokenizer(self.g_data[index], return_tensors="pt")
        tokens = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]


        obs = {'tokens': tokens,
               'attention_mask': attention_mask, 
               'target': torch.tensor(self.y_data[index].values, dtype=torch.float32)}
        
        return obs 

# only env data
class EDataset(Dataset):

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
        self.x_data = pd.read_csv(x_path, index_col=0).reset_index() # reset index col
        self.y_data = pd.read_csv(y_path, index_col=0).reset_index()

        # last 2240 features are lat/long and EC data
        self.e_data = self.x_data.iloc[:, 2240:] 

    def __len__(self):
        # return length (number of rows) in dataset
        return len(self.y_data)
    
    def __getitem__(self, index):
        """
        Parameters
            index (int): index to return data from
        Returns:
            obs (dict): dict of 'ec_data' (x values), and 'target' (y value)
        """

        # get env data
        env_data = torch.tensor(self.e_data[index].values, dtype=torch.float32)

        obs = {'ec_data': env_data,
               'target': torch.tensor(self.y_data[index].values, dtype=torch.float32)}
        
        return obs 