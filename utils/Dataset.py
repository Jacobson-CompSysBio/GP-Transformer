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
        return len(self.data)
    
    def __getitem__(self, index):
        """
        Parameters
            index (int): index to return data from
        Returns:
            tokens (torch.tensor): tokenized SNP sequence
            attention_mask (torch.tensor): attention mask for tokenized SNP sequence
            env_data (torch.tensor): environmental data observation
        """

        # get genotype data
        inputs = self.tokenizer(self.g_data[index], return_tensors="pt")
        tokens = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # get env data
        env_data = torch.tensor(self.e_data.values, dtype=torch.float32)

        inp = {'tokens': tokens, 'attention_mask': attention_mask, 'env_data': env_data}
        target = {'target': torch.tensor(self.y_data[index].values, dtype=torch.float32)}
        
        return inp, target 

# only genotype data
class GDataset(Dataset):

    def __init__(self, 
                 data_path, 
                 tokenizer):
        """
        Parameters:
            data_path (str): dataset to use 
            tokenizer (hf tokenizer): tokenizer to use for markers
        """

        # NOTE: not sure what our input data will look like yet... 
        # probably want to make it a numpy array or tensor or something
        self.g_data = pd.read_csv(data_path).iloc[:, :2240]

        # get tokenizer ready
        self.tokenizer = tokenizer

    def __len__(self):
        # return length (number of rows) in dataset
        return len(self.g_data)
    
    def __getitem__(self, index):
        """
        Parameters
            index (int): index to return data from
        Returns:
            tokens (torch.tensor): tokenized SNP sequence
            attention_mask (torch.tensor): attention mask for tokenized SNP sequence
        """

        # get genotype data
        inputs = self.tokenizer(self.g_data[index], return_tensors="pt")
        tokens = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        return tokens, attention_mask
    
# only environmental data
class EDataset(Dataset):

    def __init__(self, 
                 data_path, 
                 tokenizer):
        """
        Parameters:
            data_path (str): dataset to use 
            tokenizer (hf tokenizer): tokenizer to use for markers
        """

        # NOTE: not sure what our input data will look like yet... 
        # probably want to make it a numpy array or tensor or something
        self.e_data = pd.read_csv(data_path).iloc[:, 2240:] 

    def __len__(self):
        # return length (number of rows) in dataset
        return len(self.e_data)
    
    def __getitem__(self, index):
        """
        Parameters
            index (int): index to return data from
        Returns:
            env_data (torch.tensor): environmental data observation
        """
        # get env data
        env_data = torch.tensor(self.e_data[index].values, dtype=torch.float32)
        
        return env_data