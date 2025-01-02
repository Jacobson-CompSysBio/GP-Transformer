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
                 data_path, 
                 tokenizer):
        """
        Parameters:
            data_path (str): dataset to use 
            tokenizer (hf tokenizer): tokenizer to use for markers
        """

        # NOTE: not sure what our input data will look like yet... 
        # probably want to make it a numpy array or tensor or something
        self.data = pd.read_csv(data_path)

        # first 2240 features are genotype data
        self.g_data = self.data.iloc[:, :2240]

        # last 2240 features are lat/long and EC data
        self.e_data = self.data.iloc[:, 2240:] 

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
        env_data = torch.tensor(self.e_data[index].values, dtype=torch.float32)
        
        return tokens, attention_mask, env_data

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