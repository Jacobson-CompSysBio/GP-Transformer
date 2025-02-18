# import necessary libraries
import time
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

class GxE_Dataset(Dataset):

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
            y_path = data_path + 'y_train.csv'
        elif split == 'val':
            x_path = data_path + 'X_val.csv'
            y_path = data_path + 'y_val.csv'
        elif split == 'sub':
            x_path = data_path + 'X_test.csv'
            y_path = data_path + 'X_test.csv'
        else:
            assert data_path != '../data/maize_data_2014-2023_vs_2024/', '2024 y_test.csv not available'
            x_path = data_path + 'X_test.csv'
            y_path = data_path + 'y_test.csv'
        
        # standard scaler init
        self.scaler = StandardScaler()

        # load data
        self.x_data = pd.read_csv(x_path, index_col=0).reset_index(drop=True) # reset index col
        if split == "sub":
            self.x_data = self.x_data.drop(columns=['Env', 'Hybrid', 'Yield_Mg_ha'])

        self.y_data = pd.read_csv(y_path, index_col=0).reset_index(drop=True)
        if split == "sub":
            self.y_data = self.y_data[['Env', 'Hybrid', 'Yield_Mg_ha']]

        # first 2240 features are genotype data
        self.g_data = self.x_data.iloc[:, :-374] * 2 # make these ints

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
            obs (dict): dict of 'g_data', 'e_data' (x values), and 'target' (y value)
        """

        # get genotype data
        tokens = torch.tensor(self.g_data.iloc[index, :].values, dtype=torch.long) # (add 2 to make everything positive)

        # get env data
        env_data = torch.tensor(self.e_data.iloc[index, :].values, dtype=torch.float32)

        x = {'g_data': tokens,  
             'e_data': env_data}
        
        if self.split == 'sub':
            y = {'Env': self.y_data.iloc[index, 0],
                 'Hybrid': self.y_data.iloc[index, 1],
                 'Yield_Mg_ha': self.y_data.iloc[index, 2]}
            return x, y

        y = torch.tensor(self.y_data.iloc[index].values, dtype=torch.float32)
        
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